"""
Live GPU / CPU / RAM numbers for the dashboard.

GPU: NVML (package `nvidia-ml-py`) when installed, otherwise the `nvidia-smi` program that ships with the NVIDIA driver.
CPU / RAM: `psutil`. Anything that is not available simply shows as "n/a" - monitoring never breaks a job.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

try:
    import psutil
except Exception:                                   # pragma: no cover
    psutil = None

KEYS = ("gpu_util", "vram_used_mb", "vram_total_mb", "gpu_temp", "gpu_power", "cpu", "ram_pct", "ram_used_gb")


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class GpuReader:
    """Reads GPU number `index` through NVML or nvidia-smi."""

    def __init__(self, index: int = 0):
        self.index = index
        self._nv = self._h = self._smi = None
        self.name, self.total_mb = "", None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._nv = pynvml
            n = pynvml.nvmlDeviceGetName(self._h)
            self.name = n.decode() if isinstance(n, bytes) else str(n)
            self.total_mb = pynvml.nvmlDeviceGetMemoryInfo(self._h).total / 2 ** 20
        except Exception:
            self._nv = self._h = None
            self._smi = shutil.which("nvidia-smi")
            if self._smi:
                row = self._query_smi()
                if row:
                    self.name, self.total_mb = row["name"], row["vram_total_mb"]
                else:
                    self._smi = None

    @property
    def available(self) -> bool:
        return bool(self._nv or self._smi)

    @property
    def uses_smi(self) -> bool:
        return bool(self._smi) and not self._nv

    def _query_smi(self) -> Optional[dict]:
        cols = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        try:
            r = subprocess.run([self._smi, f"--id={self.index}", f"--query-gpu={cols}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
        except Exception:
            return None
        if r.returncode != 0 or len(parts) < 6:
            return None
        return {"name": parts[0], "gpu_util": _num(parts[1]), "vram_used_mb": _num(parts[2]),
                "vram_total_mb": _num(parts[3]), "gpu_temp": _num(parts[4]), "gpu_power": _num(parts[5])}

    def read(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {k: None for k in ("gpu_util", "vram_used_mb", "vram_total_mb", "gpu_temp", "gpu_power")}
        if self._nv:
            nv, h = self._nv, self._h
            try:
                out["gpu_util"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            try:
                m = nv.nvmlDeviceGetMemoryInfo(h)
                out["vram_used_mb"], out["vram_total_mb"] = m.used / 2 ** 20, m.total / 2 ** 20
            except Exception:
                pass
            try:
                out["gpu_temp"] = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                out["gpu_power"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
        elif self._smi:
            row = self._query_smi() or {}
            for k in out:
                out[k] = row.get(k)
        return out


def device_info() -> Dict:
    """Static facts about this computer, for the sidebar / header."""
    g = GpuReader()
    info = {"gpu": g.name if g.available else "", "vram_gb": (g.total_mb or 0) / 1024.0 if g.available else 0.0,
            "cpu_cores": os.cpu_count() or 0, "ram_gb": 0.0}
    if psutil:
        try:
            info["ram_gb"] = psutil.virtual_memory().total / 2 ** 30
        except Exception:
            pass
    return info


class SystemMonitor:
    """Samples in a background thread; `history()` feeds the live charts, `summary()` gives averages and peaks."""

    def __init__(self, interval: float = 0.5, max_points: int = 3000, sampler: Optional[Callable[[], dict]] = None):
        self.interval = interval
        self._sampler = sampler
        self._gpu: Optional[GpuReader] = None
        self._hist: Deque[dict] = deque(maxlen=max_points)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._peak: Dict[str, float] = {}
        self._sum: Dict[str, float] = {}
        self._n: Dict[str, int] = {}

    # ------------------------------------------------------------ sampling
    def _default_sample(self) -> dict:
        s = dict(self._gpu.read()) if self._gpu and self._gpu.available else {}
        if psutil:
            try:
                s["cpu"] = psutil.cpu_percent(None)
                vm = psutil.virtual_memory()
                s["ram_pct"], s["ram_used_gb"] = vm.percent, (vm.total - vm.available) / 2 ** 30
            except Exception:
                pass
        return s

    def _take(self) -> None:
        try:
            s = (self._sampler or self._default_sample)()
        except Exception:
            s = {}
        s = {k: s.get(k) for k in KEYS}
        s["t"] = round(time.time() - self._t0, 2)
        with self._lock:
            self._hist.append(s)
            for k in KEYS:
                v = s[k]
                if v is None:
                    continue
                self._peak[k] = max(self._peak.get(k, v), v)
                self._sum[k] = self._sum.get(k, 0.0) + v
                self._n[k] = self._n.get(k, 0) + 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._take()

    def start(self) -> "SystemMonitor":
        if self._thread:
            return self
        if self._sampler is None:
            self._gpu = GpuReader()
            if self._gpu.uses_smi:                      # spawning nvidia-smi is slower than NVML
                self.interval = max(self.interval, 1.0)
            if psutil:
                psutil.cpu_percent(None)                # the first call always returns 0.0
        self._t0 = time.time()
        self._take()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mstt-monitor")
        self._thread.start()
        return self

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=3)
        self._thread = None
        self._take()

    # ------------------------------------------------------------ reading
    def history(self) -> List[dict]:
        with self._lock:
            return list(self._hist)

    def latest(self) -> dict:
        with self._lock:
            return dict(self._hist[-1]) if self._hist else {k: None for k in KEYS}

    def summary(self) -> Dict[str, Optional[float]]:
        with self._lock:
            out: Dict[str, Optional[float]] = {}
            for k in KEYS:
                out[f"{k}_peak"] = self._peak.get(k)
                out[f"{k}_avg"] = self._sum[k] / self._n[k] if self._n.get(k) else None
            return out

    @property
    def has_gpu(self) -> bool:
        return bool(self._gpu and self._gpu.available) or any(h.get("gpu_util") is not None for h in self.history()[-3:])

    @property
    def gpu_name(self) -> str:
        return self._gpu.name if self._gpu and self._gpu.available else ""
