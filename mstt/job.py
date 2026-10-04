"""
Runs a transcription job in a BACKGROUND THREAD so the web page stays responsive: the page only reads
`job.view()` once a second. Clicking around (or even reloading the tab) can no longer interrupt a run.
"""
from __future__ import annotations

import shutil
import threading
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

from . import config, pipeline
from .monitor import SystemMonitor
from .state import FileState, JobState


@dataclass
class JobView:
    """A frozen copy of everything the dashboard shows."""
    files: List[FileState] = field(default_factory=list)
    batches: List[dict] = field(default_factory=list)
    phase: str = ""
    elapsed: float = 0.0
    running: bool = False
    progress: float = 0.0
    batches_done: int = 0
    batches_total: int = 0
    history: List[dict] = field(default_factory=list)       # GPU / CPU samples over time
    latest: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)             # peaks + averages of the samples
    has_gpu: bool = False
    gpu_name: str = ""
    out_dir: str = ""
    error: str = ""

    @property
    def n_total(self) -> int:
        return len(self.files)

    @property
    def n_done(self) -> int:
        return sum(f.status == "done" for f in self.files)

    @property
    def n_failed(self) -> int:
        return sum(f.status == "failed" for f in self.files)

    @property
    def audio_total(self) -> float:
        return sum(f.audio_sec for f in self.files)

    @property
    def audio_done(self) -> float:
        return sum(f.audio_sec for f in self.files if f.status == "done")

    @property
    def throughput(self) -> float:
        """Seconds of audio finished per second of wall-clock time."""
        return self.audio_done / self.elapsed if self.elapsed > 0 else 0.0


class Job:
    def __init__(self, engine, paths: List[str], out_dir: str, tmp_dir: Optional[str] = None,
                 batch_seconds: Optional[float] = None, silence_gap: Optional[float] = None,
                 monitor: Optional[SystemMonitor] = None):
        self.engine, self.paths, self.out_dir, self.tmp_dir = engine, list(paths), out_dir, tmp_dir
        self.batch_seconds, self.silence_gap = batch_seconds, silence_gap
        self.state = JobState()
        self.monitor = monitor or SystemMonitor(interval=config.MONITOR_INTERVAL)
        self.result: Optional[pipeline.JobResult] = None
        self.error = ""
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ control
    def start(self) -> "Job":
        self._thread = threading.Thread(target=self._main, daemon=True, name="mstt-job")
        self._thread.start()
        return self

    def cancel(self) -> None:
        self.state.cancel()

    def join(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    # ------------------------------------------------------------ worker
    def _main(self) -> None:
        try:
            self.monitor.start()
            self.result = pipeline.run(self.engine, self.paths, self.out_dir, state=self.state,
                                       batch_seconds=self.batch_seconds, silence_gap=self.silence_gap)
        except Exception as e:                                   # unexpected - never leave the page waiting forever
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            for f in self.state.snapshot():
                if not f.finished:
                    self.state.fail(f.name, self.error)
            self.state.end()
        finally:
            self.monitor.stop()
            if self.tmp_dir:
                shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self._done.set()

    # ------------------------------------------------------------ reading
    def view(self) -> JobView:
        hist = self.monitor.history()
        return JobView(
            files=self.state.snapshot(), batches=self.state.batch_list(), phase=self.state.phase,
            elapsed=self.state.elapsed(), running=not self.done, progress=self.state.progress(),
            batches_done=self.state.batches_done, batches_total=self.state.batches_total,
            history=hist, latest=self.monitor.latest(), summary=self.monitor.summary(),
            has_gpu=self.monitor.has_gpu, gpu_name=self.monitor.gpu_name, out_dir=self.out_dir, error=self.error)
