"""
Run this if anything fails:   python check_setup.py
Tests each important package in its own process, so you can see exactly which one Windows (or anything else) blocks.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKS = [
    ("numpy", "import numpy"),
    ("streamlit", "import streamlit"),
    ("torch", "import torch; print('CUDA GPU:' , torch.cuda.is_available())"),
    ("numba  (the app has a built-in workaround if this one is blocked)", "import numba"),
    ("librosa  (needs numba -> uses the app's workaround)", "import librosa"),
    ("soundfile", "import soundfile"),
    ("psutil  (CPU / RAM numbers on the dashboard)", "import psutil"),
    ("pynvml  (GPU numbers on the dashboard; nvidia-smi is used if this one is missing)", "import pynvml; pynvml.nvmlInit(); print('GPU:', pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)))"),
    ("NeMo speech models  (uses the app's workaround)", "import nemo.collections.asr"),
]


ROOT = str(Path(__file__).resolve().parent)
# these packages import numba, so they are tested WITH the app's numba workaround switched on (like the real app)
WITH_FIX = f"import sys; sys.path.insert(0, {ROOT!r}); from mstt import compat; compat.fix_numba(); "


def run(code: str, with_fix: bool = False):
    r = subprocess.run([sys.executable, "-c", (WITH_FIX if with_fix else "") + code], capture_output=True, text=True)
    last = (r.stderr.strip().splitlines() or [""])[-1] if r.returncode else (r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode == 0, last


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})\n")
    bad = 0
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        print("  [WARN] NeMo supports Python 3.10 - 3.12; you have a different version.")
    from mstt import media
    ff = media.ffmpeg_path()
    ok = False
    if ff:
        try:
            ok = subprocess.run([ff, "-version"], capture_output=True).returncode == 0
        except OSError as e:                        # e.g. Windows blocking ffmpeg.exe
            ff = f"{ff}  ({e})"
    print(f"  [{'OK  ' if ok else 'FAIL'}] ffmpeg  {ff or '-> pip install imageio-ffmpeg  (or: winget install ffmpeg)'}")
    bad += not ok
    for label, code in CHECKS:
        ok, detail = run(code, with_fix=("librosa" in label or "NeMo" in label))
        optional = "dashboard" in label                     # only the live GPU/CPU numbers depend on these
        print(f"  [{'OK  ' if ok else ('WARN' if optional else 'FAIL')}] {label}  {detail if detail else ''}")
        bad += (not ok) and not optional
    if bad:
        print("\nA FAIL with 'Application Control policy has blocked this file' means Windows refuses to load that "
              "package's compiled file.\nFix: run the project inside WSL2 (see README.md), or ask the PC admin to allow it.")
    else:
        print("\nEverything loads. Start the app with:  streamlit run app.py")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
