"""Audio extraction: any video/audio file -> mono float32 at 16 kHz (ffmpeg, no temp files, video is ignored)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional

import numpy as np


class FFmpegNotFoundError(RuntimeError):
    pass


class DecodeError(RuntimeError):
    pass


def ffmpeg_path() -> Optional[str]:
    """System ffmpeg if present, otherwise the binary bundled with the `imageio-ffmpeg` pip package."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration(path: str) -> Optional[float]:
    """Length of a media file in seconds (ffmpeg only reads the header - very fast). None when unknown."""
    exe = ffmpeg_path()
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-nostdin", "-hide_banner", "-i", str(path)], capture_output=True, timeout=60)
    except Exception:
        return None
    m = _DURATION.search(r.stderr.decode(errors="ignore"))
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def decode_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    exe = ffmpeg_path()
    if not exe:
        raise FFmpegNotFoundError("ffmpeg not found. Run: pip install imageio-ffmpeg   (or: winget install ffmpeg)")
    cmd = [exe, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-f", "f32le", "pipe:1"]
    name = os.path.basename(path)
    try:
        r = subprocess.run(cmd, capture_output=True)
    except OSError as e:                              # e.g. Windows blocking the bundled ffmpeg.exe
        raise DecodeError(f"{name}: could not start ffmpeg ({e}). Try: winget install ffmpeg") from e
    if r.returncode != 0:
        err = r.stderr.decode(errors="ignore")
        if "does not contain any stream" in err:
            raise DecodeError(f"{name}: no audio track found in this file")
        raise DecodeError(f"{name}: could not read audio ({err.strip()[-200:]})")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()
