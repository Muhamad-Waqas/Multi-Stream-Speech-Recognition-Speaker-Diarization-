"""Transcript -> TXT / SRT files and an in-memory ZIP."""
from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, List

from .alignment import Turn


def fmt_time(sec: float, srt: bool = False) -> str:
    ms_total = int(round(max(sec, 0.0) * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}.{ms // 10:02d}"


def to_txt(turns: List[Turn]) -> str:
    return "".join(f"[{fmt_time(t.start)} - {fmt_time(t.end)}] {t.speaker}: {t.text}\n" for t in turns)


def to_srt(turns: List[Turn]) -> str:
    return "".join(f"{i}\n{fmt_time(t.start, True)} --> {fmt_time(t.end, True)}\n[{t.speaker}] {t.text}\n\n"
                   for i, t in enumerate(turns, 1))


def write_files(turns: List[Turn], out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext, render in (("txt", to_txt), ("srt", to_srt)):
        with open(os.path.join(out_dir, f"{name}.{ext}"), "w", encoding="utf-8", newline="") as f:
            f.write(render(turns))


def make_zip(results: Dict[str, List[Turn]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, turns in results.items():
            z.writestr(f"{name}.txt", to_txt(turns))
            z.writestr(f"{name}.srt", to_srt(turns))
    return buf.getvalue()
