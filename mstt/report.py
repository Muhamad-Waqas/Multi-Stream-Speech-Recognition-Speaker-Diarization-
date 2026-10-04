"""Numbers for the dashboard / CSV report, and per-speaker statistics."""
from __future__ import annotations

import csv
import io
from typing import Dict, List

from .alignment import Turn
from .state import STATUS_WEIGHT, FileState

STATUS_LABEL = {"queued": "⏳ Queued", "extracting": "🎞️ Extracting audio", "ready": "📥 Ready for GPU",
                "transcribing": "🧠 Transcribing", "diarizing": "🗣️ Finding speakers", "aligning": "✍️ Aligning",
                "done": "✅ Done", "failed": "❌ Failed"}


def fmt_dur(sec: float) -> str:
    """75 -> '1:15',  3725 -> '1:02:05'"""
    sec = int(round(max(sec or 0.0, 0.0)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def table_rows(files: List[FileState]) -> List[dict]:
    """One dict per file with display-ready values (used by the live table)."""
    rows = []
    for i, f in enumerate(files, 1):
        rows.append({
            "#": i,
            "File": f.name,
            "Status": STATUS_LABEL.get(f.status, f.status),
            "Progress": round(100 * STATUS_WEIGHT.get(f.status, 0.0)),
            "Audio": fmt_dur(f.audio_sec) if f.audio_sec else "-",
            "Extract (s)": round(f.t_extract, 1),
            "Transcribe (s)": round(f.t_asr, 1),
            "Speakers (s)": round(f.t_diar, 1),
            "Total (s)": round(f.t_total, 1),
            "Speed (x real-time)": round(f.speed, 1),
            "Done at": fmt_dur(f.finished_at) if f.finished_at is not None else "-",
            "Speakers found": f.speakers if f.status == "done" else None,
            "Words": f.words if f.status == "done" else None,
            "Batch": f.batch or None,
            "Note": f.error if f.status == "failed" else "",
        })
    return rows


CSV_COLUMNS = ["file", "status", "audio_seconds", "extract_seconds", "transcribe_seconds", "speaker_detection_seconds",
               "align_save_seconds", "total_processing_seconds", "speed_x_realtime", "finished_at_seconds",
               "speakers_found", "words", "turns", "batch", "size_mb", "error"]


def to_csv(files: List[FileState]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for f in files:
        w.writerow([f.name, f.status, round(f.audio_sec, 2), round(f.t_extract, 2), round(f.t_asr, 2), round(f.t_diar, 2),
                    round(f.t_post, 2), round(f.t_total, 2), round(f.speed, 2),
                    "" if f.finished_at is None else round(f.finished_at, 2),
                    f.speakers, f.words, f.turns, f.batch, round(f.size_mb, 1), f.error])
    return buf.getvalue()


def speaker_stats(turns: List[Turn]) -> List[Dict]:
    """Talk time per speaker, e.g. [{'speaker': 'speaker_0', 'seconds': 31.2, 'share': 0.62, 'turns': 5, 'words': 90}]"""
    acc: Dict[str, Dict] = {}
    for t in turns:
        a = acc.setdefault(t.speaker, {"speaker": t.speaker, "seconds": 0.0, "turns": 0, "words": 0})
        a["seconds"] += max(t.end - t.start, 0.0)
        a["turns"] += 1
        a["words"] += len(t.text.split())
    total = sum(a["seconds"] for a in acc.values()) or 1.0
    out = sorted(acc.values(), key=lambda a: a["speaker"])
    for a in out:
        a["share"] = a["seconds"] / total
    return out
