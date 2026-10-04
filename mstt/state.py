"""Live, thread-safe progress of one job: per-file status + timings. The worker thread writes, the web page reads."""
from __future__ import annotations

import dataclasses
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# how far along a file is in each status (used for the progress bars)
STATUS_WEIGHT = {"queued": 0.0, "extracting": 0.08, "ready": 0.2, "transcribing": 0.35,
                 "diarizing": 0.65, "aligning": 0.9, "done": 1.0, "failed": 1.0}


@dataclass
class FileState:
    name: str
    path: str = ""
    size_mb: float = 0.0
    status: str = "queued"            # queued extracting ready transcribing diarizing aligning done failed
    error: str = ""
    audio_sec: float = 0.0            # length of the audio
    t_extract: float = 0.0            # seconds: ffmpeg audio extraction
    t_asr: float = 0.0                # seconds: this file's share of the speech-to-text GPU time
    t_diar: float = 0.0               # seconds: this file's share of the speaker-detection GPU time
    t_post: float = 0.0               # seconds: aligning words to speakers + saving
    finished_at: Optional[float] = None   # seconds since the job started when this file was finished
    batch: int = 0
    speakers: int = 0
    words: int = 0
    turns: int = 0

    @property
    def t_total(self) -> float:
        return self.t_extract + self.t_asr + self.t_diar + self.t_post

    @property
    def speed(self) -> float:
        """Audio seconds produced per second of processing (x real-time)."""
        return self.audio_sec / self.t_total if self.t_total > 0 and self.audio_sec > 0 else 0.0

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed")


class JobState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.files: Dict[str, FileState] = {}
        self.batches: List[dict] = []
        self.phase = "Idle"
        self.started: Optional[float] = None
        self.ended: Optional[float] = None
        self.batches_total = 0
        self.batches_done = 0
        self._cancel = threading.Event()

    # ------------------------------------------------------------ writing
    def begin(self, items: Sequence[Tuple[str, str]]) -> None:
        """items = [(display name, path), ...]"""
        with self._lock:
            self.files = {}
            for name, path in items:
                try:
                    size = os.path.getsize(path) / 2 ** 20
                except OSError:
                    size = 0.0
                self.files[name] = FileState(name=name, path=path, size_mb=size)
            self.batches, self.batches_total, self.batches_done = [], 0, 0
            self.started, self.ended = time.time(), None
            self.phase = "Starting ..."

    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.ended or time.time()) - self.started

    def update(self, name: str, **kw) -> None:
        with self._lock:
            f = self.files[name]
            for k, v in kw.items():
                setattr(f, k, v)

    def fail(self, name: str, error: str, **kw) -> None:
        with self._lock:
            if self.files[name].status != "failed":
                self.update(name, status="failed", error=error, finished_at=self.elapsed(), **kw)

    def finish(self, name: str, **kw) -> None:
        self.update(name, status="done", finished_at=self.elapsed(), **kw)

    def add_batch(self, info: dict) -> None:
        with self._lock:
            self.batches.append(info)
            self.batches_done = len(self.batches)

    def set_batches(self, total: int, done: int) -> None:
        with self._lock:
            self.batches_total, self.batches_done = total, done

    def set_phase(self, text: str) -> None:
        self.phase = text

    def end(self) -> None:
        with self._lock:
            self.ended = time.time()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------ reading
    def snapshot(self) -> List[FileState]:
        with self._lock:
            return [dataclasses.replace(f) for f in self.files.values()]

    def batch_list(self) -> List[dict]:
        with self._lock:
            return [dict(b) for b in self.batches]

    def progress(self) -> float:
        with self._lock:
            if not self.files:
                return 0.0
            return sum(STATUS_WEIGHT.get(f.status, 0.0) for f in self.files.values()) / len(self.files)
