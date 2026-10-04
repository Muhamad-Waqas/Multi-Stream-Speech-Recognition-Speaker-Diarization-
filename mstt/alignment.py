"""Words (from Parakeet) + speaker segments (from Sortformer) -> speaker-labelled turns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str


def parse_segment(seg) -> SpeakerSegment:
    """Sortformer returns strings like '0.080 3.600 speaker_0'."""
    if isinstance(seg, str):
        parts = seg.strip().split()
        if len(parts) >= 3:
            return SpeakerSegment(speaker=" ".join(parts[2:]), start=float(parts[0]), end=float(parts[1]))
    elif isinstance(seg, (tuple, list)) and len(seg) >= 3:
        return SpeakerSegment(speaker=str(seg[2]), start=float(seg[0]), end=float(seg[1]))
    raise ValueError(f"Could not understand diarization segment: {type(seg)} -> {seg}")


def words_from_hyp(hyp) -> List[Word]:
    ts = getattr(hyp, "timestamp", None) or {}
    return [Word(text=w["word"], start=float(w["start"]), end=float(w["end"])) for w in ts.get("word", [])]


def assign_words_to_speakers(words: List[Word], segments: List[SpeakerSegment]) -> List[Tuple[Word, str]]:
    """Each word goes to the speaker segment it overlaps the most."""
    words = sorted(words, key=lambda w: w.start)
    if not words:
        return []
    if not segments:
        return [(w, "UNKNOWN") for w in words]

    seg_s = np.array([s.start for s in segments])
    seg_e = np.array([s.end for s in segments])
    spk = [s.speaker for s in segments]
    w_s = np.array([w.start for w in words])
    w_e = np.array([w.end for w in words])

    assigned, prev, chunk = [], None, 2048
    for i in range(0, len(words), chunk):
        ov = np.minimum(w_e[i:i + chunk, None], seg_e[None, :]) - np.maximum(w_s[i:i + chunk, None], seg_s[None, :])
        best = ov.argmax(axis=1)
        best_ov = ov[np.arange(len(best)), best]
        for j in range(len(best)):
            speaker = spk[best[j]] if best_ov[j] > 0 else (prev or "UNKNOWN")
            assigned.append((words[i + j], speaker))
            prev = speaker
    return assigned


def merge_into_turns(assigned: List[Tuple[Word, str]], silence_gap: float = 0.8) -> List[Turn]:
    """Consecutive words by the same speaker form one turn; a speaker change or a long pause starts a new one."""
    turns: List[Turn] = []
    cur: List[Word] = []
    cur_spk, last_end = None, None
    for word, spk in assigned:
        if cur and (spk != cur_spk or word.start - last_end > silence_gap):
            turns.append(Turn(cur_spk, cur[0].start, cur[-1].end, " ".join(w.text for w in cur)))
            cur = []
        cur.append(word)
        cur_spk, last_end = spk, word.end
    if cur:
        turns.append(Turn(cur_spk, cur[0].start, cur[-1].end, " ".join(w.text for w in cur)))
    return turns
