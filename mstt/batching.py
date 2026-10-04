"""Decides how files are grouped into GPU batches (pure functions, no GPU needed)."""
from __future__ import annotations

from typing import Hashable, List, Optional, Sequence, Tuple

from . import config


def plan_batches(items: Sequence[Tuple[Hashable, float]], max_files: int, max_seconds: float) -> List[List[Hashable]]:
    """
    items = [(key, seconds), ...]  ->  [[key, ...], ...]  longest files first.

    Files of similar length share a batch (little padding wasted). A batch costs
    (number of files) x (longest file) seconds; it is closed when that would exceed `max_seconds`
    or the batch already holds `max_files`. A file longer than the limit simply gets a batch of its own.
    """
    ordered = sorted(items, key=lambda kv: kv[1], reverse=True)
    batches: List[List[Hashable]] = []
    cur: List[Hashable] = []
    cur_max = 0.0
    for key, sec in ordered:
        if not cur:
            cur, cur_max = [key], sec
        elif len(cur) < max_files and (len(cur) + 1) * cur_max <= max_seconds:
            cur.append(key)
        else:
            batches.append(cur)
            cur, cur_max = [key], sec
    if cur:
        batches.append(cur)
    return batches


def auto_batch_seconds(vram_gb: Optional[float]) -> float:
    """Padded-audio-seconds limit for one batch, from the GPU memory (None / 0 = no GPU)."""
    if not vram_gb:
        return float(config.CPU_BATCH_SECONDS)
    return float(min(max(vram_gb * config.BATCH_SECONDS_PER_VRAM_GB, config.BATCH_SECONDS_MIN), config.BATCH_SECONDS_MAX))
