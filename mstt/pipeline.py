"""
The whole job for up to 30 files, as a small assembly line so the GPU never waits for the CPU:

  probe     read every file's length (header only, instant)
  extract   ffmpeg threads turn video/audio into 16 kHz audio, a couple of batches AHEAD of the GPU
  GPU       files sorted by length are grouped into batches (see batching.py); Parakeet + Sortformer run per batch
  finish    worker threads attach words to speakers and write TXT + SRT while the GPU already works on the next batch

Every step updates `JobState` (status + timings per file) so the web page can show live progress.
One bad file (corrupt, no audio) never stops the others.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import config, exporters, media, report
from .alignment import Turn, assign_words_to_speakers, merge_into_turns, parse_segment, words_from_hyp
from .batching import plan_batches
from .state import FileState, JobState

Progress = Callable[[float, str], None]


@dataclass
class JobResult:
    transcripts: Dict[str, List[Turn]] = field(default_factory=dict)    # name -> speaker turns (in input order)
    failed: Dict[str, str] = field(default_factory=dict)                # name -> reason
    seconds: float = 0.0
    out_dir: str = ""
    files: List[FileState] = field(default_factory=list)                # final per-file timings
    batches: List[dict] = field(default_factory=list)                   # per-batch GPU timings
    audio_seconds: float = 0.0                                          # audio successfully processed


def unique_names(paths: List[str]) -> Dict[str, str]:
    """file path -> display name (file name without extension; duplicates get _1, _2 ...)."""
    seen, out = {}, {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        n = seen.get(stem, 0)
        seen[stem] = n + 1
        out[p] = stem if n == 0 else f"{stem}_{n}"
    return out


def _estimate(path: str, probed: Optional[float]) -> float:
    if probed:
        return max(probed, 0.1)
    try:                                                   # unknown length: guess from file size (over-estimating is safe)
        return max(os.path.getsize(path) / 16000.0, 1.0)
    except OSError:
        return 1.0


def run(engine, paths: List[str], out_dir: str, progress: Optional[Progress] = None, state: Optional[JobState] = None,
        batch_seconds: Optional[float] = None, silence_gap: Optional[float] = None,
        decode_workers: Optional[int] = None) -> JobResult:
    say = progress or (lambda frac, text: None)
    if not paths:
        raise ValueError("No files to process.")
    if len(paths) > config.MAX_FILES:
        raise ValueError(f"Too many files: {len(paths)} (the limit is {config.MAX_FILES}).")
    silence_gap = config.SILENCE_GAP if silence_gap is None else silence_gap

    names = unique_names(paths)
    state = state or JobState()
    state.begin([(names[p], p) for p in paths])
    res = JobResult(out_dir=out_dir)
    t0 = state.started
    last = [0.0]

    def tell(text: str) -> None:
        state.set_phase(text)
        last[0] = max(last[0], min(state.progress(), 0.999))      # never goes backwards, 1.0 is reserved for "Done"
        say(last[0], text)

    workers = decode_workers or max(1, min(len(paths), os.cpu_count() or 2, config.DECODE_WORKERS))
    decode_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mstt-decode")
    post_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mstt-post")
    post_futures = []

    def decode(p: str):
        name = names[p]
        state.update(name, status="extracting")
        t = time.time()
        try:
            audio = media.decode_audio(p, config.SAMPLE_RATE)
            if len(audio) < config.SAMPLE_RATE * 0.2:
                raise media.DecodeError("no audio track, or audio shorter than 0.2 seconds")
        except Exception as e:
            state.fail(name, str(e)[-300:], t_extract=time.time() - t)
            raise
        state.update(name, status="ready", audio_sec=len(audio) / config.SAMPLE_RATE, t_extract=time.time() - t)
        return audio

    def post(name: str, hyp, seg_raw):
        state.update(name, status="aligning")
        t = time.time()
        try:
            words = words_from_hyp(hyp)
            segments = [parse_segment(s) for s in seg_raw]
            turns = merge_into_turns(assign_words_to_speakers(words, segments), silence_gap)
            exporters.write_files(turns, out_dir, name)
        except Exception as e:
            msg = f"post-processing error: {repr(e)[:250]}"
            state.fail(name, msg, t_post=time.time() - t)
            return name, None, msg
        state.finish(name, t_post=time.time() - t, speakers=len({t_.speaker for t_ in turns}),
                     words=len(words), turns=len(turns))
        return name, turns, None

    try:
        # 1 - lengths of all files (header only)
        tell(f"Reading the length of {len(paths)} file(s) ...")
        with ThreadPoolExecutor(max_workers=workers) as pp:
            probed = list(pp.map(media.probe_duration, paths))
        est = {p: _estimate(p, d) for p, d in zip(paths, probed)}
        for p in paths:
            state.update(names[p], audio_sec=est[p])

        budget = lambda: engine.effective_batch_seconds(batch_seconds)      # noqa: E731  (can shrink after an out-of-memory)
        remaining = list(paths)
        futures: Dict[str, object] = {}
        batch_no = 0

        def top_up() -> None:
            """Keep about PREFETCH_FACTOR batches of audio extracting ahead of the GPU (bounded RAM)."""
            limit = config.PREFETCH_FACTOR * budget()
            inflight = sum(est[p] for p in futures)
            for p in sorted(remaining, key=lambda q: -est[q]):
                if p in futures:
                    continue
                if inflight and inflight + est[p] > limit:
                    break
                futures[p] = decode_pool.submit(decode, p)
                inflight += est[p]

        while remaining:
            if state.cancelled:
                for p in remaining:
                    res.failed[names[p]] = "stopped by the user"
                    state.fail(names[p], "stopped by the user")
                break
            top_up()
            plan = plan_batches([(p, est[p]) for p in remaining], config.MAX_BATCH_FILES, budget())
            batch = plan[0]
            state.set_batches(total=batch_no + len(plan), done=batch_no)
            for p in batch:
                if p not in futures:
                    futures[p] = decode_pool.submit(decode, p)
            in_batch = set(batch)
            remaining = [p for p in remaining if p not in in_batch]

            # 2 - wait for this batch's audio
            tell(f"Batch {batch_no + 1}/{batch_no + len(plan)}: extracting audio ...")
            ok_names, audios = [], []
            for p in batch:
                try:
                    audio = futures.pop(p).result()
                except Exception as e:
                    res.failed[names[p]] = str(e)[-300:]
                    continue
                ok_names.append(names[p])
                audios.append(audio)
            top_up()                                            # start extracting upcoming files while the GPU works
            if not audios:
                continue

            # 3 - both models on this batch
            batch_no += 1
            for n in ok_names:
                state.update(n, status="transcribing", batch=batch_no)
            tell(f"Batch {batch_no}/{batch_no - 1 + len(plan)}: {len(audios)} file(s) on the "
                 f"{'GPU' if getattr(engine, 'device', 'cpu') == 'cuda' else 'CPU'} ...")

            def on_stage(stage: str, _names=tuple(ok_names)) -> None:
                for n in _names:
                    state.update(n, status=stage)

            total_sec = sum(len(a) for a in audios) / config.SAMPLE_RATE
            padded = len(audios) * max(len(a) for a in audios) / config.SAMPLE_RATE
            try:
                inf = engine.infer(audios, on_stage=on_stage)
            except Exception as e:
                for n in ok_names:
                    res.failed[n] = f"model error: {repr(e)[:250]}"
                    state.fail(n, res.failed[n])
                continue
            for n, a in zip(ok_names, audios):
                share = (len(a) / config.SAMPLE_RATE) / total_sec
                state.update(n, t_asr=inf.asr_sec * share, t_diar=inf.diar_sec * share)
            state.add_batch({"batch": batch_no, "files": len(audios), "audio_sec": round(total_sec, 1),
                             "padded_sec": round(padded, 1), "asr_sec": round(inf.asr_sec, 2),
                             "diar_sec": round(inf.diar_sec, 2), "peak_vram_mb": round(inf.peak_vram_mb)})
            state.set_batches(total=batch_no + len(plan) - 1, done=batch_no)

            # 4 - align + save in worker threads while the next batch is already running
            for n, hyp, seg_raw in zip(ok_names, inf.hyps, inf.segs):
                post_futures.append(post_pool.submit(post, n, hyp, seg_raw))
            del audios, inf

        tell("Saving transcripts ...")
        done: Dict[str, List[Turn]] = {}
        for fut in post_futures:
            name, turns, err = fut.result()
            if err:
                res.failed[name] = err
            else:
                done[name] = turns
        res.transcripts = {names[p]: done[names[p]] for p in paths if names[p] in done}      # input order
        res.failed = {names[p]: res.failed[names[p]] for p in paths if names[p] in res.failed}
    finally:
        decode_pool.shutdown(wait=True, cancel_futures=True)
        post_pool.shutdown(wait=True)

    state.end()
    res.seconds = time.time() - t0
    res.files = state.snapshot()
    res.batches = state.batch_list()
    res.audio_seconds = sum(f.audio_sec for f in res.files if f.status == "done")
    if res.transcripts:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "performance_report.csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write(report.to_csv(res.files))
    state.set_phase("Done")
    say(1.0, "Done")
    return res
