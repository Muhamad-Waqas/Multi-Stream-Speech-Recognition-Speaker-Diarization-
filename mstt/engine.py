"""
The two AI models, loaded once and run on a batch of files:
  * Parakeet-TDT  -> words with timestamps
  * Sortformer    -> who spoke when
If the GPU runs out of memory the batch is split in half and retried automatically, and the engine REMEMBERS the
limit it learned so later batches (and later runs) start with a size that fits.
"""
from __future__ import annotations

import contextlib
import gc
import logging
import threading
import time
from typing import Callable, List, NamedTuple, Optional

import numpy as np

from . import batching, compat, config

try:                                    # torch is optional here so the logic can be tested without it
    import torch
except Exception:                       # pragma: no cover
    torch = None

log = logging.getLogger(__name__)


def _oom_types() -> tuple:
    types_ = [MemoryError]
    if torch is not None and hasattr(torch, "cuda") and hasattr(torch.cuda, "OutOfMemoryError"):
        types_.append(torch.cuda.OutOfMemoryError)
    return tuple(types_)


OOM_ERRORS = _oom_types()


class InferResult(NamedTuple):
    hyps: list              # one Parakeet hypothesis per audio
    segs: list              # one list of speaker segments per audio
    asr_sec: float          # wall time of the speech-to-text step for the whole batch
    diar_sec: float         # wall time of the speaker step for the whole batch
    peak_vram_mb: float     # highest GPU memory PyTorch used during this batch (0 on CPU)


class Engine:
    def __init__(self, asr_model, diar_model, device: str = "cpu", sample_rate: int = config.SAMPLE_RATE):
        self.asr, self.diar = asr_model, diar_model
        self.device, self.sample_rate = device, sample_rate
        self.long_attention = False
        self.oom_cap = float("inf")                      # padded-audio-seconds limit learned from out-of-memory events
        self.vram_gb = self._vram_gb()
        self.bf16 = bool(config.ASR_BF16 and device == "cuda" and torch is not None and self._bf16_ok())
        self._lock = threading.Lock()                    # one batch on the GPU at a time (even with several browser tabs)

    # ---------------------------------------------------------------- helpers
    def _vram_gb(self) -> float:
        if torch is not None and self.device == "cuda":
            try:
                return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _bf16_ok() -> bool:
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False

    def precision(self) -> str:
        return "bf16 (Parakeet) + fp32 (Sortformer)" if self.bf16 else "fp32"

    def auto_batch_seconds(self) -> float:
        return batching.auto_batch_seconds(self.vram_gb)

    def effective_batch_seconds(self, requested: Optional[float] = None) -> float:
        """Padded-audio-seconds limit for a batch: the requested/auto value, never above what already ran out of memory."""
        base = float(requested) if requested else self.auto_batch_seconds()
        return max(60.0, min(base, self.oom_cap))

    def _free_gpu(self):
        gc.collect()
        if torch is not None and self.device == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _no_grad():
        return torch.inference_mode() if torch is not None else contextlib.nullcontext()

    def _autocast(self):
        if self.bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _peak_mb(self) -> float:
        if torch is not None and self.device == "cuda":
            try:
                return torch.cuda.max_memory_allocated() / 2 ** 20
            except Exception:
                pass
        return 0.0

    def _split_on_oom(self, fn: Callable, audios: List[np.ndarray]) -> list:
        """Run fn(audios); on out-of-memory split the batch in half and try again."""
        try:
            return fn(audios)
        except OOM_ERRORS:
            self._free_gpu()
            if len(audios) == 1:
                raise
            padded = len(audios) * max(len(a) for a in audios) / self.sample_rate
            self.oom_cap = min(self.oom_cap, max(60.0, padded * 0.5))      # remember: next batches stay below this
            mid = len(audios) // 2
            log.warning("Out of GPU memory with %d files -> retrying as %d + %d (batch limit now %.0f s)",
                        len(audios), mid, len(audios) - mid, self.oom_cap)
            return self._split_on_oom(fn, audios[:mid]) + self._split_on_oom(fn, audios[mid:])

    def _set_attention(self, longest_sec: float):
        """Parakeet's full attention handles ~24 min per file; for longer audio switch to local attention."""
        need_local = longest_sec > config.LONG_AUDIO_SEC
        if need_local == self.long_attention:
            return
        try:
            if need_local:
                self.asr.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
                self.asr.change_subsampling_conv_chunking_factor(1)
            else:
                self.asr.change_attention_model(self_attention_model="rel_pos", att_context_size=[-1, -1])
            self.long_attention = need_local
        except Exception as e:
            log.warning("Could not change attention mode: %s", e)

    # ---------------------------------------------------------------- the two models
    def _transcribe(self, audios: List[np.ndarray]):
        out = self.asr.transcribe(audios, batch_size=len(audios), timestamps=True, verbose=False)
        if isinstance(out, tuple):                       # some NeMo versions return (hypotheses, extra)
            out = out[0]
        return list(out)

    def _asr(self, audios: List[np.ndarray]) -> list:
        try:
            with self._no_grad(), self._autocast():
                return self._transcribe(audios)
        except OOM_ERRORS:
            raise
        except Exception as e:
            if not self.bf16:
                raise
            log.warning("bf16 transcription failed (%s) -> switching to fp32", e)
            self.bf16 = False
            with self._no_grad():
                return self._transcribe(audios)

    def _diar(self, audios: List[np.ndarray]) -> list:
        with self._no_grad():
            pred = self.diar.diarize(audio=audios, batch_size=len(audios), sample_rate=self.sample_rate)
            if len(pred) == len(audios):
                return list(pred)
            # batching not supported by this NeMo version -> one file at a time
            return [self.diar.diarize(audio=[a], batch_size=1, sample_rate=self.sample_rate)[0] for a in audios]

    # ---------------------------------------------------------------- public
    def infer(self, audios: List[np.ndarray], on_stage: Optional[Callable[[str], None]] = None) -> InferResult:
        """Both models on one batch. Returns words + speaker segments (same order as `audios`) and the timings."""
        stage = on_stage or (lambda s: None)
        with self._lock:
            self._set_attention(max(len(a) for a in audios) / self.sample_rate)
            if torch is not None and self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            stage("transcribing")
            t = time.time()
            hyps = self._split_on_oom(self._asr, audios)
            asr_sec = time.time() - t
            self._free_gpu()
            stage("diarizing")
            t = time.time()
            segs = self._split_on_oom(self._diar, audios)
            diar_sec = time.time() - t
            peak = self._peak_mb()
            self._free_gpu()
        return InferResult(hyps, segs, asr_sec, diar_sec, peak)


def load_engine(say: Optional[Callable[[str], None]] = None) -> Engine:
    """Load both models once. First run downloads them from Hugging Face (a few GB), later runs use the cache."""
    say = say or (lambda m: None)
    compat.fix_numba()                                   # Windows Application Control workaround (see compat.py)
    try:
        import torch as _torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as e:
        raise RuntimeError(f"A required package could not be loaded: {e}\n"
                           "Activate your virtual environment and run: pip install -r requirements.txt") from e

    try:                                                 # keep NeMo's console output quiet
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)
    except Exception:
        pass

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No CUDA GPU found - running on CPU will be MUCH slower.")

    say("Loading speech-to-text model (Parakeet) ...")
    asr = nemo_asr.models.ASRModel.from_pretrained(config.ASR_MODEL).to(device).eval()
    say("Loading speaker model (Sortformer) ...")
    diar = SortformerEncLabelModel.from_pretrained(config.DIAR_MODEL).to(device).eval()
    return Engine(asr, diar, device=device)
