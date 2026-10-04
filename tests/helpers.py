"""Fake models + synthetic media generated with ffmpeg (so tests need no GPU, NeMo or downloads)."""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mstt.engine import Engine          # noqa: E402
from mstt.media import ffmpeg_path      # noqa: E402


class Hyp:
    """Mimics a NeMo Hypothesis: one word per second of audio."""
    def __init__(self, n):
        self.timestamp = {"word": [{"word": f"w{i}", "start": float(i), "end": i + 0.9} for i in range(n)]}


class FakeASR:
    def __init__(self, oom_above=None):
        self.calls, self.oom_above, self.attention = [], oom_above, "full"

    def transcribe(self, audios, batch_size=4, timestamps=False, verbose=True):
        assert batch_size == len(audios)
        if self.oom_above and len(audios) > self.oom_above:
            raise MemoryError("fake out of memory")
        self.calls.append([round(len(a) / 16000) for a in audios])
        return [Hyp(int(len(a) / 16000)) for a in audios]

    def change_attention_model(self, self_attention_model=None, att_context_size=None):
        self.attention = self_attention_model

    def change_subsampling_conv_chunking_factor(self, n):
        pass


class FakeDiar:
    """Alternates speaker_0 / speaker_1 every 4 seconds."""
    def __init__(self):
        self.calls = []

    def diarize(self, audio, batch_size=1, sample_rate=16000):
        self.calls.append([round(len(a) / 16000) for a in audio])
        out = []
        for a in audio:
            secs, t, k, segs = len(a) / 16000, 0.0, 0, []
            while t < secs:
                segs.append(f"{t:.3f} {min(t + 4, secs):.3f} speaker_{k % 2}")
                t += 4
                k += 1
            out.append(segs)
        return out


def make_engine(oom_above=None):
    return Engine(FakeASR(oom_above), FakeDiar(), device="cpu")


def _ff(*args):
    subprocess.run([ffmpeg_path(), "-y", "-v", "error", *args], check=True)


def make_wav(path, seconds, freq=300):
    _ff("-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", "-ar", "44100", "-ac", "2", str(path))


def make_video(path, seconds):
    _ff("-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest",
        "-c:v", "libx264", "-c:a", "aac", str(path))


def make_silent_video(path, seconds):
    _ff("-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}", "-c:v", "libx264", str(path))


def tmpdir():
    return tempfile.TemporaryDirectory()
