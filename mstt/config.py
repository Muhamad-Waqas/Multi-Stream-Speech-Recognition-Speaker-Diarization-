"""All settings in one place."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAX_FILES = 30                      # most videos/audio files per run
SAMPLE_RATE = 16000                 # both models expect 16 kHz mono audio

ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"                 # speech -> words + timestamps
DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"  # who spoke when (up to 4 speakers)

SILENCE_GAP = 0.8                   # a pause longer than this (seconds) starts a new speaker turn
LONG_AUDIO_SEC = 20 * 60            # Parakeet switches to local attention above this length
OUTPUT_DIR = ROOT / "outputs"       # transcripts are saved here as well

MEDIA_EXTS = ["mp4", "mov", "mkv", "avi", "webm", "flv", "m4v", "wmv", "ts",
              "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma"]

# ---------------------------------------------------------------- scaling to many files
# Files are sorted by length and sent to the GPU in batches. A batch is limited by
#   * MAX_BATCH_FILES, and
#   * "padded audio seconds" = (number of files) x (longest file in the batch)   <- what GPU memory really depends on.
# The limit is chosen automatically from your GPU memory and lowered automatically if the GPU runs out of memory.
MAX_BATCH_FILES = 30
BATCH_SECONDS_PER_VRAM_GB = 300     # auto limit = VRAM (GB) x 300 s   (24 GB GPU -> 120 min of padded audio per batch)
BATCH_SECONDS_MIN = 600
BATCH_SECONDS_MAX = 43200
CPU_BATCH_SECONDS = 1800            # used when there is no CUDA GPU
DECODE_WORKERS = 8                  # ffmpeg processes running at the same time
PREFETCH_FACTOR = 2.0               # decode up to this many batches AHEAD of the GPU (keeps RAM bounded)

ASR_BF16 = os.environ.get("MSTT_BF16", "1") != "0"   # bf16 autocast for Parakeet on GPUs that support it (faster, less memory). MSTT_BF16=0 turns it off.
MONITOR_INTERVAL = 0.5              # seconds between GPU / CPU samples on the dashboard
