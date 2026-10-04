# 🎙️ MultiStream STT — Speech-to-Text & Speaker Diarization

A GPU-accelerated speech intelligence pipeline for processing multiple audio/video files, generating timestamped transcripts and speaker-labelled speech segments using NVIDIA Parakeet-TDT and Sortformer.

Upload or point to **up to 30 videos / audio files** and get a **speaker-labelled transcript with timestamps** for each,
plus a live dashboard with **processing time of every file and GPU / CPU / memory usage**.

* Speech → text: **NVIDIA Parakeet-TDT 0.6B v3** · Who spoke when: **NVIDIA Streaming Sortformer 4spk v2.1**
* Audio extraction: **ffmpeg** · Web page: **Streamlit**

## What is new in this version
| | Before (10 files) | Now (30 files) |
|---|---|---|
| Files per run | 10 | **30** (`MAX_FILES` in `msst/config.py`) |
| GPU work | one giant batch; one 1-hour file padded every short file to 1 hour | files **sorted by length**, grouped into batches of similar length; batch size limited by **GPU memory** (automatic) |
| Out of memory | split in half, forgotten afterwards | split in half **and remembered** - later batches/runs start below the limit |
| CPU vs GPU | extract all audio, *then* start the GPU | audio of the **next batches is extracted while the GPU works**; saving transcripts also overlaps the next batch; RAM stays bounded |
| Precision | fp32 | Parakeet in **bf16** on GPUs that support it (faster, less memory; `MSST_BF16=0` turns it off) |
| Page | blocks while running, clicking around interrupted the run | job runs in a **background thread**; page refreshes itself every second; you can click, switch tabs or reload |
| Big videos | browser upload only | also **"Folder on this PC"**: read files in place, no upload/copy |
| UI | basic | dashboard: KPI cards, file tiles, per-file table (extract / transcribe / speakers / total time, speed ×real-time, finish time), GPU + CPU + memory charts, per-batch table, **speaker timeline** and talk-time per speaker, CSV performance report, **Stop** button |

## How it works
```
files ─► 1. read every file's length (header only)
      ─► 2. sort by length -> batches   (batch cost = files x longest file  <=  limit from your GPU memory)
      ─► 3. ffmpeg threads extract audio a couple of batches ahead of the GPU
      ─► 4. per batch: Parakeet (words) then Sortformer (speakers) on the GPU
      ─► 5. worker threads give every word to the speaker it overlaps most -> speaker turns -> TXT + SRT
```
A corrupt file or a video without sound never stops the others.

## Install (Windows, VS Code terminal)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # if PowerShell refuses:  Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124    # CUDA build! pick yours at pytorch.org
pip install -r requirements.txt
python check_setup.py               # shows exactly what works and what does not
streamlit run app.py                # http://localhost:8501   (or double-click run_windows.bat)
```
Python **3.10 – 3.12**. The first run downloads the two models (a few GB).
The Windows "Application Control policy has blocked this file" (numba) error is handled automatically by `msst/compat.py`;
if `check_setup.py` shows another blocked package, use **WSL2** (`wsl --install -d Ubuntu`, then the same steps in Ubuntu).

## Using it
1. **Upload files** (left) or choose **Folder on this PC** and paste a folder path.
2. Press **🚀 Transcribe**. Watch the **📊 Live dashboard**. Press **⏹ Stop** to finish the current batch and skip the rest.
3. Open **📝 Transcripts**: pick a video, see *who speaks when*, read/download TXT, SRT, ZIP. Dashboard has the **performance report (.csv)**.

Everything is also saved in `outputs/<date-time>/` (`*.txt`, `*.srt`, `performance_report.csv`).

### Reading the timing columns
* **Extract / Transcribe / Speakers / Total (s)** - processing time of that file. Transcribe and Speakers are *this file's share* of its batch's GPU time (split by audio length), because a batch runs on the GPU together.
* **Done at** - time since the run started when that file was finished (short-file batches finish earlier than long ones).
* **Speed (×RT)** - seconds of audio per second of processing. **Throughput** (KPI) - audio finished per second of waiting for the whole run.

## Tuning for 30 files
Sidebar → **Automatic GPU batch size** (default): limit = GPU memory (GB) × 300 s of *padded* audio (24 GB → 2 h per batch).
If you want to experiment, switch it off and set minutes manually: bigger = fewer, faster batches but more GPU memory.
If the GPU runs out of memory the engine halves the batch, remembers the limit, and continues - nothing to do.
Other knobs live in `msst/config.py` (`BATCH_SECONDS_PER_VRAM_GB`, `DECODE_WORKERS`, `PREFETCH_FACTOR`).
Rule of thumb: **RAM** needs roughly 4 bytes × 16 000 per second of audio *in flight* (≈ 230 MB per hour of audio) - only about 2 batches are in memory at once.

## Files
```
app.py            the page (upload left; live dashboard + transcripts right)
check_setup.py    run this first if something fails
msst/
  config.py       settings (MAX_FILES = 30, models, batching knobs)
  batching.py     sort by length, group into GPU batches
  engine.py       loads the models, runs a batch, learns from out-of-memory, bf16
  pipeline.py     probe -> extract -> GPU -> align/save assembly line, per-file timings
  state.py        live per-file status and timings (thread-safe)
  job.py          runs the pipeline in a background thread; what the dashboard reads
  monitor.py      GPU (NVML / nvidia-smi), CPU, RAM sampling
  report.py       table rows, CSV report, speaker statistics
  media.py alignment.py exporters.py compat.py
tests/            44 automated tests (fake models, no GPU needed):  python -m unittest discover -s tests
```

## Limits
* **Languages:** Parakeet-TDT 0.6B v3 supports 25 European languages. **Urdu, Hindi, Arabic and most Asian languages are not supported** and give wrong text.
* **Speakers:** up to **4 per file** (Sortformer limit). Overlapping speech and noisy audio reduce accuracy.
* **No NVIDIA GPU?** It runs on the CPU, many times slower (the dashboard then shows CPU/RAM only).
* One GPU is used. Several browser tabs share it safely (one batch at a time).
* Only transcribe people with their consent.
