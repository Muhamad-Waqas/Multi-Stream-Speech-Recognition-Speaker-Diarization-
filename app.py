"""
MultiStream STT Studio - speech to text with speaker labels for up to 30 videos/audio files at once.

Run:  streamlit run app.py
Left: choose files.  Right: live dashboard (progress, per-file timing, GPU / CPU / memory) and the transcripts.
The job runs in a background thread, so clicking around in the page never interrupts it.
"""
from __future__ import annotations

import html
import importlib.util
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mstt import compat                        # noqa: E402
compat.fix_numba()                             # must run BEFORE NeMo is imported (Windows Application Control fix)

import streamlit as st                         # noqa: E402

from mstt import batching, config, engine, exporters, media, monitor, report   # noqa: E402
from mstt.job import Job, JobView              # noqa: E402

st.set_page_config(page_title="MultiStream STT Studio", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

PALETTE = ["#4f46e5", "#e11d48", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d"]
STATUS_COLOR = {"queued": "#cbd5e1", "extracting": "#38bdf8", "ready": "#818cf8", "transcribing": "#6366f1",
                "diarizing": "#a855f7", "aligning": "#f59e0b", "done": "#10b981", "failed": "#ef4444"}
BUSY = ("extracting", "transcribing", "diarizing", "aligning")

CSS = """
<style>
.block-container{padding-top:1.3rem;max-width:1650px}
header[data-testid="stHeader"]{background:transparent}
.hero{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;padding:1.35rem 1.8rem;border-radius:1.1rem;color:#fff;background:linear-gradient(120deg,#312e81 0%,#4f46e5 48%,#0ea5e9 100%);box-shadow:0 10px 28px rgba(79,70,229,.25);margin-bottom:1.1rem}
.hero-t{font-size:1.85rem;font-weight:800;letter-spacing:-.02em;line-height:1.2}
.hero-s{opacity:.92;margin-top:.25rem;font-size:.95rem}
.chips{display:flex;gap:.5rem;flex-wrap:wrap}
.chip{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);padding:.3rem .85rem;border-radius:2rem;font-size:.8rem;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:.35rem 0 .8rem}
.kpi{background:#fff;border:1px solid #e5e7eb;border-top:3px solid var(--accent,#4f46e5);border-radius:.8rem;padding:.65rem .95rem;box-shadow:0 1px 3px rgba(15,23,42,.06)}
.kpi-l{font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;color:#64748b;font-weight:700}
.kpi-v{font-size:1.5rem;font-weight:800;color:#0f172a;line-height:1.3;font-variant-numeric:tabular-nums}
.kpi-s{font-size:.78rem;color:#64748b}
.gbar{height:6px;border-radius:4px;background:#e2e8f0;margin-top:.4rem;overflow:hidden}
.gbar>span{display:block;height:100%;border-radius:4px;background:var(--accent,#4f46e5)}
.sec{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b;font-weight:700;margin:.9rem 0 .35rem}
.tiles{display:flex;flex-wrap:wrap;gap:.35rem}
.tile{width:2.15rem;height:2.15rem;border-radius:.55rem;color:#fff;font-size:.74rem;font-weight:700;display:flex;align-items:center;justify-content:center;cursor:default}
.tile.busy{animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.55}}
.legend{display:flex;flex-wrap:wrap;gap:.9rem;margin-top:.45rem;font-size:.75rem;color:#64748b}
.legend i{display:inline-block;width:.7rem;height:.7rem;border-radius:.2rem;margin-right:.3rem;vertical-align:-1px}
.turn{padding:.6rem .85rem;margin-bottom:.45rem;border-radius:.7rem;background:#f8fafc;border:1px solid #eef2f7;line-height:1.6;color:#0f172a}
.spk{display:inline-block;padding:.05rem .65rem;border-radius:1rem;color:#fff;font-size:.76rem;font-weight:700;margin-right:.5rem}
.ts{color:#94a3b8;font-size:.78rem;margin-right:.5rem;font-variant-numeric:tabular-nums}
.tl{position:relative;height:1.6rem;border-radius:.5rem;background:#eef2f7;overflow:hidden}
.tl span{position:absolute;top:0;bottom:0;opacity:.92}
.tl-axis{display:flex;justify-content:space-between;font-size:.72rem;color:#94a3b8;margin-top:.15rem}
.spk-row{display:flex;flex-wrap:wrap;gap:.6rem;margin:.55rem 0 .2rem}
.spk-card{display:flex;align-items:center;gap:.5rem;background:#fff;border:1px solid #e5e7eb;border-radius:.7rem;padding:.35rem .75rem;font-size:.82rem;color:#0f172a}
.spk-card b{font-variant-numeric:tabular-nums}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
"""

BLOCKED_HELP = (
    "Windows is blocking a program file (Application Control / Smart App Control). The app already works around the "
    "common case (numba), so this is a different file. Run `python check_setup.py` to see exactly which package is "
    "blocked. The most reliable fix is to run the project in **WSL2 (Ubuntu)** - see README.md."
)


@dataclass
class Settings:
    batch_seconds: Optional[float]      # None = automatic
    silence_gap: float


# ====================================================================== small helpers
@st.cache_resource(show_spinner=False)
def get_engine():
    """Models are loaded once and reused for every run."""
    return engine.load_engine()


@st.cache_data(show_spinner=False)
def get_device() -> dict:
    return monitor.device_info()


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-() ]", "_", Path(name).name) or "upload"


def unique_dest(folder: Path, name: str) -> Path:
    dest, n = folder / name, 1
    while dest.exists():
        dest = folder / f"{Path(name).stem}_{n}{Path(name).suffix}"
        n += 1
    return dest


def list_media(folder: str) -> List[str]:
    """Media files directly inside a folder (for 'process a folder on this PC' - no upload needed)."""
    folder = (folder or "").strip().strip('"').strip("'")
    if not folder:
        return []
    try:
        d = Path(folder).expanduser()
        if not d.is_dir():
            return []
        exts = {"." + e for e in config.MEDIA_EXTS}
        return sorted(str(p) for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    except OSError:
        return []


def f0(x, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:.0f}{unit}"


def f1(x, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:.1f}{unit}"


def gb(mb) -> Optional[float]:
    return None if mb is None else mb / 1024.0


def show_table(rows, **kw) -> None:
    """st.dataframe stretched to the full width (works on old and new Streamlit versions)."""
    try:
        st.dataframe(rows, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(rows, use_container_width=True, hide_index=True, **kw)


def kpi(label: str, value: str, sub: str = "", accent: str = "#4f46e5", bar: Optional[float] = None) -> str:
    gauge = f'<div class="gbar"><span style="width:{max(0.0, min(bar, 100.0)):.0f}%"></span></div>' if bar is not None else ""
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-l">{html.escape(label)}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div>{gauge}</div>')


def kpi_row(*cards: str) -> None:
    st.markdown('<div class="kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def speaker_colors(turns) -> dict:
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted({t.speaker for t in turns}))}


def turns_html(turns) -> str:
    if not turns:
        return "<i>No speech was detected in this file.</i>"
    colors = speaker_colors(turns)
    return "".join(
        f'<div class="turn"><span class="spk" style="background:{colors[t.speaker]}">{html.escape(t.speaker)}</span>'
        f'<span class="ts">{exporters.fmt_time(t.start)}</span>{html.escape(t.text)}</div>' for t in turns)


def timeline_html(turns) -> str:
    """One coloured bar showing who speaks when, plus a card per speaker with talk time."""
    colors = speaker_colors(turns)
    end = max(t.end for t in turns) or 1.0
    bars = "".join(
        f'<span style="left:{100 * t.start / end:.3f}%;width:{max(100 * (t.end - t.start) / end, 0.2):.3f}%;'
        f'background:{colors[t.speaker]}" title="{html.escape(t.speaker)} {exporters.fmt_time(t.start)}"></span>' for t in turns)
    cards = "".join(
        f'<div class="spk-card"><span class="spk" style="background:{colors[s["speaker"]]};margin:0">{html.escape(s["speaker"])}</span>'
        f'<b>{report.fmt_dur(s["seconds"])}</b> talking · {s["share"] * 100:.0f}% · {s["turns"]} turns · {s["words"]} words</div>'
        for s in report.speaker_stats(turns))
    return (f'<div class="tl">{bars}</div><div class="tl-axis"><span>0:00</span><span>{report.fmt_dur(end)}</span></div>'
            f'<div class="spk-row">{cards}</div>')


def tiles_html(files) -> str:
    tiles = "".join(
        f'<div class="tile{" busy" if f.status in BUSY else ""}" style="background:{STATUS_COLOR.get(f.status, "#cbd5e1")}" '
        f'title="{html.escape(f.name)} - {html.escape(report.STATUS_LABEL.get(f.status, f.status))}">{i}</div>'
        for i, f in enumerate(files, 1))
    legend = "".join(f'<span><i style="background:{STATUS_COLOR[s]}"></i>{html.escape(report.STATUS_LABEL[s].split(" ", 1)[1])}</span>'
                     for s in ("queued", "extracting", "ready", "transcribing", "diarizing", "aligning", "done", "failed"))
    return f'<div class="tiles">{tiles}</div><div class="legend">{legend}</div>'


# ====================================================================== dashboard
def history_series(v: JobView, keys: dict) -> dict:
    """keys = {'GPU %': 'gpu_util', ...}  ->  {'Time (s)': [...], 'GPU %': [...]} (only series that have data)."""
    h = v.history
    if len(h) > 600:
        h = h[::max(1, len(h) // 600)]
    data = {"Time (s)": [round(x["t"], 1) for x in h]}
    for label, key in keys.items():
        vals = [x.get(key) for x in h]
        if any(val is not None for val in vals):
            data[label] = [None if val is None else (val / 1024.0 if key.startswith("vram") else val) for val in vals]
    return data


def draw_chart(data: dict, colors: dict, title: str) -> None:
    series = [k for k in data if k != "Time (s)"]
    st.markdown(f'<div class="sec">{html.escape(title)}</div>', unsafe_allow_html=True)
    if len(data["Time (s)"]) < 2 or not series:
        st.caption("Collecting data ..." if series else "Not available on this computer.")
        return
    st.line_chart(data, x="Time (s)", y=series, color=[colors[s] for s in series], height=190)


def render_dashboard(v: JobView) -> None:
    total = v.n_total
    finished = v.n_done + v.n_failed
    if v.error:
        st.error(f"The run stopped unexpectedly: {v.error}")
    if v.running:
        st.progress(min(max(v.progress, 0.0), 1.0),
                    text=f"{v.phase}  ·  {finished}/{total} files  ·  {report.fmt_dur(v.elapsed)} elapsed")
    else:
        st.progress(1.0, text=f"Finished  ·  {v.n_done} done, {v.n_failed} failed  ·  total time {report.fmt_dur(v.elapsed)}")

    g, s = v.latest or {}, v.summary or {}
    batch_txt = f"batch {min(v.batches_done + (1 if v.running else 0), max(v.batches_total, 1))}/{max(v.batches_total, 1)}"
    kpi_row(
        kpi("Files finished", f"{finished}<span style='font-size:1rem;color:#94a3b8'> / {total}</span>",
            f"{v.n_done} done · {v.n_failed} failed", "#4f46e5", 100 * finished / max(total, 1)),
        kpi("Elapsed time", report.fmt_dur(v.elapsed), batch_txt if v.batches_total else "preparing", "#0ea5e9"),
        kpi("Audio processed", f"{v.audio_done / 60:.1f} min", f"of {v.audio_total / 60:.1f} min in total", "#8b5cf6"),
        kpi("Throughput", f"{v.throughput:.0f}×", "seconds of audio per second of waiting", "#10b981"),
    )
    if v.has_gpu:
        used, tot = g.get("vram_used_mb"), g.get("vram_total_mb")
        kpi_row(
            kpi("GPU utilization", f0(g.get("gpu_util"), "%"),
                f"peak {f0(s.get('gpu_util_peak'), '%')} · average {f0(s.get('gpu_util_avg'), '%')}", "#4f46e5", g.get("gpu_util")),
            kpi("GPU memory", f1(gb(used), " GB"),
                f"peak {f1(gb(s.get('vram_used_mb_peak')))} of {f0(gb(tot))} GB",
                "#e11d48", (100 * used / tot) if used and tot else None),
            kpi("GPU temperature · power", f0(g.get("gpu_temp"), " °C"),
                f"{f0(g.get('gpu_power'))} W now · peak {f0(s.get('gpu_power_peak'))} W", "#f59e0b"),
            kpi("CPU · RAM", f0(g.get("cpu"), "%"),
                f"RAM {f1(g.get('ram_used_gb'))} GB ({f0(g.get('ram_pct'), '%')})", "#0891b2", g.get("cpu")),
        )
    else:
        kpi_row(kpi("GPU", "not detected", "no NVIDIA GPU / driver found - running on the CPU", "#94a3b8"),
                kpi("CPU", f0(g.get("cpu"), "%"), f"peak {f0(s.get('cpu_peak'), '%')}", "#0891b2", g.get("cpu")),
                kpi("RAM", f1(g.get("ram_used_gb"), " GB"), f"{f0(g.get('ram_pct'), '%')} used", "#f59e0b", g.get("ram_pct")))

    st.markdown('<div class="sec">Files</div>', unsafe_allow_html=True)
    st.markdown(tiles_html(v.files), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        draw_chart(history_series(v, {"GPU %": "gpu_util", "CPU %": "cpu"}), {"GPU %": "#4f46e5", "CPU %": "#f59e0b"},
                   "GPU and CPU utilization (%)")
    with c2:
        draw_chart(history_series(v, {"GPU memory (GB)": "vram_used_mb", "RAM (GB)": "ram_used_gb"}),
                   {"GPU memory (GB)": "#e11d48", "RAM (GB)": "#0891b2"}, "GPU memory and RAM (GB)")

    st.markdown('<div class="sec">Every file - status and processing time</div>', unsafe_allow_html=True)
    cc = st.column_config
    show_table(report.table_rows(v.files), column_config={
        "#": cc.NumberColumn("#", width="small"),
        "File": cc.TextColumn("File", width="medium"),
        "Progress": cc.ProgressColumn("Progress", min_value=0, max_value=100, format="%d%%"),
        "Extract (s)": cc.NumberColumn("Extract (s)", format="%.1f", help="ffmpeg: video/audio -> 16 kHz audio"),
        "Transcribe (s)": cc.NumberColumn("Transcribe (s)", format="%.1f", help="this file's share of the speech-to-text GPU time"),
        "Speakers (s)": cc.NumberColumn("Speakers (s)", format="%.1f", help="this file's share of the speaker-detection GPU time"),
        "Total (s)": cc.NumberColumn("Total (s)", format="%.1f", help="all processing time of this file"),
        "Speed (x real-time)": cc.NumberColumn("Speed (×RT)", format="%.1f", help="seconds of audio per second of processing"),
        "Done at": cc.TextColumn("Done at", help="time since the run started when this file finished"),
    })
    if v.batches:
        with st.expander(f"GPU batches ({len(v.batches)})"):
            names = {"batch": "Batch", "files": "Files", "audio_sec": "Audio (s)", "padded_sec": "Padded audio (s)",
                     "asr_sec": "Transcribe (s)", "diar_sec": "Speakers (s)", "peak_vram_mb": "Peak GPU memory (MB)"}
            show_table([{names[k]: b.get(k) for k in names} for b in v.batches])
    if not v.running and v.files:
        st.download_button("⬇ Performance report (.csv)", data=report.to_csv(v.files),
                           file_name="performance_report.csv", mime="text/csv")


def live_panel() -> None:
    """Redraws once a second while a job runs (Streamlit fragment); when the job ends it reloads the whole page."""
    ss = st.session_state
    job = ss.get("job")
    if job is not None:
        render_dashboard(job.view())
        if job.done:
            st.rerun()
    elif ss.get("last_view") is not None:
        render_dashboard(ss.last_view)
    else:
        st.info("The live dashboard appears here when you press **Transcribe**: progress and processing time of every "
                "file, GPU utilization, GPU memory, CPU and RAM.")


# ====================================================================== transcripts
def render_transcripts() -> None:
    ss = st.session_state
    results = ss.get("transcripts")
    if not results:
        st.info("Transcripts appear here when the run finishes.")
        return
    name = st.selectbox("Video", list(results))
    turns = results[name]
    view = ss.get("last_view")
    fs = next((f for f in (view.files if view else []) if f.name == name), None)
    if fs:
        kpi_row(kpi("Audio length", report.fmt_dur(fs.audio_sec), f"{fs.size_mb:.0f} MB file", "#8b5cf6"),
                kpi("Processing time", f"{fs.t_total:.1f} s", f"extract {fs.t_extract:.1f} · transcribe {fs.t_asr:.1f} · speakers {fs.t_diar:.1f}", "#0ea5e9"),
                kpi("Speed", f"{fs.speed:.0f}×", "faster than real-time", "#10b981"),
                kpi("Speakers", str(fs.speakers), f"{fs.words} words · {fs.turns} turns", "#e11d48"))
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ This transcript (.txt)", data=exporters.to_txt(turns), file_name=f"{name}.txt", mime="text/plain")
    c2.download_button("⬇ Subtitles (.srt)", data=exporters.to_srt(turns), file_name=f"{name}.srt", mime="text/plain")
    c3.download_button(f"📦 All {len(results)} (.zip)", data=exporters.make_zip(results),
                       file_name="transcripts.zip", mime="application/zip")
    if turns:
        st.markdown('<div class="sec">Who speaks when</div>', unsafe_allow_html=True)
        st.markdown(timeline_html(turns), unsafe_allow_html=True)
    st.markdown('<div class="sec">Transcript</div>', unsafe_allow_html=True)
    with st.container(height=520, border=True):
        st.markdown(turns_html(turns), unsafe_allow_html=True)


# ====================================================================== page parts
def sidebar(dev: dict) -> Settings:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        auto_sec = batching.auto_batch_seconds(dev.get("vram_gb"))
        auto = st.toggle("Automatic GPU batch size", value=True,
                         help="Files are sorted by length and sent to the GPU in batches. Automatic = chosen from your GPU memory "
                              "and lowered by itself if the GPU runs out of memory.")
        if auto:
            st.caption(f"≈ {auto_sec / 60:.0f} minutes of audio per batch" + (f" (from {dev['vram_gb']:.0f} GB GPU memory)" if dev.get("vram_gb") else " (CPU)"))
            batch_seconds = None
        else:
            start = int(max(5, min(720, round(auto_sec / 60 / 5) * 5)))
            batch_seconds = 60.0 * st.slider("Audio minutes per GPU batch", 5, 720, start, step=5,
                                             help="Bigger = fewer, faster batches but more GPU memory. (files x longest file)")
        silence = st.slider("Pause that starts a new speaker turn (seconds)", 0.2, 3.0, float(config.SILENCE_GAP), step=0.1)
        st.divider()
        st.markdown("### 🖥️ This computer")
        gpu_line = f"{dev['gpu']} · {dev['vram_gb']:.0f} GB" if dev.get("gpu") else "none detected (CPU mode)"
        st.caption(f"**GPU:** {gpu_line}")
        st.caption(f"**CPU:** {dev.get('cpu_cores', '?')} cores · **RAM:** {dev.get('ram_gb', 0):.0f} GB")
        st.caption("**Models:** Parakeet-TDT 0.6B v3 + Streaming Sortformer 4spk v2.1")
    return Settings(batch_seconds=batch_seconds, silence_gap=float(silence))


def hero(dev: dict) -> None:
    chip = f"⚡ {html.escape(dev['gpu'])} · {dev['vram_gb']:.0f} GB" if dev.get("gpu") else "🐢 CPU mode (no NVIDIA GPU detected)"
    st.markdown(
        f'<div class="hero"><div><div class="hero-t">🎙️ MultiStream STT Studio</div>'
        f'<div class="hero-s">Speech-to-text with speaker labels for up to {config.MAX_FILES} videos or audio files at once</div></div>'
        f'<div class="chips"><span class="chip">{chip}</span><span class="chip">Up to {config.MAX_FILES} files per run</span>'
        f'<span class="chip">Parakeet-TDT + Sortformer</span></div></div>', unsafe_allow_html=True)


def collect_finished_job() -> None:
    """When the background job has ended, move its results into the page state."""
    ss = st.session_state
    job = ss.get("job")
    if job is None or not job.done:
        return
    view = job.view()
    res = job.result
    ss.last_view = view
    ss.transcripts = res.transcripts if res else {}
    ss.failed = res.failed if res else {f.name: f.error for f in view.files if f.status == "failed"}
    ss.seconds = res.seconds if res else view.elapsed
    ss.out_dir = res.out_dir if res else job.out_dir
    ss.job = None


def start_job(mode: str, files, found: List[str], settings: Settings) -> None:
    ss = st.session_state
    tmp: Optional[Path] = None
    try:
        if mode == "upload":
            tmp = Path(tempfile.mkdtemp(prefix="mstt_"))
            paths = []
            for f in files:
                dest = unique_dest(tmp, safe_filename(f.name))
                dest.write_bytes(f.getbuffer())
                paths.append(str(dest))
        else:
            paths = list(found)
        with st.spinner("Loading models (the first run downloads them and can take a few minutes) ..."):
            try:
                eng = get_engine()
            except Exception as e:
                st.error(f"Could not load the models: {type(e).__name__}: {e}")
                if "Application Control" in str(e) or "DLL load failed" in str(e):
                    st.info(BLOCKED_HELP)
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)
                return
        out_dir = str(config.OUTPUT_DIR / time.strftime("%Y%m%d-%H%M%S"))
        job = Job(eng, paths, out_dir, tmp_dir=str(tmp) if tmp else None,
                  batch_seconds=settings.batch_seconds, silence_gap=settings.silence_gap)
        job.start()
        ss.job = job
    except Exception:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    st.rerun()                                   # redraw the page in its "running" state (buttons disabled, Stop shown)


def upload_panel(running: bool, settings: Settings) -> None:
    ss = st.session_state
    st.subheader("1 · Upload")
    mode_label = st.radio("Where are your files?", ["Upload files", "Folder on this PC"], horizontal=True, disabled=running,
                          help="'Folder on this PC' reads the files where they are - no upload, best for big videos.")
    files, found = [], []
    if mode_label == "Upload files":
        mode = "upload"
        files = st.file_uploader(f"Drop up to {config.MAX_FILES} videos or audio files", type=config.MEDIA_EXTS,
                                 accept_multiple_files=True, disabled=running) or []
        count = len(files)
    else:
        mode = "folder"
        folder = st.text_input("Folder path", placeholder=r"C:\Videos\batch1", disabled=running)
        found = list_media(folder)
        count = len(found)
        if folder.strip():
            st.caption(f"Found {count} media file(s) in this folder." if count else "No media files found - check the path.")

    too_many = count > config.MAX_FILES
    if too_many:
        st.error(f"You selected {count} files - the limit is {config.MAX_FILES}. "
                 f"Remove {count - config.MAX_FILES} and try again.")

    problems = []
    if not media.ffmpeg_path():
        problems.append("ffmpeg not found. Run `pip install imageio-ffmpeg` and restart the app.")
    if importlib.util.find_spec("nemo") is None:
        problems.append("NVIDIA NeMo is not installed. Run `pip install -r requirements.txt`.")
    for p in problems:
        st.error(p)

    go = st.button("🚀 Transcribe", type="primary", disabled=not count or too_many or bool(problems) or running)
    if running:
        if st.button("⏹ Stop after the current batch"):
            ss.job.cancel()
        st.caption("Running in the background - you can switch tabs or change settings; the run continues.")
    if go:
        for k in ("transcripts", "failed", "last_view"):
            ss.pop(k, None)
        start_job(mode, files, found, settings)

    if ss.get("failed"):
        with st.expander(f"⚠️ {len(ss.failed)} file(s) could not be processed", expanded=True):
            for name, why in ss.failed.items():
                st.write(f"**{name}** - {why}")
    if ss.get("transcripts"):
        st.success(f"Finished {len(ss.transcripts)} file(s) in {ss.seconds:.0f} s. Files also saved in: {ss.out_dir}")


def main() -> None:
    ss = st.session_state
    st.markdown(CSS, unsafe_allow_html=True)
    collect_finished_job()
    dev = get_device()
    settings = sidebar(dev)
    hero(dev)

    left, right = st.columns([1, 2.3], gap="large")
    with left:
        running = ss.get("job") is not None and not ss.job.done
        upload_panel(running, settings)
    with right:
        running = ss.get("job") is not None and not ss.job.done      # the upload panel may just have started a job
        tab_dash, tab_text = st.tabs(["📊 Live dashboard", "📝 Transcripts"])
        with tab_dash:
            st.fragment(run_every=1.0 if running else None)(live_panel)()
        with tab_text:
            render_transcripts()


main()
