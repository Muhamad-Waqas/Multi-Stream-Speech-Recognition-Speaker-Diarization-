"""Runs app.py top to bottom against the stub streamlit: upload -> background job -> dashboard -> transcript view."""
import importlib.util
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

import helpers
import stub_streamlit
from stub_streamlit import FakeUpload

APP = str(Path(__file__).resolve().parents[1] / "app.py")
_real_find_spec = importlib.util.find_spec


def fake_find_spec(name, *a, **k):                       # pretend NeMo is installed (it is not in the test sandbox)
    return object() if name == "nemo" else _real_find_spec(name, *a, **k)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.tmpdir()
        self.dir = Path(self.tmp.name)
        import mstt.config as config
        self.config = config
        self._old_out = config.OUTPUT_DIR
        config.OUTPUT_DIR = self.dir / "outputs"
        self.engine = helpers.make_engine()

    def tearDown(self):
        self.config.OUTPUT_DIR = self._old_out
        self.tmp.cleanup()

    def wav(self, name, seconds):
        p = self.dir / name
        helpers.make_wav(p, seconds)
        return FakeUpload(name, p.read_bytes())

    def run_app(self, st, load_engine=None):
        sys.modules["streamlit"] = st
        with mock.patch("importlib.util.find_spec", fake_find_spec), \
             mock.patch("mstt.engine.load_engine", load_engine or (lambda say=None: self.engine)):
            runpy.run_path(APP, run_name="__main__")

    def run_to_completion(self, st):
        """Press Transcribe, wait for the background thread, then 'reload' the page like Streamlit does."""
        self.run_app(st)
        job = st.session_state.get("job")
        self.assertIsNotNone(job, "the Transcribe button should have started a background job")
        self.assertTrue(job.join(60))
        st.press = set()
        st.log.clear(); st.downloads.clear(); st.html_blocks.clear(); st.tables.clear(); st.charts.clear()
        self.run_app(st)
        return job

    # ------------------------------------------------------------------
    def test_fresh_page_layout(self):
        st = stub_streamlit.make_stub()
        self.run_app(st)
        subs = [t for k, t in st.log if k == "subheader"]
        self.assertEqual(subs, ["1 · Upload"])
        self.assertTrue(st.buttons["🚀 Transcribe"])                        # disabled until files are chosen
        self.assertEqual(st.downloads, [])
        self.assertEqual(st.fragments, [None])                              # nothing running -> no polling
        self.assertTrue(any(k == "info" and "live dashboard" in t for k, t in st.log))
        self.assertTrue(any("hero" in b for b in st.html_blocks))

    def test_thirty_uploads_end_to_end(self):
        st = stub_streamlit.make_stub()
        st.uploads = [self.wav(f"video{i:02d}.wav", 4 + i % 6) for i in range(30)]
        st.press = {"Transcribe"}
        job = self.run_to_completion(st)
        res = st.session_state["transcripts"]
        self.assertEqual(sorted(res), sorted(f"video{i:02d}" for i in range(30)))
        self.assertEqual(st.session_state["failed"], {})
        self.assertEqual(len(self.engine.asr.calls), 1)                     # one GPU batch for all thirty
        self.assertEqual(len(self.engine.asr.calls[0]), 30)
        self.assertGreater(st.reruns, 0)                                    # page reruns once the job is started
        self.assertIsNone(st.session_state["job"])                          # job collected
        sel = [t for k, t in st.log if k == "selectbox"][-1].split(",")
        self.assertEqual(len(sel), 30)
        labels = [d[0] for d in st.downloads]
        self.assertTrue(any(".txt" in l for l in labels) and any(".zip" in l for l in labels) and any(".srt" in l for l in labels))
        self.assertTrue(any("Performance report" in l for l in labels))
        self.assertGreater([d[2] for d in st.downloads if d[1] == "transcripts.zip"][0], 100)
        self.assertTrue(any(k == "success" and "Finished 30 file(s)" in t for k, t in st.log))
        self.assertEqual(len(list((self.dir / "outputs").glob("*/*.txt"))), 30)       # saved on disk too
        self.assertEqual(len(list((self.dir / "outputs").glob("*/performance_report.csv"))), 1)
        self.assertEqual(list(Path(__import__("tempfile").gettempdir()).glob("mstt_*")), [])   # temp uploads removed
        # dashboard shows one row per file with timing columns, a tile per file, GPU numbers, and a speaker timeline
        rows = st.tables[0]
        self.assertEqual(len(rows), 30)
        for col in ("File", "Status", "Progress", "Audio", "Extract (s)", "Transcribe (s)", "Speakers (s)", "Total (s)", "Done at"):
            self.assertIn(col, rows[0])
        self.assertTrue(all(r["Status"] == "✅ Done" for r in rows))
        html_all = "".join(st.html_blocks)
        self.assertEqual(len(__import__("re").findall(r'class="tile[ "]', html_all)), 30)
        self.assertIn("GPU", html_all)
        self.assertIn('class="tl"', html_all)                                # who-speaks-when bar
        self.assertIn("speaker_0", html_all)

    def test_page_polls_while_running_and_stop_button_exists(self):
        st = stub_streamlit.make_stub()
        st.uploads = [self.wav(f"v{i}.wav", 3) for i in range(3)]
        st.press = {"Transcribe"}
        self.run_app(st)
        job = st.session_state["job"]
        st.press = set()
        job.join(60)
        # simulate a rerun that happens while the job is still registered but running
        job._done.clear()
        st.fragments.clear()
        st.buttons.clear()
        self.run_app(st)
        self.assertEqual(st.fragments, [1.0])                               # fragment polls once a second
        self.assertTrue(st.buttons["🚀 Transcribe"])                        # disabled while running
        self.assertIn("⏹ Stop after the current batch", st.buttons)
        job._done.set()

    def test_more_than_thirty_is_blocked(self):
        st = stub_streamlit.make_stub()
        st.uploads = [self.wav(f"v{i}.wav", 2) for i in range(31)]
        st.press = {"Transcribe"}
        self.run_app(st)
        self.assertTrue(st.buttons["🚀 Transcribe"])
        self.assertNotIn("transcripts", st.session_state)
        self.assertIsNone(st.session_state.get("job"))
        self.assertTrue(any(k == "error" and "limit is 30" in t for k, t in st.log))
        self.assertEqual(self.engine.asr.calls, [])

    def test_folder_mode_reads_files_in_place(self):
        folder = self.dir / "batch"
        folder.mkdir()
        for i in range(5):
            helpers.make_wav(folder / f"talk{i}.wav", 4 + i)
        (folder / "notes.txt").write_text("not media")
        st = stub_streamlit.make_stub()
        st.radio_value, st.folder = "Folder on this PC", str(folder)
        st.press = {"Transcribe"}
        self.run_to_completion(st)
        self.assertEqual(sorted(st.session_state["transcripts"]), [f"talk{i}" for i in range(5)])
        self.assertEqual(len(list(folder.glob("*.wav"))), 5)               # originals untouched, nothing copied or deleted

    def test_bad_file_is_reported_and_others_still_shown(self):
        st = stub_streamlit.make_stub()
        st.uploads = [self.wav("good.wav", 6), FakeUpload("broken.mp4", b"not a video")]
        st.press = {"Transcribe"}
        self.run_to_completion(st)
        self.assertEqual(list(st.session_state["transcripts"]), ["good"])
        self.assertEqual(list(st.session_state["failed"]), ["broken"])
        self.assertTrue(any(k == "write" and "broken" in t for k, t in st.log))
        self.assertTrue(any(r["Status"] == "❌ Failed" for r in st.tables[0]))

    def test_same_file_name_twice(self):
        st = stub_streamlit.make_stub()
        st.uploads = [self.wav("clip.wav", 4), self.wav("clip.wav", 6)]
        st.press = {"Transcribe"}
        self.run_to_completion(st)
        self.assertEqual(sorted(st.session_state["transcripts"]), ["clip", "clip_1"])

    def test_windows_blocked_error_shows_help_not_a_crash(self):
        def boom(say=None):
            raise RuntimeError("DLL load failed while importing foo: An Application Control policy has blocked this file.")
        st = stub_streamlit.make_stub()
        st.uploads = [self.wav("a.wav", 3)]
        st.press = {"Transcribe"}
        self.run_app(st, load_engine=boom)
        self.assertTrue(any(k == "error" and "Could not load the models" in t for k, t in st.log))
        self.assertTrue(any(k == "info" and "WSL2" in t for k, t in st.log))
        self.assertNotIn("transcripts", st.session_state)
        self.assertIsNone(st.session_state.get("job"))
        self.assertEqual(list(Path(__import__("tempfile").gettempdir()).glob("mstt_*")), [])

    def test_manual_batch_size_setting_is_used(self):
        st = stub_streamlit.make_stub()
        st.toggle_values = {"Automatic GPU batch size": False}
        st.slider_values = {"Audio minutes per GPU batch": 5}              # 5 minutes = 300 padded seconds
        st.uploads = [self.wav(f"c{i}.wav", 40 + i) for i in range(10)]    # 10 x ~45 s = 450 s > 300 s -> must split
        st.press = {"Transcribe"}
        self.run_to_completion(st)
        self.assertGreater(len(self.engine.asr.calls), 1)
        self.assertEqual(len(st.session_state["transcripts"]), 10)

    def test_transcript_text_is_html_escaped(self):
        from mstt.alignment import Turn
        st = stub_streamlit.make_stub()
        st.session_state["transcripts"] = {"x": [Turn("speaker_0", 0, 1, "<script>alert(1)</script>")]}
        st.session_state["seconds"], st.session_state["out_dir"] = 1.0, "o"
        self.run_app(st)
        html_all = "".join(st.html_blocks)
        self.assertNotIn("<script>", html_all)
        self.assertIn("&lt;script&gt;", html_all)

    def test_no_speech_file_shows_message(self):
        st = stub_streamlit.make_stub()
        st.session_state["transcripts"] = {"quiet": []}
        st.session_state["seconds"], st.session_state["out_dir"] = 1.0, "o"
        self.run_app(st)
        self.assertIn("No speech was detected", "".join(st.html_blocks))


if __name__ == "__main__":
    unittest.main()
