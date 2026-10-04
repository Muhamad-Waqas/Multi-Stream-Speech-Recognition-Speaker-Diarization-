import os
import time
import unittest
from pathlib import Path

import helpers
from mstt import batching, config, exporters, media, monitor, pipeline, report
from mstt.job import Job
from mstt.state import FileState, JobState
from mstt.alignment import (Turn, Word, SpeakerSegment, assign_words_to_speakers, merge_into_turns, parse_segment)


class AlignmentTests(unittest.TestCase):
    def test_parse_segment(self):
        s = parse_segment("0.080 3.600 speaker_0")
        self.assertEqual((s.start, s.end, s.speaker), (0.08, 3.6, "speaker_0"))
        with self.assertRaises(ValueError):
            parse_segment("garbage")

    def test_words_go_to_most_overlapping_speaker_and_pauses_split_turns(self):
        words = [Word("hi", 0.0, 0.5), Word("there", 0.6, 1.0), Word("yes", 4.2, 4.6), Word("later", 9.0, 9.4)]
        segs = [SpeakerSegment("A", 0, 3), SpeakerSegment("B", 4, 6)]
        turns = merge_into_turns(assign_words_to_speakers(words, segs), silence_gap=0.8)
        self.assertEqual([(t.speaker, t.text) for t in turns],
                         [("A", "hi there"), ("B", "yes"), ("B", "later")])     # 'later' = long pause -> new turn

    def test_no_segments_or_words(self):
        self.assertEqual(assign_words_to_speakers([], [SpeakerSegment("A", 0, 1)]), [])
        self.assertEqual(assign_words_to_speakers([Word("x", 0, 1)], [])[0][1], "UNKNOWN")


class ExportTests(unittest.TestCase):
    TURNS = [Turn("speaker_0", 0.0, 2.5, "hello"), Turn("speaker_1", 3661.5, 3663.0, "bye")]

    def test_txt_srt(self):
        self.assertIn("[00:00:00.00 - 00:00:02.50] speaker_0: hello", exporters.to_txt(self.TURNS))
        srt = exporters.to_srt(self.TURNS)
        self.assertIn("00:00:00,000 --> 00:00:02,500", srt)
        self.assertIn("01:01:01,500 --> 01:01:03,000", srt)

    def test_zip(self):
        import io, zipfile
        z = zipfile.ZipFile(io.BytesIO(exporters.make_zip({"a": self.TURNS, "b": self.TURNS})))
        self.assertEqual(sorted(z.namelist()), ["a.srt", "a.txt", "b.srt", "b.txt"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.tmpdir()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def files(self, n, seconds=6):
        out = []
        for i in range(n):
            p = self.dir / f"clip{i:02d}.wav"
            helpers.make_wav(p, seconds + i)                    # different lengths
            out.append(str(p))
        return out

    def test_ten_files_one_gpu_batch(self):
        eng = helpers.make_engine()
        paths = self.files(10)
        events = []
        res = pipeline.run(eng, paths, str(self.dir / "out"), lambda f, t: events.append((f, t)))
        self.assertEqual(len(res.transcripts), 10)
        self.assertEqual(res.failed, {})
        self.assertEqual(len(eng.asr.calls), 1)                 # ALL 10 files in ONE ASR call
        self.assertEqual(len(eng.diar.calls), 1)                # ... and ONE diarization call
        self.assertEqual(len(eng.asr.calls[0]), 10)
        # every file is its own transcript with the right length (one word per second)
        for i in range(10):
            turns = res.transcripts[f"clip{i:02d}"]
            self.assertEqual(sum(len(t.text.split()) for t in turns), 6 + i)
            self.assertEqual({t.speaker for t in turns} <= {"speaker_0", "speaker_1"}, True)
        for name in res.transcripts:
            self.assertTrue(os.path.exists(self.dir / "out" / f"{name}.txt"))
            self.assertTrue(os.path.exists(self.dir / "out" / f"{name}.srt"))
        fracs = [f for f, _ in events]
        self.assertEqual(fracs, sorted(fracs))                  # progress never goes backwards
        self.assertEqual(fracs[-1], 1.0)

    def test_thirty_one_files_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.run(helpers.make_engine(), self.files(31, 2), str(self.dir / "out"))
        with self.assertRaises(ValueError):
            pipeline.run(helpers.make_engine(), [], str(self.dir / "out"))
        self.assertEqual(config.MAX_FILES, 30)

    def test_thirty_files_fit_in_one_gpu_batch_with_timings(self):
        eng = helpers.make_engine()
        paths = self.files(30, seconds=4)
        state = JobState()
        res = pipeline.run(eng, paths, str(self.dir / "out"), state=state)
        self.assertEqual(len(res.transcripts), 30)
        self.assertEqual(res.failed, {})
        self.assertEqual([len(c) for c in eng.asr.calls], [30])        # all 30 in ONE batch
        self.assertEqual(list(res.transcripts), [f"clip{i:02d}" for i in range(30)])     # input order kept
        self.assertEqual(len(res.files), 30)
        for i, f in enumerate(res.files):
            self.assertEqual(f.status, "done")
            self.assertAlmostEqual(f.audio_sec, 4 + i, delta=0.2)
            self.assertIsNotNone(f.finished_at)
            self.assertGreaterEqual(f.t_total, f.t_extract)
            self.assertGreater(f.words, 0)
            self.assertEqual(f.batch, 1)
        self.assertEqual(len(res.batches), 1)
        self.assertEqual(res.batches[0]["files"], 30)
        self.assertTrue(os.path.exists(self.dir / "out" / "performance_report.csv"))
        self.assertGreater(res.audio_seconds, 0)

    def test_small_batch_limit_sorts_by_length_and_keeps_every_transcript_with_its_file(self):
        eng = helpers.make_engine()
        paths = self.files(12, seconds=5)                       # 5 .. 16 seconds
        res = pipeline.run(eng, paths, str(self.dir / "out"), batch_seconds=60)
        self.assertEqual(len(res.transcripts), 12)
        self.assertGreater(len(eng.asr.calls), 1)               # several batches
        for call in eng.asr.calls:                              # every batch stays under the padded-audio limit
            self.assertLessEqual(len(call) * max(call), 60 + 1)
        flat = [x for call in eng.asr.calls for x in call]
        self.assertEqual(flat, sorted(flat, reverse=True))      # longest files go first
        for i in range(12):
            turns = res.transcripts[f"clip{i:02d}"]
            self.assertEqual(sum(len(t.text.split()) for t in turns), 5 + i)

    def test_stop_button_marks_remaining_files(self):
        eng = helpers.make_engine()
        state = JobState()
        orig = eng.asr.transcribe

        def stop_after_first(*a, **k):
            out = orig(*a, **k)
            state.cancel()
            return out
        eng.asr.transcribe = stop_after_first
        res = pipeline.run(eng, self.files(10, 5), str(self.dir / "out"), state=state, batch_seconds=60)
        self.assertGreater(len(res.transcripts), 0)
        self.assertGreater(len(res.failed), 0)
        self.assertTrue(all("stopped" in why for why in res.failed.values()))
        self.assertEqual(len(res.transcripts) + len(res.failed), 10)

    def test_bad_files_do_not_stop_the_others(self):
        good = self.files(3)
        corrupt = self.dir / "broken.mp4"
        corrupt.write_bytes(b"this is not a video")
        silent = self.dir / "silent.mp4"
        helpers.make_silent_video(silent, 2)
        video = self.dir / "talk.mp4"
        helpers.make_video(video, 5)
        eng = helpers.make_engine()
        res = pipeline.run(eng, good + [str(corrupt), str(silent), str(video)], str(self.dir / "out"))
        self.assertEqual(sorted(res.transcripts), ["clip00", "clip01", "clip02", "talk"])
        self.assertEqual(sorted(res.failed), ["broken", "silent"])
        self.assertIn("no audio track", res.failed["silent"])
        self.assertEqual(len(eng.asr.calls[0]), 4)              # only the 4 good files reached the GPU

    def test_all_files_bad(self):
        bad = self.dir / "bad.wav"
        bad.write_bytes(b"nope")
        eng = helpers.make_engine()
        res = pipeline.run(eng, [str(bad)], str(self.dir / "out"))
        self.assertEqual(res.transcripts, {})
        self.assertEqual(list(res.failed), ["bad"])
        self.assertEqual(eng.asr.calls, [])                     # GPU never touched

    def test_out_of_memory_splits_batch_and_keeps_order(self):
        eng = helpers.make_engine(oom_above=3)                  # fake GPU can only hold 3 files at once
        paths = self.files(10)
        res = pipeline.run(eng, paths, str(self.dir / "out"))
        self.assertEqual(len(res.transcripts), 10)
        self.assertEqual(res.failed, {})
        self.assertTrue(all(len(c) <= 3 for c in eng.asr.calls))
        self.assertLess(eng.oom_cap, float("inf"))                       # the engine remembers the limit it learned
        self.assertLessEqual(eng.effective_batch_seconds(), eng.oom_cap)
        self.assertLessEqual(eng.effective_batch_seconds(10 ** 6), eng.oom_cap)
        for i in range(10):                                     # each transcript still belongs to ITS file
            turns = res.transcripts[f"clip{i:02d}"]
            self.assertEqual(sum(len(t.text.split()) for t in turns), 6 + i)

    def test_unrecoverable_gpu_error_marks_files_failed(self):
        eng = helpers.make_engine()
        eng.asr.transcribe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        res = pipeline.run(eng, self.files(2, 2), str(self.dir / "out"))
        self.assertEqual(res.transcripts, {})
        self.assertEqual(len(res.failed), 2)

    def test_duplicate_names_get_unique_transcripts(self):
        (self.dir / "a").mkdir()
        (self.dir / "b").mkdir()
        p1, p2 = self.dir / "a" / "clip.wav", self.dir / "b" / "clip.wav"
        helpers.make_wav(p1, 3)
        helpers.make_wav(p2, 5)
        res = pipeline.run(helpers.make_engine(), [str(p1), str(p2)], str(self.dir / "out"))
        self.assertEqual(sorted(res.transcripts), ["clip", "clip_1"])

    def test_long_audio_switches_attention_mode(self):
        eng = helpers.make_engine()
        eng._set_attention(config.LONG_AUDIO_SEC + 1)
        self.assertEqual(eng.asr.attention, "rel_pos_local_attn")
        eng._set_attention(10)
        self.assertEqual(eng.asr.attention, "rel_pos")


class BatchingTests(unittest.TestCase):
    def test_plan_groups_similar_lengths_and_respects_limits(self):
        items = [("a", 600), ("b", 590), ("c", 60), ("d", 55), ("e", 10)]
        plan = batching.plan_batches(items, max_files=30, max_seconds=1300)
        self.assertEqual(plan[0], ["a", "b"])                      # 2 x 600 = 1200 <= 1300, a third would not fit
        flat = [k for b in plan for k in b]
        self.assertEqual(sorted(flat), ["a", "b", "c", "d", "e"])
        for b in plan:
            longest = max(dict(items)[k] for k in b)
            self.assertTrue(len(b) == 1 or len(b) * longest <= 1300)

    def test_file_longer_than_the_limit_gets_its_own_batch(self):
        self.assertEqual(batching.plan_batches([("big", 5000), ("s", 5)], 30, 1000), [["big"], ["s"]])

    def test_max_files_per_batch(self):
        plan = batching.plan_batches([(i, 1.0) for i in range(30)], max_files=8, max_seconds=10 ** 6)
        self.assertEqual([len(b) for b in plan], [8, 8, 8, 6])

    def test_auto_budget_scales_with_gpu_memory(self):
        self.assertEqual(batching.auto_batch_seconds(None), config.CPU_BATCH_SECONDS)
        self.assertLess(batching.auto_batch_seconds(8), batching.auto_batch_seconds(24))
        self.assertEqual(batching.auto_batch_seconds(24), 7200)
        self.assertEqual(batching.auto_batch_seconds(1000), config.BATCH_SECONDS_MAX)


class MediaProbeTests(unittest.TestCase):
    def test_probe_duration(self):
        with helpers.tmpdir() as d:
            p = Path(d) / "x.wav"
            helpers.make_wav(p, 5)
            self.assertAlmostEqual(media.probe_duration(str(p)), 5.0, delta=0.2)
            self.assertIsNone(media.probe_duration(str(Path(d) / "missing.wav")))


class ReportTests(unittest.TestCase):
    def test_fmt_dur(self):
        self.assertEqual(report.fmt_dur(75), "1:15")
        self.assertEqual(report.fmt_dur(3725), "1:02:05")
        self.assertEqual(report.fmt_dur(None), "0:00")

    def test_speaker_stats_and_csv_and_rows(self):
        turns = [Turn("speaker_0", 0, 6, "a b c"), Turn("speaker_1", 6, 8, "d e"), Turn("speaker_0", 8, 10, "f")]
        st_ = report.speaker_stats(turns)
        self.assertEqual([s["speaker"] for s in st_], ["speaker_0", "speaker_1"])
        self.assertAlmostEqual(st_[0]["share"], 0.8)
        self.assertEqual((st_[0]["turns"], st_[0]["words"]), (2, 4))
        files = [FileState("a", status="done", audio_sec=60.0, t_extract=1.0, t_asr=2.0, t_diar=1.0, t_post=0.5, finished_at=9.0, words=5),
                 FileState("b", status="failed", error="boom")]
        csv_text = report.to_csv(files)
        self.assertIn("a,done,60.0,1.0,2.0,1.0,0.5,4.5,13.33,9.0", csv_text.replace("\r", ""))
        self.assertIn("boom", csv_text)
        rows = report.table_rows(files)
        self.assertEqual(rows[0]["Status"], "✅ Done")
        self.assertEqual(rows[0]["Progress"], 100)
        self.assertEqual(rows[1]["Note"], "boom")


class MonitorTests(unittest.TestCase):
    def test_samples_peaks_and_averages(self):
        vals = iter([10.0, 50.0, 30.0] + [30.0] * 1000)
        mon = monitor.SystemMonitor(interval=0.02, sampler=lambda: {"gpu_util": next(vals), "vram_used_mb": 2048.0})
        mon.start()
        time.sleep(0.2)
        mon.stop()
        hist = mon.history()
        self.assertGreaterEqual(len(hist), 4)
        self.assertEqual(hist[0]["gpu_util"], 10.0)
        self.assertEqual(hist[0]["cpu"], None)                      # missing values stay None, never crash
        s = mon.summary()
        self.assertEqual(s["gpu_util_peak"], 50.0)
        self.assertTrue(10.0 < s["gpu_util_avg"] < 50.0)
        self.assertEqual(s["vram_used_mb_peak"], 2048.0)
        self.assertEqual(mon.latest()["vram_used_mb"], 2048.0)
        self.assertTrue(mon.has_gpu)

    def test_a_crashing_sampler_never_breaks_anything(self):
        def boom():
            raise RuntimeError("driver exploded")
        mon = monitor.SystemMonitor(interval=0.02, sampler=boom)
        mon.start()
        time.sleep(0.1)
        mon.stop()
        self.assertGreater(len(mon.history()), 0)
        self.assertFalse(mon.has_gpu)

    def test_real_monitor_without_gpu_runs(self):
        mon = monitor.SystemMonitor(interval=0.05).start()
        time.sleep(0.15)
        mon.stop()
        self.assertGreater(len(mon.history()), 1)
        info = monitor.device_info()
        self.assertIn("cpu_cores", info)


class JobThreadTests(unittest.TestCase):
    def test_background_job_30_files(self):
        with helpers.tmpdir() as d:
            d = Path(d)
            paths = []
            for i in range(30):
                p = d / f"v{i:02d}.wav"
                helpers.make_wav(p, 3 + i % 5)
                paths.append(str(p))
            eng = helpers.make_engine()
            job = Job(eng, paths, str(d / "out"), monitor=monitor.SystemMonitor(interval=0.02, sampler=lambda: {"gpu_util": 42.0})).start()
            seen_running = False
            while not job.done:
                v = job.view()
                seen_running = seen_running or v.running
                time.sleep(0.01)
            self.assertTrue(job.join(5))
            v = job.view()
            self.assertEqual((v.n_done, v.n_failed, v.n_total), (30, 0, 30))
            self.assertFalse(v.running)
            self.assertEqual(v.progress, 1.0)
            self.assertGreater(len(v.history), 0)
            self.assertEqual(v.summary["gpu_util_peak"], 42.0)
            self.assertGreater(v.throughput, 0)
            self.assertEqual(len(job.result.transcripts), 30)

    def test_unexpected_error_does_not_hang_the_page(self):
        with helpers.tmpdir() as d:
            p = Path(d) / "a.wav"
            helpers.make_wav(p, 3)
            eng = helpers.make_engine()
            eng.effective_batch_seconds = lambda requested=None: 1 / 0          # simulate a bug deep inside
            job = Job(eng, [str(p)], str(Path(d) / "out"), monitor=monitor.SystemMonitor(interval=0.05, sampler=lambda: {})).start()
            self.assertTrue(job.join(10))
            v = job.view()
            self.assertIn("ZeroDivisionError", v.error)
            self.assertEqual(v.n_failed, 1)


if __name__ == "__main__":
    unittest.main()
