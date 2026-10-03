"""Pure Python parser and process-barrier checks; no Isaac or torch imports."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import threading
import time as real_time
import unittest
from unittest import mock
import uuid

import profile_environment as benchmark


class FastTime:
    """Use real scheduling with accelerated benchmark deadlines/start delays."""

    def __init__(self, scale=1000.0):
        self.scale = scale
        self.origin = real_time.monotonic()

    def monotonic(self):
        return (real_time.monotonic() - self.origin) * self.scale

    def time(self):
        return 1700000000.0 + self.monotonic()

    def sleep(self, seconds):
        real_time.sleep(min(max(seconds / self.scale, 0.00001), 0.001))


class BenchmarkArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        environment = self.project / "local_insertion" / "envs.py"
        environment.parent.mkdir(parents=True)
        environment.write_text("# Parser fixture only.\n")
        self.output = self.root / "new_output"

    def arguments(self, *extra):
        return ["--project", str(self.project), "--output", str(self.output), *extra]

    def assert_rejected(self, *extra, expected):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            benchmark.parse_args(self.arguments(*extra))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(expected, stderr.getvalue())

    def test_defaults_and_valid_diagnostic_window(self):
        args = benchmark.parse_args(self.arguments())
        self.assertEqual(args.steps, 600)
        self.assertEqual(args.warmup_steps, 150)
        self.assertEqual(args.mode, "throughput")
        self.assertIsNone(args.barrier)
        args = benchmark.parse_args(
            self.arguments("--mode", "diagnostic", "--steps", "450", "--trace-last-steps", "30")
        )
        self.assertEqual(args.trace_last_steps, 30)

    def test_fewer_than_450_measured_steps_are_rejected(self):
        for steps in (0, 150, 300, 449):
            with self.subTest(steps=steps):
                self.assert_rejected("--steps", str(steps), expected="at least 450 steps")

    def test_incomplete_measured_episodes_are_rejected(self):
        for steps in (451, 599, 601):
            with self.subTest(steps=steps):
                self.assert_rejected("--steps", str(steps), expected="multiples of 150")

    def test_incomplete_warmup_episodes_are_rejected(self):
        for steps in (0, 149, 151, 299):
            with self.subTest(steps=steps):
                self.assert_rejected("--warmup-steps", str(steps), expected="complete 150-step episodes")

    def test_throughput_trace_is_rejected(self):
        self.assert_rejected("--trace-last-steps", "1", expected="throughput runs must be uninstrumented")

    def test_invalid_or_missing_barrier_token_is_rejected(self):
        common = ["--barrier", str(self.root / "barrier"), "--participants", "2"]
        self.assert_rejected(*common, expected="fresh --barrier-token UUID")
        for token in ("", "old-run", "../../other", "12345"):
            with self.subTest(token=token):
                self.assert_rejected(*common, "--barrier-token", token, expected="fresh --barrier-token UUID")

    def test_barrier_options_require_concurrent_workers(self):
        self.assert_rejected("--participants", "2", expected="shared barrier is required")
        self.assert_rejected("--barrier", str(self.root / "barrier"), expected="shared barrier is required")
        self.assert_rejected("--barrier-token", str(uuid.uuid4()), expected="requires --barrier")

    def test_invalid_worker_and_participant_counts_are_rejected(self):
        self.assert_rejected("--participants", "0", expected="Invalid barrier participant")
        self.assert_rejected("--worker-id", "-1", expected="Invalid barrier participant")
        self.assert_rejected("--worker-id", "1", expected="Invalid barrier participant")

    def test_existing_output_is_rejected(self):
        self.output.mkdir()
        self.assert_rejected(expected="must be a new directory")


class BenchmarkBarrierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        project = self.root / "project"
        environment = project / "local_insertion" / "envs.py"
        environment.parent.mkdir(parents=True)
        environment.write_text("# Parser fixture only.\n")
        self.token = str(uuid.uuid4())
        self.barrier = self.root / "barrier"
        self.args = [
            benchmark.parse_args([
                "--project", str(project),
                "--output", str(self.root / f"output_{worker}"),
                "--barrier", str(self.barrier),
                "--barrier-token", self.token,
                "--participants", "2",
                "--worker-id", str(worker),
            ])
            for worker in range(2)
        ]
        self.fast_time = FastTime()
        self.clock_patch = mock.patch.object(benchmark, "time", self.fast_time)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

    def write_manifest(self, *, token=None, participants=2):
        self.barrier.mkdir()
        benchmark.write_json(self.barrier / "manifest.json", {
            "token": self.token if token is None else token,
            "participants": participants,
        })

    def test_fresh_two_workers_pass_the_same_start_barrier(self):
        outcomes = {}

        def run(worker):
            try:
                benchmark.wait_for_peers(self.args[worker])
                outcomes[worker] = self.fast_time.monotonic()
            except BaseException as exc:
                outcomes[worker] = exc

        # Start the follower first to exercise waiting for the leader's manifest.
        threads = [threading.Thread(target=run, args=(worker,), daemon=True) for worker in (1, 0)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        self.assertFalse(any(thread.is_alive() for thread in threads), "Barrier worker did not terminate")
        self.assertEqual(set(outcomes), {0, 1})
        for worker, outcome in outcomes.items():
            self.assertNotIsInstance(outcome, BaseException, f"Worker {worker}: {outcome}")
        manifest = json.loads((self.barrier / "manifest.json").read_text())
        self.assertEqual(manifest, {"token": self.token, "participants": 2})
        start_at = json.loads((self.barrier / "go.json").read_text())["start_at_monotonic_s"]
        for worker in range(2):
            self.assertGreaterEqual(outcomes[worker], start_at)
            self.assertEqual(
                (self.barrier / f"worker_{worker}.ready").read_text(),
                str(self.args[worker].output.resolve()),
            )

    def test_leader_rejects_existing_directory(self):
        self.barrier.mkdir()
        with self.assertRaises(FileExistsError):
            benchmark.wait_for_peers(self.args[0])
        self.assertFalse((self.barrier / "manifest.json").exists())

    def test_follower_rejects_stale_manifest(self):
        self.write_manifest(token=str(uuid.uuid4()))
        with self.assertRaisesRegex(ValueError, "different benchmark run"):
            benchmark.wait_for_peers(self.args[1])
        self.assertFalse((self.barrier / "worker_1.ready").exists())

    def test_follower_rejects_manifest_participant_mismatch(self):
        self.write_manifest(participants=3)
        with self.assertRaisesRegex(ValueError, "different benchmark run"):
            benchmark.wait_for_peers(self.args[1])

    def test_follower_rejects_existing_go_signal(self):
        self.write_manifest()
        benchmark.write_json(self.barrier / "go.json", {"start_at_monotonic_s": 0.0})
        with self.assertRaisesRegex(ValueError, "existing start signal"):
            benchmark.wait_for_peers(self.args[1])
        self.assertFalse((self.barrier / "worker_1.ready").exists())

    def test_duplicate_worker_is_rejected_without_overwriting_ready_file(self):
        self.write_manifest()
        ready = self.barrier / "worker_1.ready"
        ready.write_text("original-worker")
        with self.assertRaises(FileExistsError):
            benchmark.wait_for_peers(self.args[1])
        self.assertEqual(ready.read_text(), "original-worker")
        self.assertFalse((self.barrier / "go.json").exists())

    def test_absent_leader_times_out_quickly_under_mocked_clock(self):
        with mock.patch.object(benchmark, "time", FastTime(scale=10000.0)):
            with self.assertRaisesRegex(TimeoutError, "leader"):
                benchmark.wait_for_peers(self.args[1])

    def test_absent_peer_times_out_quickly_under_mocked_clock(self):
        with mock.patch.object(benchmark, "time", FastTime(scale=10000.0)):
            with self.assertRaisesRegex(TimeoutError, "did not become ready"):
                benchmark.wait_for_peers(self.args[0])

    def test_single_worker_does_not_touch_barrier(self):
        self.args[0].barrier = None
        benchmark.wait_for_peers(self.args[0])
        self.assertFalse(self.barrier.exists())


if __name__ == "__main__":
    unittest.main()
