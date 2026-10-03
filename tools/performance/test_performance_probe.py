"""Deterministic, CPU-only checks for the isolated performance probe."""

import json
import threading
import unittest
from contextlib import contextmanager

from performance_probe import Probe


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Example:
    def __init__(self, clock):
        self.clock = clock

    def child(self, value, *, amount=2.0):
        self.clock.advance(amount)
        return value

    def parent(self, value, *, offset):
        self.clock.advance(1.0)
        result = self.child(value, amount=2.0)
        self.clock.advance(3.0)
        return result + offset

    def broken(self):
        self.clock.advance(4.0)
        raise ValueError("original failure")

    def catches_child(self):
        self.clock.advance(1.0)
        try:
            self.broken()
        except ValueError:
            pass
        self.clock.advance(1.0)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.example = Example(self.clock)
        self.probe = Probe(clock=self.clock)
        self.addCleanup(self.probe.restore)

    def test_nested_exclusive_does_not_double_count(self):
        self.probe.wrap(self.example, "parent", "parent")
        self.probe.wrap(self.example, "child", "child")
        self.assertEqual(self.example.parent(10, offset=7), 17)
        result = self.probe.summary()
        self.assertEqual(result["root_wall_s"], 6.0)
        self.assertEqual(result["root_calls"], 1)
        self.assertEqual(result["completed_calls"], 2)
        self.assertEqual(result["regions"]["parent"]["inclusive_s"], 6.0)
        self.assertEqual(result["regions"]["parent"]["exclusive_s"], 4.0)
        self.assertEqual(result["regions"]["child"]["exclusive_s"], 2.0)
        self.assertEqual(
            sum(region["exclusive_s"] for region in result["regions"].values()), 6.0
        )
        json.dumps(result, allow_nan=False)

    def test_exception_propagates_and_nesting_is_cleared(self):
        self.probe.wrap(self.example, "broken", "broken")
        self.probe.wrap(self.example, "child", "child")
        with self.assertRaisesRegex(ValueError, "original failure"):
            self.example.broken()
        self.assertEqual(self.example.child("ok", amount=3.0), "ok")
        result = self.probe.summary()
        self.assertEqual(result["active_calls"], 0)
        self.assertEqual(result["root_calls"], 2)
        self.assertEqual(result["root_wall_s"], 7.0)
        self.assertEqual(result["regions"]["broken"]["errors"], 1)
        self.assertEqual(result["regions"]["child"]["errors"], 0)

    def test_caught_nested_exception_still_subtracts_child_time(self):
        self.probe.wrap(self.example, "catches_child", "parent")
        self.probe.wrap(self.example, "broken", "child")
        self.example.catches_child()
        result = self.probe.summary()["regions"]
        self.assertEqual(result["parent"]["exclusive_s"], 2.0)
        self.assertEqual(result["parent"]["errors"], 0)
        self.assertEqual(result["child"]["exclusive_s"], 4.0)
        self.assertEqual(result["child"]["errors"], 1)

    def test_restore_removes_inherited_method_shadow(self):
        original_function = self.example.child.__func__
        self.assertNotIn("child", vars(self.example))
        self.probe.wrap(self.example, "child", "child")
        self.assertIn("child", vars(self.example))
        self.probe.restore()
        self.probe.restore()
        self.assertNotIn("child", vars(self.example))
        self.assertIs(self.example.child.__func__, original_function)
        self.assertIs(self.example.child.__self__, self.example)

    def test_restore_retains_exact_original_instance_attribute(self):
        def original(value, *, amount):
            return value, amount

        self.example.child = original
        self.probe.wrap(self.example, "child", "child")
        self.assertEqual(self.example.child("x", amount=11), ("x", 11))
        self.probe.restore()
        self.assertIs(vars(self.example)["child"], original)

    def test_multiple_wrappers_capture_their_own_callable_and_label(self):
        other = Example(self.clock)
        self.probe.wrap(self.example, "child", "one")
        self.probe.wrap(other, "parent", "two")
        self.assertEqual(self.example.child(3, amount=5), 3)
        self.assertEqual(other.parent(9, offset=2), 11)
        result = self.probe.summary()["regions"]
        self.assertEqual(result["one"]["inclusive_s"], 5.0)
        self.assertEqual(result["two"]["inclusive_s"], 6.0)

    def test_optional_regions_exit_even_on_exception(self):
        events = []

        @contextmanager
        def region(label):
            events.append(("enter", label))
            try:
                yield
            finally:
                events.append(("exit", label))

        self.probe = Probe(clock=self.clock, region_factory=region)
        self.addCleanup(self.probe.restore)
        self.probe.wrap(self.example, "broken", "failure")
        with self.assertRaises(ValueError):
            self.example.broken()
        self.assertEqual(events, [("enter", "failure"), ("exit", "failure")])
        self.assertEqual(self.probe.summary()["active_calls"], 0)

    def test_clear_resets_measurements_but_keeps_wrapper(self):
        self.probe.wrap(self.example, "child", "child")
        self.example.child(1)
        self.probe.clear()
        self.assertEqual(self.probe.summary()["regions"], {})
        self.assertEqual(self.probe.summary()["root_wall_s"], 0.0)
        self.example.child(2, amount=3.0)
        self.assertEqual(self.probe.summary()["regions"]["child"]["count"], 1)

    def test_clear_inside_call_is_rejected_and_exception_cleanup_works(self):
        self.example.callback = self.probe.clear
        self.probe.wrap(self.example, "callback", "callback")
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.example.callback()
        self.assertEqual(self.probe.summary()["active_calls"], 0)
        self.probe.clear()

    def test_snapshot_is_independent(self):
        self.probe.wrap(self.example, "child", "child")
        self.example.child(1)
        snapshot = self.probe.summary()
        snapshot["regions"]["child"]["count"] = 900
        self.assertEqual(self.probe.summary()["regions"]["child"]["count"], 1)

    def test_duplicate_wrap_is_rejected_without_losing_original(self):
        self.probe.wrap(self.example, "child", "first")
        with self.assertRaisesRegex(ValueError, "Already wrapped"):
            self.probe.wrap(self.example, "child", "second")
        self.example.child(1)
        self.assertEqual(set(self.probe.summary()["regions"]), {"first"})

    def test_clock_exception_cleans_frame(self):
        calls = 0

        def unreliable_clock():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("clock failed")
            return self.clock()

        self.probe = Probe(clock=unreliable_clock)
        self.addCleanup(self.probe.restore)
        self.probe.wrap(self.example, "child", "child")
        with self.assertRaisesRegex(RuntimeError, "clock failed"):
            self.example.child(1)
        self.assertEqual(self.probe.summary()["active_calls"], 0)
        self.example.child(2)
        self.assertEqual(self.probe.summary()["root_calls"], 1)

    def test_recursive_same_label_has_exclusive_total_equal_to_root_time(self):
        def recurse(depth):
            self.clock.advance(1.0)
            if depth:
                self.example.recurse(depth - 1)

        self.example.recurse = recurse
        self.probe.wrap(self.example, "recurse", "recursive")
        self.example.recurse(2)
        result = self.probe.summary()
        self.assertEqual(result["root_wall_s"], 3.0)
        self.assertEqual(result["regions"]["recursive"]["count"], 3)
        self.assertEqual(result["regions"]["recursive"]["inclusive_s"], 6.0)
        self.assertEqual(result["regions"]["recursive"]["exclusive_s"], 3.0)

    def test_concurrent_calls_are_independent_roots(self):
        barrier = threading.Barrier(2)
        local_clock = threading.local()
        failures = []

        def clock():
            return getattr(local_clock, "now", 0.0)

        def overlap():
            barrier.wait(timeout=2)
            local_clock.now = 5.0
            barrier.wait(timeout=2)

        self.probe = Probe(clock=clock)
        self.addCleanup(self.probe.restore)
        self.example.overlap = overlap
        self.probe.wrap(self.example, "overlap", "parallel")

        def run():
            try:
                self.example.overlap()
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        result = self.probe.summary()
        self.assertEqual(result["root_calls"], 2)
        self.assertEqual(result["active_calls"], 0)
        self.assertEqual(result["root_wall_s"], 10.0)
        self.assertEqual(result["regions"]["parallel"]["exclusive_s"], 10.0)


if __name__ == "__main__":
    unittest.main()
