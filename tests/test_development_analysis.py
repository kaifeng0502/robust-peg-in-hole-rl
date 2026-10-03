"""Offline diagnostic integrity checks; synthetic fixtures are not performance evidence."""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_development import (  # noqa: E402
    EpisodeMetrics,
    EvaluationCriteria,
    _paired,
    analyze_runs,
    bin_index,
    initial_features,
    verify_traces,
)


class DevelopmentAnalysisTest(unittest.TestCase):
    @staticmethod
    def successful_trace():
        criteria = EvaluationCriteria(dt_s=0.5)
        metadata = {"criteria": vars(criteria), "protocol": {"horizon_steps": 3}}
        accumulator = EpisodeMetrics(criteria, 0.025)
        samples = []
        for step in range(1, 4):
            accumulator.update(0.001, 0.025, 1.0, 2.0, interval_success=True, interval_ever_success=True)
            result = accumulator.result()
            samples.append({
                "case_id": "a", "step": step, "time_s": step * 0.5,
                "target_depth_m": 0.025, "xy_error_m": 0.001, "depth_m": 0.025,
                "reward": 1.0, "task_reward": 1.0, "commanded_force_n": 2.0,
                "interval_success": True, "interval_ever_success": True,
                "final_success": result["final_success"], "held_success": result["held_success"],
                "action": [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            })
        trials = [{**accumulator.result(), "case_id": "a", "task_episode_reward": 3.0}]
        return metadata, samples, trials

    def test_axis_tilt_excludes_axial_twist_and_quaternion_sign(self):
        angle = math.radians(3)
        state = {
            "held_root_state": [0.003, 0.004, 0, math.cos(angle / 2), math.sin(angle / 2), 0, 0],
            "fixed_root_state": [0, 0, 0, 0, 0, 0, 1],  # Pure 180-degree yaw.
            "episode_friction": 0.8,
        }
        features = initial_features(state)
        self.assertAlmostEqual(features["initial_root_xy_mm"], 5.0)
        self.assertAlmostEqual(features["initial_axis_tilt_deg"], 3.0)
        state["held_root_state"][3:7] = [-value for value in state["held_root_state"][3:7]]
        self.assertEqual(initial_features(state), features)
        state["held_root_state"][3:7] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "Quaternion norm"):
            initial_features(state)

    def test_fixed_bin_boundaries_have_no_gaps_and_preserve_nominal_friction_limit(self):
        for value, expected in ((0, 0), (2.499, 0), (2.5, 1), (5, 2), (10, 3)):
            self.assertEqual(bin_index("initial_root_xy_mm", value), expected)
        for value, expected in ((0.3, 0), (0.6, 1), (0.9, 2), (1.2, 2), (1.20001, 3)):
            self.assertEqual(bin_index("friction", value), expected)
        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                bin_index("friction", value)

    def test_full_committed_integration_traces_are_verified_and_paired(self):
        base = PROJECT_ROOT / "benchmarks" / "gpu_integration_20261003_autodl"
        report = analyze_runs([base / name for name in ("zero_residual", "spiral", "ppo")])
        self.assertEqual(report["verified_trace_rows"], {"zero_residual": 1200, "spiral": 1200, "ppo": 1200})
        self.assertTrue(any("does not re-evaluate raw 120 Hz" in limit for limit in report["interpretation_limits"]))
        self.assertEqual(
            {name: data["overall"]["held_successes"] for name, data in report["methods"].items()},
            {"zero_residual": 2, "spiral": 4, "ppo": 3},
        )
        paired = report["paired"]["ppo_vs_spiral"]
        self.assertEqual(paired["counts"], {
            "both_success": 2, "left_only_success": 1, "right_only_success": 2, "both_failure": 3,
        })
        for method in report["methods"].values():
            for bins in method["strata"].values():
                self.assertEqual(sum(row["cases"] for row in bins), 8)
                for row in bins:
                    if row["cases"] == 0:
                        self.assertIsNone(row["held_success_rate"])
                        self.assertIsNone(row["wilson95"])

    def test_trace_corruption_is_rejected_even_when_trial_summary_is_present(self):
        metadata, samples, trials = self.successful_trace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trajectories.jsonl"

            def save(rows):
                trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            save(samples)
            self.assertEqual(verify_traces(root, metadata, trials), 3)
            variants = [samples[:-1], [samples[0], samples[0], *samples[1:]]]
            corrupted = copy.deepcopy(samples)
            corrupted[-1]["depth_m"] = 0.030
            variants.append(corrupted)
            for rows in variants:
                save(rows)
                with self.assertRaises(ValueError):
                    verify_traces(root, metadata, trials)
            save(samples)
            altered_trials = copy.deepcopy(trials)
            altered_trials[0]["peak_commanded_force_n"] = 1.0
            with self.assertRaisesRegex(ValueError, "metric mismatch"):
                verify_traces(root, metadata, altered_trials)

    def test_duplicate_keys_and_nonfinite_json_are_rejected_before_scoring(self):
        metadata, samples, trials = self.successful_trace()
        first = json.dumps(samples[0])
        variants = {
            "duplicate reward": '{"reward": 9, ' + first[1:],
            "nested duplicate": '{"extra": {"a": 0, "a": 1}, ' + first[1:],
        }
        for token in ("NaN", "Infinity", "-Infinity", "1e400", "-1e400"):
            variants[token] = '{"extra": ' + token + ', ' + first[1:]
        remaining = "".join(json.dumps(row) + "\n" for row in samples[1:])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, line in variants.items():
                with self.subTest(label=label):
                    (root / "trajectories.jsonl").write_text(line + "\n" + remaining, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        verify_traces(root, metadata, trials)

    def test_trace_metric_types_and_actions_are_validated(self):
        metadata, samples, trials = self.successful_trace()
        variants = []
        for field in ("time_s", "target_depth_m", "xy_error_m", "depth_m", "reward", "task_reward", "commanded_force_n"):
            for value in (str(samples[0][field]), True, 10**400):
                row = copy.deepcopy(samples[0])
                row[field] = value
                variants.append((f"{field}={value!r}", row))
        for action in (
            None, [], [0.0] * 5, [0.0] * 7, [[0.0] * 6],
            ["0", 0, 0, 0, 0, 0], [True, 0, 0, 0, 0, 0],
            [float("nan"), 0, 0, 0, 0, 0], [float("inf"), 0, 0, 0, 0, 0],
            [1.0001, 0, 0, 0, 0, 0], [-1.0001, 0, 0, 0, 0, 0],
        ):
            row = copy.deepcopy(samples[0])
            row["action"] = action
            variants.append((f"action={action!r}", row))
        row = copy.deepcopy(samples[0])
        del row["action"]
        variants.append(("missing action", row))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, row in variants:
                with self.subTest(label=label):
                    (root / "trajectories.jsonl").write_text(
                        "".join(json.dumps(item) + "\n" for item in [row, *samples[1:]]), encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        verify_traces(root, metadata, trials)

    def test_interval_all_success_requires_interval_any_success(self):
        metadata, samples, trials = self.successful_trace()
        # This corruption used to leave every reconstructed trial metric unchanged.
        samples[0]["interval_ever_success"] = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trajectories.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "interval_success requires interval_ever_success"):
                verify_traces(root, metadata, trials)

    def test_paired_time_uses_only_common_successes(self):
        def row(success, time):
            return {
                "held_success": success, "success_time_s": time,
                "final_xy_error_m": 0.001, "peak_commanded_force_n": 2.0,
            }

        left = {"a": row(True, 5.0), "b": row(True, 1.0), "c": row(False, None)}
        right = {"a": row(True, 6.0), "b": row(False, None), "c": row(True, 9.0)}
        result = _paired(left, right)
        self.assertEqual(result["left_minus_right_success_rate"], 0.0)
        times = result["left_minus_right_success_time_s_on_both_success"]
        self.assertEqual(times["count"], 1)
        self.assertEqual(times["mean"], -1.0)


if __name__ == "__main__":
    unittest.main()
