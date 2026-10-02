"""Provenance gates use synthetic fixtures; these are not simulation results."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "local_insertion"))

from evaluation_cases import (  # noqa: E402
    INITIAL_STATE_KEYS,
    case_set_sha256,
    compare_runs,
    make_case_set,
    read_case_set,
    read_json,
    validate_case_set,
    write_case_set,
    write_json_new,
)


class CaseSetTest(unittest.TestCase):
    def test_deterministic_unique_cases(self):
        first = make_case_set()
        self.assertEqual(first, make_case_set())
        self.assertEqual(len(first["cases"]), 128)
        self.assertEqual(len({case["seed"] for case in first["cases"]}), 128)
        self.assertNotEqual(first["case_set_id"], make_case_set(seed=1)["case_set_id"])

    def test_checksum_is_independent_of_object_key_order(self):
        cases = make_case_set(2)
        reversed_keys = dict(reversed(list(cases.items())))
        self.assertEqual(case_set_sha256(cases), case_set_sha256(reversed_keys))

    def test_tampered_case_is_rejected(self):
        cases = make_case_set(2)
        cases["cases"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_case_set(cases)

    def test_duplicate_ids_or_seeds_rejected_even_with_recomputed_checksum(self):
        for key in ("case_id", "seed"):
            with self.subTest(key=key):
                cases = make_case_set(2)
                cases["cases"][1][key] = cases["cases"][0][key]
                cases["case_set_id"] = case_set_sha256(cases)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    validate_case_set(cases)

    def test_invalid_schema_and_values_are_rejected(self):
        for trials, seed in ((0, 1), (-1, 1), (True, 1), (1, False), (1, -1)):
            with self.subTest(trials=trials, seed=seed), self.assertRaises(ValueError):
                make_case_set(trials, seed)
        for key, value in (("schema_version", True), ("distribution", "different"), ("cases", [])):
            cases = make_case_set(2)
            cases[key] = value
            cases["case_set_id"] = case_set_sha256(cases)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_case_set(cases)

    def test_roundtrip_and_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            cases = make_case_set(2)
            write_case_set(path, cases)
            self.assertEqual(read_case_set(path), cases)
            with self.assertRaises(FileExistsError):
                write_case_set(path, make_case_set(3))
            self.assertEqual(read_case_set(path), cases)

    def test_ambiguous_or_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_json(path)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case_set = make_case_set(2)

    def fixture(self, method):
        criteria = {
            "dt_s": 0.5,
            "hold_duration_s": 1.0,
            "xy_tolerance_m": 0.0025,
            "depth_tolerance_m": 0.001,
        }
        protocol = {
            "name": "local_insertion_held_v1",
            "horizon_steps": 4,
            "criteria": copy.deepcopy(criteria),
            "shared_environment_config": {"synthetic_fixture": True},
        }
        metadata = {
            "schema_version": 1,
            "method": method,
            "case_set_id": self.case_set["case_set_id"],
            "case_set": copy.deepcopy(self.case_set),
            "trial_count": len(self.case_set["cases"]),
            "run_complete": True,
            "protocol_id": self.protocol_digest(protocol),
            "protocol": protocol,
            "code_revision": "revision",
            "runtime_versions": {"python": "3.11", "isaaclab": "2.2.1"},
            "checkpoint_sha256": "a" * 64 if method == "ppo" else None,
            "criteria": criteria,
        }
        rows = [
            {
                "case_id": case["case_id"],
                "reset_seed": case["seed"],
                "initial_state": {key: [0.1, [0.2, 0.3]] for key in INITIAL_STATE_KEYS},
                "held_success": index == 0,
                "ever_success": index == 0,
                "final_success": index == 0,
                "had_hold_anytime": index == 0,
                "success_time_s": 1.0 if index == 0 else None,
                "failure_category": None if index == 0 else "insufficient_depth",
                "peak_commanded_force_n": 1.0,
                "max_depth_m": 0.02,
                "episode_reward": 1.0 + index,
                "task_episode_reward": 3.0 + index,
                "sample_count": 4,
                "elapsed_time_s": 2.0,
            }
            for index, case in enumerate(self.case_set["cases"])
        ]
        return metadata, rows

    @staticmethod
    def protocol_digest(protocol):
        # The producer's canonical digest is verified by the comparison gate.
        return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def save_fixture(self, name, fixture):
        directory = self.root / name
        metadata, rows = fixture
        write_json_new(directory / "metadata.json", metadata)
        write_json_new(directory / "trials.json", rows)
        return directory

    def compare_changed(self, mutation, message=None):
        reference = self.fixture("spiral")
        candidate = self.fixture("ppo")
        mutation(*candidate)
        reference_dir = self.save_fixture("spiral", reference)
        candidate_dir = self.save_fixture("ppo", candidate)
        if message:
            with self.assertRaisesRegex(ValueError, message):
                compare_runs([reference_dir, candidate_dir])
        else:
            with self.assertRaises(ValueError):
                compare_runs([reference_dir, candidate_dir])

    def test_valid_comparison_ignores_trial_order(self):
        reference = self.save_fixture("spiral", self.fixture("spiral"))
        candidate_fixture = self.fixture("ppo")
        candidate_fixture[1].reverse()
        candidate = self.save_fixture("ppo", candidate_fixture)
        summary = compare_runs([reference, candidate])
        self.assertTrue(summary["comparison_valid"])
        self.assertEqual(summary["methods"]["ppo"]["held_success_rate"], 0.5)
        self.assertEqual(summary["methods"]["ppo"]["success_rate"], 0.5)
        self.assertEqual(summary["methods"]["ppo"]["mean_task_episode_reward"], 3.5)
        self.assertEqual(summary["methods"]["ppo"]["mean_episode_reward"], 1.5)
        self.assertEqual(len(summary["methods"]["ppo"]["success_rate_ci95"]), 2)
        self.assertEqual(summary["initial_state_rtol"], 0.0)

    def test_equal_seeds_with_different_physical_states_rejected(self):
        def mutate(metadata, rows):
            rows[0]["initial_state"]["robot_joint_pos"][1][0] += 1e-4

        self.compare_changed(mutate, "initial-state mismatch")

    def test_absolute_tolerance_has_no_relative_component(self):
        reference_fixture = self.fixture("spiral")
        candidate_fixture = self.fixture("ppo")
        reference_fixture[1][0]["initial_state"]["robot_joint_pos"] = [1000.0]
        candidate_fixture[1][0]["initial_state"]["robot_joint_pos"] = [1000.0005]
        reference = self.save_fixture("spiral", reference_fixture)
        candidate = self.save_fixture("ppo", candidate_fixture)
        with self.assertRaisesRegex(ValueError, "initial-state mismatch"):
            compare_runs([reference, candidate])
        self.assertTrue(compare_runs([reference, candidate], atol=0.001)["comparison_valid"])

    def test_small_roundoff_accepted(self):
        reference = self.save_fixture("spiral", self.fixture("spiral"))
        candidate_fixture = self.fixture("ppo")
        candidate_fixture[1][0]["initial_state"]["robot_joint_pos"][0] += 5e-7
        candidate = self.save_fixture("ppo", candidate_fixture)
        self.assertTrue(compare_runs([reference, candidate])["comparison_valid"])

    def test_missing_required_state_rejected_even_if_both_runs_omit_it(self):
        directories = []
        for method in ("spiral", "ppo"):
            fixture = self.fixture(method)
            del fixture[1][0]["initial_state"]["held_root_state"]
            directories.append(self.save_fixture(method, fixture))
        with self.assertRaisesRegex(ValueError, "lacks required physical state"):
            compare_runs(directories)

    def test_state_shape_mismatch_rejected(self):
        self.compare_changed(
            lambda metadata, rows: rows[0]["initial_state"].update(robot_joint_pos=[0.1]),
            "initial-state shape",
        )

    def test_state_boolean_is_not_a_numeric_value(self):
        self.compare_changed(
            lambda metadata, rows: rows[0]["initial_state"].update(robot_joint_pos=[True]),
            "finite numeric",
        )

    def test_nonfinite_metric_or_state_rejected(self):
        reference = self.save_fixture("spiral", self.fixture("spiral"))
        candidate = self.save_fixture("ppo", self.fixture("ppo"))
        data = read_json(candidate / "trials.json")
        data[0]["episode_reward"] = float("nan")
        (candidate / "trials.json").write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare_runs([reference, candidate])

    def test_missing_metric_rejected(self):
        self.compare_changed(lambda metadata, rows: rows[0].pop("episode_reward"), "missing trial metrics")

    def test_inconsistent_success_metrics_rejected(self):
        self.compare_changed(lambda metadata, rows: rows[0].update(final_success=False), "held_success")

    def test_overflowed_json_number_rejected(self):
        reference = self.save_fixture("spiral", self.fixture("spiral"))
        candidate = self.save_fixture("ppo", self.fixture("ppo"))
        path = candidate / "trials.json"
        path.write_text(path.read_text().replace('"episode_reward": 1.0', '"episode_reward": 1e999'))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare_runs([reference, candidate])

    def test_both_runs_truncated_equally_still_rejected(self):
        directories = []
        for method in ("spiral", "ppo"):
            fixture = self.fixture(method)
            fixture[1].pop()
            directories.append(self.save_fixture(method, fixture))
        with self.assertRaisesRegex(ValueError, "incomplete run"):
            compare_runs(directories)

    def test_unfinished_run_rejected(self):
        self.compare_changed(lambda metadata, rows: metadata.update(run_complete=False), "run_complete")

    def test_full_case_list_of_shortened_episodes_rejected(self):
        directories = []
        for method in ("spiral", "ppo"):
            fixture = self.fixture(method)
            for row in fixture[1]:
                row["sample_count"] = 2
                row["elapsed_time_s"] = 1.0
            directories.append(self.save_fixture(method, fixture))
        with self.assertRaisesRegex(ValueError, "sample_count differs"):
            compare_runs(directories)

    def test_wrong_elapsed_duration_rejected(self):
        self.compare_changed(lambda metadata, rows: rows[0].update(elapsed_time_s=1.0), "elapsed_time_s")

    def test_wrong_trial_count_rejected(self):
        self.compare_changed(lambda metadata, rows: metadata.update(trial_count=1), "trial_count")

    def test_duplicate_trial_ids_rejected(self):
        self.compare_changed(lambda metadata, rows: rows.append(copy.deepcopy(rows[0])), "duplicate trial")

    def test_reset_seed_must_match_manifest(self):
        self.compare_changed(lambda metadata, rows: rows[0].update(reset_seed=0), "reset_seed")

    def test_ppo_checkpoint_provenance_required(self):
        self.compare_changed(lambda metadata, rows: metadata.pop("checkpoint_sha256"), "checkpoint")

    def test_protocol_mismatch_rejected(self):
        self.compare_changed(lambda metadata, rows: metadata.update(protocol_id="different"), "protocol_id")

    def test_protocol_body_tampering_rejected(self):
        self.compare_changed(
            lambda metadata, rows: metadata["protocol"].update(horizon_steps=8), "protocol_id checksum"
        )

    def test_missing_embedded_protocol_rejected(self):
        self.compare_changed(lambda metadata, rows: metadata.pop("protocol"), "embedded protocol")

    def test_rehashed_protocol_criteria_must_match_metadata_criteria(self):
        def mutate(metadata, rows):
            metadata["protocol"]["criteria"]["hold_duration_s"] = 0.5
            metadata["protocol_id"] = self.protocol_digest(metadata["protocol"])

        self.compare_changed(mutate, "criteria differ")

    def test_protocol_hold_cannot_exceed_horizon(self):
        def mutate(metadata, rows):
            metadata["criteria"]["hold_duration_s"] = 3.0
            metadata["protocol"]["criteria"]["hold_duration_s"] = 3.0
            metadata["protocol_id"] = self.protocol_digest(metadata["protocol"])

        self.compare_changed(mutate, "shorter than the hold")

    def test_protocol_zero_dt_rejected_after_valid_checksum(self):
        def mutate(metadata, rows):
            metadata["criteria"]["dt_s"] = 0.0
            metadata["protocol"]["criteria"]["dt_s"] = 0.0
            metadata["protocol_id"] = self.protocol_digest(metadata["protocol"])

        self.compare_changed(mutate, "dt_s must be positive")

    def test_runtime_mismatch_rejected(self):
        self.compare_changed(
            lambda metadata, rows: metadata["runtime_versions"].update(isaaclab="different"),
            "runtime_versions",
        )

    def test_criteria_mismatch_rejected(self):
        self.compare_changed(
            lambda metadata, rows: metadata["criteria"].update(hold_duration_s=0.5), "criteria"
        )

    def test_duplicate_method_rejected(self):
        first = self.save_fixture("spiral1", self.fixture("spiral"))
        second = self.save_fixture("spiral2", self.fixture("spiral"))
        with self.assertRaisesRegex(ValueError, "duplicate method"):
            compare_runs([first, second])

    def test_invalid_tolerance_rejected(self):
        for atol in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(atol=atol), self.assertRaisesRegex(ValueError, "atol"):
                compare_runs([], atol=atol)

    def test_cli_mismatch_creates_no_summary(self):
        reference = self.save_fixture("spiral", self.fixture("spiral"))
        candidate_fixture = self.fixture("ppo")
        candidate_fixture[1][0]["initial_state"]["episode_friction"] = [0.7]
        candidate = self.save_fixture("ppo", candidate_fixture)
        output = self.root / "comparison.json"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "compare_evaluations.py"),
             "--runs", str(reference), str(candidate), "--output", str(output)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        self.assertIn("initial-state shape", result.stderr)

    def test_cli_valid_summary_and_existing_summary_is_preserved(self):
        reference = self.save_fixture("spiral", self.fixture("spiral"))
        candidate = self.save_fixture("ppo", self.fixture("ppo"))
        output = self.root / "comparison.json"
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "compare_evaluations.py"),
            "--runs", str(reference), str(candidate), "--output", str(output),
        ]
        first = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        summary = read_json(output)
        self.assertEqual(summary, json.loads(first.stdout))
        self.assertEqual(summary["methods"]["ppo"]["success_rate"], 0.5)
        self.assertEqual(summary["methods"]["ppo"]["mean_task_episode_reward"], 3.5)
        self.assertEqual(summary["methods"]["ppo"]["failure_counts"], {"insufficient_depth": 1})
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_refuses_to_overwrite_case_manifest(self):
        output = self.root / "cases.json"
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "make_evaluation_cases.py"),
            "--trials", "2", "--output", str(output),
        ]
        first = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
