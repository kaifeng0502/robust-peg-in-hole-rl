"""Candidate provenance checks use copied integration evidence, not new trials."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_ppo_candidates import compare_candidates  # noqa: E402


class PPOCandidateComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        evidence = PROJECT_ROOT / "benchmarks" / "gpu_integration_20261003_autodl"
        self.reference = evidence / "zero_residual"
        self.runs = [root / "candidate_a", root / "candidate_b"]
        for run in self.runs:
            shutil.copytree(evidence / "ppo", run)

    @staticmethod
    def update_json(path, mutate):
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_identical_checkpoint_is_allowed_but_never_labeled_independent_training(self):
        originals = [(run / "metadata.json").read_bytes() for run in self.runs]
        report = compare_candidates(self.reference, self.runs, ["best", "same_artifact"])
        self.assertEqual(report["repeated_checkpoint_groups"], [["best", "same_artifact"]])
        self.assertEqual(report["reference"]["verified_trace_rows"], 1200)
        pair = report["paired_candidates"][0]
        self.assertTrue(pair["same_checkpoint_sha256"])
        self.assertTrue(pair["same_agent_config_sha256"])
        self.assertEqual(pair["counts"], {
            "both_success": 3, "left_only_success": 0, "right_only_success": 0, "both_failure": 5,
        })
        self.assertEqual(pair["left_minus_right_success_time_s_on_both_success"]["mean"], 0)
        self.assertEqual(pair["left_minus_right_peak_commanded_force_n_all_cases"]["mean"], 0)
        self.assertEqual(pair["left_minus_right_task_episode_reward_all_cases"]["mean"], 0)
        self.assertEqual(originals, [(run / "metadata.json").read_bytes() for run in self.runs])
        self.assertTrue(all(row["method"] == "ppo" for row in report["candidates"].values()))

    def test_invalid_labels_source_and_pairwise_initial_state_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            compare_candidates(self.reference, self.runs, ["same", "same"])
        metadata_path = self.runs[1] / "metadata.json"
        original = metadata_path.read_bytes()
        self.update_json(metadata_path, lambda data: data.update(code_revision="different_source"))
        with self.assertRaisesRegex(ValueError, "code_revision differs"):
            compare_candidates(self.reference, self.runs, ["a", "b"])
        metadata_path.write_bytes(original)
        # Both candidates individually match the reference within 1e-6, but
        # their 1.5e-6 separation must not pass candidate-to-candidate analysis.
        for run, offset in zip(self.runs, (0.75e-6, -0.75e-6)):
            def mutate(rows, offset=offset):
                rows[0]["initial_state"]["held_root_state"][0] += offset

            self.update_json(run / "trials.json", mutate)
        with self.assertRaisesRegex(ValueError, "initial-state mismatch.*a_vs_b"):
            compare_candidates(self.reference, self.runs, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
