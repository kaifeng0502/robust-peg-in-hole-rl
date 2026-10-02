import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_insertion"))

from evaluation_loop import ControlSample, run_case
from evaluation_metrics import EvaluationCriteria


class FakeBackend:
    def __init__(self, samples):
        self.samples = samples
        self.reset_seeds = []
        self.observations = []
        self.actions = []
        self.initial_state = {"joint_pos": [0.5]}

    def reset(self, seed):
        self.reset_seeds.append(seed)
        return "initial_observation", self.initial_state

    def act(self, observation):
        self.observations.append(observation)
        return ("opaque_action", len(self.observations))

    def step(self, action):
        self.actions.append(action)
        # Deliberately mutate the reset output: the loop must preserve a snapshot.
        self.initial_state["joint_pos"][0] = 9.0
        return self.samples[len(self.actions) - 1]


class EvaluationLoopTest(unittest.TestCase):
    def setUp(self):
        self.case = {"case_id": "case_000001", "seed": 12345}
        self.criteria = EvaluationCriteria(dt_s=0.5, hold_duration_s=1.0)

    def sample(self, index, success=True, **overrides):
        sample = ControlSample(
            observation=f"observation_{index}",
            xy_error_m=0.0,
            depth_m=0.0245 if success else 0.0,
            reward=1.0,
            task_reward=1.25,
            commanded_force_n=2.0,
            interval_success=success,
            interval_ever_success=success,
            terminated=False,
            truncated=False,
            episode_step=index,
            action=[0.0] * 6,
        )
        return replace(sample, **overrides)

    def run_backend(self, backend, horizon, trace=None):
        return run_case(
            self.case,
            self.criteria,
            0.0245,
            horizon,
            backend.reset,
            backend.act,
            backend.step,
            trace,
        )

    def test_transient_hold_then_withdrawal_is_terminal_failure(self):
        backend = FakeBackend([self.sample(1), self.sample(2), self.sample(3, False)])
        result = self.run_backend(backend, 3)
        self.assertTrue(result["ever_success"])
        self.assertTrue(result["had_hold_anytime"])
        self.assertFalse(result["final_success"])
        self.assertFalse(result["held_success"])
        self.assertEqual(result["failure_category"], "lost_insertion")
        self.assertIsNone(result["success_time_s"])

    def test_exact_budget_seed_snapshot_and_observation_feedback(self):
        backend = FakeBackend([self.sample(1), self.sample(2), self.sample(3, False)])
        trace = []
        result = self.run_backend(backend, 2, trace.append)
        self.assertEqual(backend.reset_seeds, [12345])
        self.assertEqual(backend.actions, [("opaque_action", 1), ("opaque_action", 2)])
        self.assertEqual(backend.observations, ["initial_observation", "observation_1"])
        self.assertEqual(result["initial_state"], {"joint_pos": [0.5]})
        self.assertEqual(result["case_id"], self.case["case_id"])
        self.assertEqual(result["reset_seed"], 12345)
        self.assertEqual(result["sample_count"], 2)
        self.assertTrue(result["held_success"])
        self.assertEqual(result["elapsed_time_s"], 1.0)
        self.assertEqual(result["episode_reward"], 2.0)
        self.assertEqual(result["task_episode_reward"], 2.5)
        self.assertEqual([row["step"] for row in trace], [1, 2])
        self.assertEqual(trace[-1]["time_s"], 1.0)
        self.assertTrue(trace[-1]["held_success"])
        self.assertEqual(trace[-1]["action"], [0.0] * 6)
        self.assertEqual(trace[-1]["task_reward"], 1.25)

    def test_reset_contamination_rejected_before_geometry_is_scored(self):
        for flag in ("terminated", "truncated"):
            with self.subTest(flag=flag):
                contaminated = self.sample(1, **{flag: True}, depth_m=math.nan)
                backend = FakeBackend([contaminated, self.sample(2)])
                trace = []
                # The reset error takes precedence over the invalid post-reset
                # geometry; scoring that sample would instead raise ValueError.
                with self.assertRaisesRegex(RuntimeError, "cannot be scored"):
                    self.run_backend(backend, 2, trace.append)
                self.assertEqual(len(backend.actions), 1)
                self.assertEqual(trace, [])

    def test_wrong_counter_rejected_before_scoring_even_without_done_flag(self):
        for count in (0, 1, 3, True):
            with self.subTest(episode_step=count):
                backend = FakeBackend(
                    [self.sample(1), self.sample(2, episode_step=count, depth_m=math.nan)]
                )
                trace = []
                with self.assertRaisesRegex(RuntimeError, "expected episode step 2"):
                    self.run_backend(backend, 2, trace.append)
                self.assertEqual(len(trace), 1)

    def test_horizon_too_short_rejected_without_reset(self):
        for horizon in (0, 1, -1, 2.5, True):
            with self.subTest(horizon=horizon):
                backend = FakeBackend([])
                with self.assertRaisesRegex(ValueError, "required hold"):
                    self.run_backend(backend, horizon)
                self.assertEqual(backend.reset_seeds, [])
                self.assertEqual(backend.actions, [])

    def test_nonfinite_task_reward_and_accumulation_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(task_reward=invalid):
                backend = FakeBackend([self.sample(1, task_reward=invalid), self.sample(2)])
                trace = []
                with self.assertRaisesRegex(ValueError, "task reward must be finite"):
                    self.run_backend(backend, 2, trace.append)
                self.assertEqual(trace, [])
        backend = FakeBackend([self.sample(1, task_reward=1e308), self.sample(2, task_reward=1e308)])
        trace = []
        with self.assertRaisesRegex(ValueError, "task reward must be finite"):
            self.run_backend(backend, 2, trace.append)
        self.assertEqual(len(trace), 1)

    def test_nonfinite_episode_reward_rejected(self):
        backend = FakeBackend([self.sample(1, reward=math.nan), self.sample(2)])
        trace = []
        with self.assertRaisesRegex(ValueError, "reward must be finite"):
            self.run_backend(backend, 2, trace.append)
        self.assertEqual(trace, [])

    def test_physics_interval_flags_break_hold_at_valid_final_geometry(self):
        backend = FakeBackend(
            [self.sample(1), self.sample(2, interval_success=False, interval_ever_success=True)]
        )
        result = self.run_backend(backend, 2)
        self.assertTrue(result["final_success"])
        self.assertTrue(result["ever_success"])
        self.assertFalse(result["held_success"])
        self.assertEqual(result["trailing_hold_duration_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
