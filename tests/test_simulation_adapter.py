"""Exercise persistent tensor/reset lifecycles without starting Isaac Sim."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_insertion.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("insertion_evaluator_under_test", SCRIPT_PATH)
EVALUATOR = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(EVALUATOR)


class TensorEnvironment:
    """Minimal simulator boundary with buffers replaced during each step."""

    def __init__(self):
        import torch

        self.torch = torch
        self.unwrapped = self
        self.device = "cpu"
        self.reset_seeds = []
        self.reset_inference_modes = []
        self.step_gradient_modes = []
        self._robot = SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.zeros((1, 9)),
            joint_vel=torch.zeros((1, 9)),
            root_state_w=torch.zeros((1, 13)),
        ))

        def make_asset():
            return SimpleNamespace(
                data=SimpleNamespace(root_state_w=torch.zeros((1, 13))),
                root_physx_view=SimpleNamespace(
                    get_material_properties=lambda: torch.full((1, 1, 3), 0.7)
                ),
            )

        self._held_asset = make_asset()
        self._fixed_asset = make_asset()
        self.episode_friction = torch.full((1,), 0.7)
        self.init_fixed_pos_obs_noise = torch.zeros((1, 3))
        self.fixed_pos_obs_frame = torch.zeros((1, 3))
        self.fingertip_midpoint_pos = torch.zeros((1, 3))
        self.fingertip_midpoint_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.fingertip_midpoint_linvel = torch.zeros((1, 3))
        self.fingertip_midpoint_angvel = torch.zeros((1, 3))
        self.task_prop_gains = torch.ones((1, 6))
        self.task_deriv_gains = torch.ones((1, 6))
        self.actions = torch.zeros((1, 6))
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.evaluation_interval_peak_force = torch.zeros(1)
        self.evaluation_task_reward = torch.zeros(1)
        self.evaluation_interval_success = torch.ones(1, dtype=torch.bool)
        self.evaluation_interval_ever_success = torch.ones(1, dtype=torch.bool)

    def reset(self, seed):
        self.reset_seeds.append(seed)
        self.reset_inference_modes.append(self.torch.is_inference_mode_enabled())
        # These writes fail on the second case if step() produced inference
        # tensors that escaped its context into the persistent simulator state.
        self.evaluation_interval_peak_force.zero_()
        self.evaluation_task_reward.zero_()
        self.episode_length_buf.zero_()
        self.actions.zero_()
        return {"policy": self.torch.zeros((1, 6))}, {}

    def step(self, action):
        torch = self.torch
        self.step_gradient_modes.append(torch.is_grad_enabled())
        self.actions.copy_(action)
        self.episode_length_buf += 1
        self.evaluation_interval_peak_force = torch.maximum(
            self.evaluation_interval_peak_force, torch.full((1,), 2.0)
        )
        self.evaluation_task_reward = torch.zeros_like(self.evaluation_task_reward) + 1.25
        return (
            {"policy": torch.zeros((1, 6))},
            torch.ones(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    def insertion_geometry(self):
        return self.torch.zeros(1), self.torch.full((1,), 0.0245), self.torch.zeros((1, 3))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class SimulationAdapterTest(unittest.TestCase):
    def test_two_cases_reset_persistent_tensors_outside_inference_mode(self):
        env = TensorEnvironment()
        adapter = EVALUATOR.SimulationAdapter(env, SimpleNamespace(is_running=lambda: True))
        criteria = EVALUATOR.EvaluationCriteria(dt_s=0.5, hold_duration_s=1.0)
        samples = []
        rows = []

        def step(action):
            sample = adapter.step(action)
            self.assertIsInstance(sample, EVALUATOR.ControlSample)
            samples.append(sample)
            return sample

        for seed in (17, 29):
            rows.append(EVALUATOR.run_case(
                {"case_id": f"case_{seed}", "seed": seed},
                criteria, 0.0245, 2, adapter.reset, adapter.act, step,
            ))

        self.assertEqual(env.reset_seeds, [17, 29])
        self.assertEqual(env.reset_inference_modes, [False, False])
        self.assertEqual(env.step_gradient_modes, [False] * 4)
        self.assertEqual([sample.episode_step for sample in samples], [1, 2, 1, 2])
        for sample in samples:
            self.assertTrue(sample.interval_success)
            self.assertTrue(sample.interval_ever_success)
            self.assertFalse(sample.terminated)
            self.assertFalse(sample.truncated)
            self.assertEqual(sample.action, [0.0] * 6)
        for row in rows:
            self.assertTrue(row["held_success"])
            self.assertEqual(row["sample_count"], 2)
            self.assertEqual(row["episode_reward"], 2.0)
            self.assertEqual(row["task_episode_reward"], 2.5)
            self.assertEqual(row["peak_commanded_force_n"], 2.0)
            self.assertEqual(row["initial_state"]["actions"], [0.0] * 6)


if __name__ == "__main__":
    unittest.main()
