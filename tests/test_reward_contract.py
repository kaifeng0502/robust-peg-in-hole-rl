"""Training hold boundaries and production tensor rewards without Isaac Sim."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_insertion"))
from evaluation_metrics import EpisodeMetrics, EvaluationCriteria  # noqa: E402
from reward_contract import advance_terminal_hold  # noqa: E402


class TerminalHoldContractTest(unittest.TestCase):
    def compare_with_evaluator(self, intervals):
        criteria = EvaluationCriteria(dt_s=1.0 / 15.0)
        metrics = EpisodeMetrics(criteria, target_depth_m=0.025)
        count, ever, held_ever = 0, False, False
        first_hold_steps = []
        for step, (interval, interval_ever, final) in enumerate(intervals, start=1):
            result = advance_terminal_hold(
                count, ever, held_ever, interval, interval_ever, final, criteria.required_hold_samples
            )
            count, ever, held_ever = result[:3]
            if result.first_held_success:
                first_hold_steps.append(step)
            metrics.update(
                0.0, 0.025 if final else 0.0, 0.0, 0.0,
                interval_success=interval, interval_ever_success=interval_ever,
            )
            evaluated = metrics.result()
            self.assertEqual(result.held_success, evaluated["held_success"])
            self.assertEqual(ever, evaluated["ever_success"])
            self.assertEqual(held_ever, evaluated["had_hold_anytime"])
        return result, first_hold_steps

    def test_exact_fifteen_intervals_and_first_bonus_only_once(self):
        result, first = self.compare_with_evaluator([(True, True, True)] * 20)
        self.assertEqual(first, [15])
        self.assertTrue(result.held_success)

    def test_physics_boundary_failure_breaks_hold_despite_final_success(self):
        intervals = [(True, True, True)] * 14 + [(False, True, True)] + [(True, True, True)] * 15
        result, first = self.compare_with_evaluator(intervals)
        self.assertEqual(first, [30])
        self.assertEqual(result.consecutive_samples, 15)

    def test_recovered_hold_does_not_earn_second_first_bonus(self):
        intervals = [(True, True, True)] * 15 + [(False, False, False)] + [(True, True, True)] * 15
        result, first = self.compare_with_evaluator(intervals)
        self.assertEqual(first, [15])
        self.assertTrue(result.held_success)

    def test_previously_held_but_lost_at_timeout_is_terminal_failure(self):
        result, first = self.compare_with_evaluator([(True, True, True)] * 15 + [(False, False, False)])
        self.assertEqual(first, [15])
        self.assertTrue(result.ever_held_success)
        self.assertFalse(result.held_success)

    def test_success_only_between_control_samples_is_not_a_hold(self):
        result, first = self.compare_with_evaluator([(False, True, False)] * 30)
        self.assertTrue(result.ever_geometry_success)
        self.assertFalse(result.held_success)
        self.assertEqual(first, [])

    def test_failed_final_geometry_cannot_be_overridden_by_interval_flag(self):
        result, _ = self.compare_with_evaluator([(True, True, True)] * 14 + [(True, True, False)])
        self.assertEqual(result.consecutive_samples, 0)

    def test_invalid_sample_requirement_rejected(self):
        for count in [0, -1, 1.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                advance_terminal_hold(0, False, False, True, True, True, count)


def production_class_node(path, class_name, methods=None):
    """Load production method bodies, not copies, without simulation imports."""
    parsed = ast.parse(path.read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.decorator_list = []
    if methods is not None:
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        cls.bases = [ast.Name(id="TensorFactory", ctx=ast.Load())]
    return ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ProductionRewardContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

        class TensorFactory:
            def _reset_idx(self, env_ids):
                pass

            def _reset_buffers(self, env_ids):
                self.ep_succeeded[env_ids] = 0

            def insertion_geometry(self):
                return self.xy, self.depth, torch.zeros((len(self.xy), 3))

            def _get_curr_successes(self, threshold):
                return (self.xy < 0.0025) & (self.depth > self.cfg_task.fixed_asset_cfg.height * (1.0 - threshold))

            def _get_factory_rew_dict(self, successes):
                return {name: torch.zeros_like(self.depth) for name in ("kp_baseline", "kp_coarse", "kp_fine")}, {}

            def _log_factory_metrics(self, terms, successes):
                self.legacy_log_calls += 1
                self.ep_succeeded[successes] = 1

        namespace = {"torch": torch, "TensorFactory": TensorFactory, "advance_terminal_hold": advance_terminal_hold}
        cfg_path = ROOT / "local_insertion" / "env_cfg.py"
        exec(compile(production_class_node(cfg_path, "LocalRewardCfg"), str(cfg_path), "exec"), namespace)
        env_path = ROOT / "local_insertion" / "envs.py"
        selected = {"_get_rewards", "_reset_idx", "evaluation_success_geometry", "_sample_evaluation_geometry"}
        exec(compile(production_class_node(env_path, "LocalInsertionRLEnv", selected), str(env_path), "exec"), namespace)
        cls.reward_type = namespace["LocalRewardCfg"]
        cls.environment_type = namespace["LocalInsertionRLEnv"]

    def make_env(self, profile, count=1):
        torch = self.torch
        env = self.environment_type()
        reward = self.reward_type()
        reward.success_contract = profile
        # Isolate success terms while running the actual production reward body.
        for name in ("keypoint", "alignment", "approach", "progress", "depth", "engage", "action", "action_rate", "commanded_wrench"):
            setattr(reward, f"{name}_scale", 0.0)
        env.cfg = SimpleNamespace(
            reward=reward,
            evaluation_geometry=SimpleNamespace(xy_tolerance_m=0.0025, depth_tolerance_m=0.001),
        )
        env.cfg_task = SimpleNamespace(
            success_threshold=0.04, engage_threshold=0.9, fixed_asset_cfg=SimpleNamespace(height=0.025)
        )
        env.xy = torch.zeros(count)
        env.depth = torch.full((count,), 0.025)
        env.previous_xy_error = env.xy.clone()
        env.relative_z_command = torch.zeros(count)
        env.previous_depth = env.depth.clone()
        env.actions = torch.zeros((count, 6))
        env.previous_rl_action = env.actions.clone()
        env.ep_succeeded = torch.zeros(count, dtype=torch.long)
        env.training_hold_samples = torch.zeros(count, dtype=torch.long)
        env.training_ever_geometry_success = torch.zeros(count, dtype=torch.bool)
        env.training_ever_held_success = torch.zeros(count, dtype=torch.bool)
        env.required_training_hold_samples = 15
        env.evaluation_interval_success = torch.ones(count, dtype=torch.bool)
        env.evaluation_interval_ever_success = torch.zeros(count, dtype=torch.bool)
        env.reset_time_outs = torch.zeros(count, dtype=torch.bool)
        env.extras = {}
        env.legacy_log_calls = 0
        return env

    def test_legacy_default_keeps_instant_first_reward_and_factory_logging(self):
        self.assertEqual(self.reward_type.success_contract, "legacy")
        env = self.make_env("legacy")
        env.reset_time_outs[:] = True
        self.assertEqual(env._get_rewards().item(), 21.0)
        self.assertEqual(env._get_rewards().item(), 1.0)
        self.assertEqual(env.legacy_log_calls, 2)
        self.assertNotIn("episode", env.extras)
        self.assertNotIn("logs_rew_terminal_hold", env.extras)

    def test_fifteenth_good_interval_at_timeout_earns_first_and_terminal_bonus(self):
        env = self.make_env("terminal_hold_v2")
        for _ in range(14):
            self.assertEqual(env._get_rewards().item(), 1.0)
        env.reset_time_outs[:] = True
        self.assertEqual(env._get_rewards().item(), 121.0)
        self.assertEqual(env.legacy_log_calls, 0)
        episode = env.extras["episode"]
        self.assertEqual(episode["terminal_hold_v2/terminal_held_success"].item(), 1.0)
        self.assertEqual(episode["terminal_hold_v2/ever_held_success"].item(), 1.0)

    def test_no_terminal_bonus_before_timeout_and_episode_log_does_not_repeat(self):
        env = self.make_env("terminal_hold_v2")
        for _ in range(14):
            env._get_rewards()
        self.assertEqual(env._get_rewards().item(), 21.0)
        self.assertNotIn("episode", env.extras)
        env.reset_time_outs[:] = True
        self.assertEqual(env._get_rewards().item(), 101.0)
        self.assertIn("episode", env.extras)
        env.reset_time_outs[:] = False
        self.assertEqual(env._get_rewards().item(), 1.0)
        self.assertNotIn("episode", env.extras)

    def test_one_bad_physics_boundary_removes_per_step_and_terminal_rewards(self):
        env = self.make_env("terminal_hold_v2")
        for _ in range(15):
            env._get_rewards()
        env.evaluation_interval_success[:] = False
        env.reset_time_outs[:] = True
        self.assertEqual(env._get_rewards().item(), 0.0)
        episode = env.extras["episode"]
        self.assertEqual(episode["terminal_hold_v2/terminal_held_success"].item(), 0.0)
        self.assertEqual(episode["terminal_hold_v2/final_geometry_success"].item(), 1.0)
        self.assertEqual(episode["terminal_hold_v2/ever_held_success"].item(), 1.0)

    def test_excess_depth_earns_legacy_success_but_no_v2_success(self):
        for profile, expected in [("legacy", 21.0), ("terminal_hold_v2", 0.0)]:
            with self.subTest(profile=profile):
                env = self.make_env(profile)
                env.depth[:] = 0.030
                self.assertEqual(env._get_rewards().item(), expected)

    def test_batched_reset_clears_only_selected_episode_hold_history(self):
        torch = self.torch
        env = self.make_env("terminal_hold_v2", count=2)
        for _ in range(15):
            env._get_rewards()
        for name in ("success_latched", "search_engaged"):
            setattr(env, name, torch.ones(2, dtype=torch.bool))
        for name, width in (("success_hold_pos", 3), ("success_hold_quat", 4), ("search_start", 3), ("search_insert_xy", 2)):
            setattr(env, name, torch.zeros((2, width)))
        env.fingertip_midpoint_pos = torch.zeros((2, 3))
        env.search_time = torch.ones(2)
        env.evaluation_interval_peak_force = torch.zeros(2)
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(env.training_hold_samples.tolist(), [0, 15])
        self.assertEqual(env.training_ever_held_success.tolist(), [False, True])
        self.assertEqual(env.training_ever_geometry_success.tolist(), [False, True])
        env.evaluation_interval_success[:] = True
        self.assertEqual(env._get_rewards().tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
