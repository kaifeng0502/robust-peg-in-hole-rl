"""CPU numerical and call-order checks; no Isaac Sim import is required."""

from __future__ import annotations

import ast
import copy
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from target_updates import install, make_constants, select_and_bound_targets, update_success_hold


def original_apply_action():
    """Use the actual project method as oracle, without its Isaac-only imports."""
    project = Path(os.environ.get("INSERTION_PROJECT", Path(__file__).resolve().parents[2]))
    project_file = project / "local_insertion/envs.py"
    module = ast.parse(project_file.read_text())
    env_class = next(node for node in module.body if isinstance(node, ast.ClassDef)
                     and node.name == "LocalInsertionRLEnv")
    method = next(node for node in env_class.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_apply_action")
    namespace = {"torch": torch, "math": math,
                 "torch_utils": SimpleNamespace(quat_from_euler_xyz=quat_from_euler_xyz)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(project_file), "exec"), namespace)
    return namespace["_apply_action"]


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((cr * cp * cy + sr * sp * sy,
                        sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy), dim=-1)


class FakeEnv:
    def __init__(self, count=4, mode="residual", hold=True, dtype=None):
        dtype = dtype or torch.float32
        self.device = "cpu"
        self.num_envs = count
        self.cfg = SimpleNamespace(
            hold=SimpleNamespace(enabled=hold, downward_margin_m=0.0003,
                                 max_xy_error_m=0.004, max_z_error_m=0.006),
            ctrl=SimpleNamespace(pos_action_bounds=[0.025, 0.025, 0.025]),
            controller_mode=mode, nominal_tool_to_peg_base_m=0.05,
            nominal_insertion_depth_m=0.017, residual_xy_span_m=0.015,
            residual_z_span_m=0.006, residual_roll_pitch_span_rad=0.1,
            residual_yaw_span_rad=0.2,
        )
        self.cfg_task = SimpleNamespace(success_threshold=0.04)
        self._robot = SimpleNamespace(_data=SimpleNamespace(_sim_timestamp=1))
        self.last_update_timestamp = 0
        self.physics_dt = 1 / 120
        gen = torch.Generator().manual_seed(37)

        def random(*shape):
            return torch.randn(shape, generator=gen, dtype=dtype)

        self.actions = random(count, 6).clamp(-1, 1)
        self.fixed_pos_obs_frame = random(count, 3) * 0.01
        self.init_fixed_pos_obs_noise = random(count, 3) * 0.001
        self.fingertip_midpoint_pos = random(count, 3) * 0.02
        self.fingertip_midpoint_quat = random(count, 4)
        self.success_latched = torch.zeros(count, dtype=torch.bool)
        self.success_hold_pos = random(count, 3)
        self.success_hold_quat = random(count, 4)
        self.current_success = torch.arange(count) % 2 == 0
        self.evaluation_interval_peak_force = torch.arange(count, dtype=dtype) * 1000
        self.calls = []

    def _compute_intermediate_values(self, dt):
        self.calls.append(("intermediate", dt))
        self.last_update_timestamp = self._robot._data._sim_timestamp
        self.fingertip_midpoint_pos += 0.00001

    def _sample_evaluation_geometry(self):
        self.calls.append("geometry")

    def _get_curr_successes(self, threshold):
        self.calls.append(("success", threshold))
        return self.current_success

    def _spiral_target(self, frame):
        self.calls.append("spiral")
        return frame + 0.015, -self.fingertip_midpoint_quat

    def generate_ctrl_signals(self, target_pos, target_quat, grip):
        self.calls.append(("control", grip))
        self.command_pos = target_pos.clone()
        self.command_quat = target_quat.clone()
        self.applied_wrench = torch.cat((
            (target_pos - self.fingertip_midpoint_pos) * 10000,
            torch.zeros_like(target_pos),
        ), dim=-1)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TargetUpdateTests(unittest.TestCase):
    def assertBitsEqual(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))

    def test_hold_selection_matches_masked_reference(self):
        for count in (1, 4, 128):
            for dtype in (torch.float32, torch.float64):
                for success_mode in ("none", "partial", "all"):
                    for latched_mode in ("none", "partial", "all"):
                        for enabled in (False, True):
                            with self.subTest(count=count, dtype=dtype, success=success_mode,
                                              latched=latched_mode, enabled=enabled):
                                def mask(mode):
                                    return (torch.zeros(count, dtype=torch.bool) if mode == "none"
                                            else torch.ones(count, dtype=torch.bool) if mode == "all"
                                            else torch.arange(count) % 2 == 0)

                                env = FakeEnv(count, dtype=dtype)
                                success, latch = mask(success_mode), mask(latched_mode)
                                reference_latch = latch.clone()
                                pos = env.success_hold_pos.clone()
                                quat = env.success_hold_quat.clone()
                                reference_pos, reference_quat = pos.clone(), quat.clone()
                                new = success & ~reference_latch & enabled
                                if torch.any(new):
                                    reference_pos[new] = env.fingertip_midpoint_pos[new]
                                    reference_pos[new, 2] -= env.cfg.hold.downward_margin_m
                                    reference_quat[new] = env.fingertip_midpoint_quat[new]
                                reference_latch |= new
                                pos_pointer, quat_pointer = pos.data_ptr(), quat.data_ptr()
                                update_success_hold(success, latch, pos, quat,
                                                    env.fingertip_midpoint_pos,
                                                    env.fingertip_midpoint_quat, enabled,
                                                    make_constants(env).downward_margin)
                                self.assertBitsEqual(latch, reference_latch)
                                self.assertBitsEqual(pos, reference_pos)
                                self.assertBitsEqual(quat, reference_quat)
                                self.assertEqual(pos.data_ptr(), pos_pointer)
                                self.assertEqual(quat.data_ptr(), quat_pointer)

    def test_hold_updates_do_not_change_existing_latches_or_signed_zero(self):
        env = FakeEnv(4)
        env.fingertip_midpoint_pos[:, :2] = torch.tensor([-0.0, 0.0])
        env.success_latched[:] = torch.tensor([False, True, False, True])
        old_pos = env.success_hold_pos.clone()
        old_quat = env.success_hold_quat.clone()
        update_success_hold(torch.ones(4, dtype=torch.bool), env.success_latched,
                            env.success_hold_pos, env.success_hold_quat,
                            env.fingertip_midpoint_pos, env.fingertip_midpoint_quat,
                            True, make_constants(env).downward_margin)
        self.assertBitsEqual(env.success_hold_pos[[1, 3]], old_pos[[1, 3]])
        self.assertBitsEqual(env.success_hold_quat[[1, 3]], old_quat[[1, 3]])
        self.assertBitsEqual(env.success_hold_pos[[0, 2], :2], env.fingertip_midpoint_pos[[0, 2], :2])

    def test_target_selection_bounds_and_quaternions_match(self):
        for count in (1, 4, 128):
            for dtype in (torch.float32, torch.float64):
                for mode in ("none", "partial", "all"):
                    with self.subTest(count=count, dtype=dtype, mode=mode):
                        env = FakeEnv(count, dtype=dtype)
                        latch = torch.arange(count) % 2 == 0
                        if mode != "partial":
                            latch[:] = mode == "all"
                        target_pos = env.fingertip_midpoint_pos * 20
                        target_quat = -env.fingertip_midpoint_quat
                        frame = env.fixed_pos_obs_frame + env.init_fixed_pos_obs_noise
                        constants = make_constants(env)
                        expected_pos, expected_quat = target_pos.clone(), target_quat.clone()
                        if torch.any(latch):
                            expected_pos[latch] = env.success_hold_pos[latch]
                            expected_quat[latch] = env.success_hold_quat[latch]
                        expected_pos = frame + torch.clamp(expected_pos - frame,
                                                          -constants.bounds, constants.bounds)
                        expected_pos = env.fingertip_midpoint_pos + torch.clamp(
                            expected_pos - env.fingertip_midpoint_pos,
                            -constants.max_error, constants.max_error)
                        actual_pos, actual_quat = select_and_bound_targets(
                            target_pos, target_quat, latch, env.success_hold_pos,
                            env.success_hold_quat, frame, env.fingertip_midpoint_pos,
                            constants.bounds, constants.max_error)
                        self.assertBitsEqual(actual_pos, expected_pos)
                        self.assertBitsEqual(actual_quat, expected_quat)

    def test_absolute_bounds_are_applied_before_impedance_error(self):
        env = FakeEnv(1)
        env.success_latched[:] = False
        frame = torch.zeros((1, 3))
        current = torch.tensor([[1.0, 0.0, 0.0]])
        actual, _ = select_and_bound_targets(
            torch.tensor([[2.0, 0.0, 0.0]]), env.fingertip_midpoint_quat,
            env.success_latched, env.success_hold_pos, env.success_hold_quat,
            frame, current, torch.full((3,), 0.1), torch.full((3,), 0.2))
        self.assertBitsEqual(actual, torch.tensor([[0.8, 0.0, 0.0]]))

    def test_complete_method_matches_project_oracle_and_call_order(self):
        FakeEnv._apply_action = original_apply_action()
        for count in (1, 4, 128):
            for mode in ("residual", "zero_residual", "spiral"):
                for enabled in (False, True):
                    for stale in (False, True):
                        with self.subTest(count=count, mode=mode, enabled=enabled, stale=stale):
                            reference = FakeEnv(count, mode, enabled)
                            reference.success_latched[1::3] = True
                            if not stale:
                                reference.last_update_timestamp = 1
                            optimized = copy.deepcopy(reference)
                            restore = install(optimized)
                            try:
                                # A second action must keep previously latched poses intact.
                                for _ in range(2):
                                    reference._apply_action()
                                    optimized._apply_action()
                                    for attribute in ("success_latched", "success_hold_pos",
                                                      "success_hold_quat", "command_pos",
                                                      "command_quat", "applied_wrench",
                                                      "evaluation_interval_peak_force"):
                                        self.assertBitsEqual(getattr(optimized, attribute),
                                                             getattr(reference, attribute))
                                    self.assertEqual(optimized.calls, reference.calls)
                            finally:
                                restore()

    def test_constant_cache_dtype_matches_original_constructors(self):
        original_dtype = torch.get_default_dtype()
        try:
            for dtype in (torch.float32, torch.float64):
                torch.set_default_dtype(dtype)
                env = FakeEnv(dtype=dtype)
                constants = make_constants(env)
                self.assertEqual(constants.bounds.dtype, dtype)
                self.assertEqual(constants.max_error.dtype, dtype)
                self.assertEqual(constants.downward_margin.dtype, dtype)
                self.assertEqual(constants.downward_margin.shape, torch.Size([]))
                self.assertEqual(constants.bounds.device, env.success_hold_pos.device)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_restore_is_idempotent_and_preserves_class_and_instance_overrides(self):
        FakeEnv._apply_action = original_apply_action()
        env = FakeEnv()
        original = env._apply_action
        restore = install(env)
        self.assertIsNot(env._apply_action.__func__, original.__func__)
        restore()
        restore()
        self.assertNotIn("_apply_action", vars(env))
        self.assertIs(env._apply_action.__func__, original.__func__)
        overridden = MethodType(original_apply_action(), env)
        env._apply_action = overridden
        restore = install(env)
        restore()
        self.assertIs(env._apply_action, overridden)


if __name__ == "__main__":
    unittest.main()
