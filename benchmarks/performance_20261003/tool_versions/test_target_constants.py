"""Pinned-module source guards and CPU-only numerical oracle tests.

No Isaac imports or module-level statements are executed. Numerical checks
explicitly skip without PyTorch; passing these tests is not GPU validation.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

from target_constants import (
    BOUNDS_EXPRESSION, BOUNDS_NAME, ERROR_EXPRESSION, ERROR_NAME, EXPECTED_SOURCE_SHA256,
    _method_code, _replace_expressions, build_candidate, install, validate_cache,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


def source_path():
    configured = os.environ.get("INSERTION_PROJECT")
    if configured:
        return Path(configured) / "local_insertion/envs.py"
    parent = Path(__file__).resolve().parents[2]
    for root in (parent, parent / "insertion_contact_check/baseline_rl"):
        path = root / "local_insertion/envs.py"
        if path.is_file():
            return path
    raise FileNotFoundError("Set INSERTION_PROJECT to the measured project checkout")


def original_method(torch_module):
    path = source_path()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SOURCE_SHA256
    code = _method_code(raw.decode(), str(path))
    namespace = {"__builtins__": builtins.__dict__, "__name__": "target_constant_fixture",
                 "__file__": str(path), "torch": torch_module, "math": math,
                 "torch_utils": SimpleNamespace(quat_from_euler_xyz=quat_from_euler_xyz)}
    return FunctionType(code, namespace, code.co_name)


class ConstructorSpy:
    """Only supports install-time constructors; no numerical operation is faked."""

    def __init__(self):
        self.calls = []
        self.dtype = "float32"

    def device(self, device):
        return str(device)

    def get_default_dtype(self):
        return self.dtype

    def tensor(self, data, *, device):
        result = SimpleNamespace(values=tuple(data), device=device, dtype=self.dtype)
        self.calls.append(result)
        return result


def bare_environment(torch_module):
    cls = type("FixtureEnv", (), {"_apply_action": original_method(torch_module)})
    env = cls()
    env.device = "cpu"
    env.cfg = SimpleNamespace(ctrl=SimpleNamespace(pos_action_bounds=[0.02, 0.02, 0.06]),
                              hold=SimpleNamespace(max_xy_error_m=0.004, max_z_error_m=0.003))
    return env, cls


class SourceGuardTests(unittest.TestCase):
    def test_whole_module_replacement_changes_only_two_expressions(self):
        source = source_path().read_text()
        actual = ast.parse(_replace_expressions(source))
        expected = ast.parse(source)
        matches = 0

        class ExpectedChange(ast.NodeTransformer):
            def visit_Assign(self, node):
                nonlocal matches
                name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                if name in ("bounds", "max_error") and isinstance(node.value, ast.Call):
                    expression = ast.unparse(node.value)
                    original = BOUNDS_EXPRESSION if name == "bounds" else ERROR_EXPRESSION
                    if expression == ast.unparse(ast.parse(original, mode="eval").body):
                        node.value = ast.Name(id=BOUNDS_NAME if name == "bounds" else ERROR_NAME,
                                              ctx=ast.Load())
                        matches += 1
                return node

        expected = ExpectedChange().visit(expected)
        self.assertEqual(matches, 2)
        self.assertEqual(ast.dump(actual), ast.dump(expected))
        self.assertEqual(len(source.splitlines()), len(_replace_expressions(source).splitlines()))
        for invalid in ("", source + "\n" + BOUNDS_EXPRESSION, source.replace(ERROR_EXPRESSION, "None")):
            with self.assertRaises(ValueError):
                _replace_expressions(invalid)

    def test_compilation_does_not_execute_module(self):
        source = source_path().read_text().replace(
            "from __future__ import annotations",
            "from __future__ import annotations\n"
            "if __cache_compile_must_not_execute__:\n"
            "    raise AssertionError('must not execute')",
        )
        self.assertEqual(_method_code(source, "fixture.py").co_name, "_apply_action")

    def test_hash_and_runtime_guards(self):
        env, _ = bare_environment(ConstructorSpy())
        with patch("target_constants.Path.read_bytes", return_value=b"altered"):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                build_candidate(env)
        original = env._apply_action.__func__
        code = original.__code__
        original.__code__ = code.replace(co_firstlineno=code.co_firstlineno + 1)
        with self.assertRaises(TypeError) as raised:
            build_candidate(env)
        difference = json.loads(str(raised.exception).split(": ", 1)[1])
        self.assertIn("co_firstlineno", difference)
        self.assertNotIn("_apply_action", vars(env))

    def test_install_preserves_globals_builtins_and_other_instances(self):
        backend = ConstructorSpy()
        env, cls = bare_environment(backend)
        untouched = cls()
        original = env._apply_action.__func__
        original_globals = dict(original.__globals__)
        restore = install(env)
        try:
            replacement = env._apply_action.__func__
            self.assertIn(BOUNDS_NAME, replacement.__code__.co_names)
            self.assertIn(ERROR_NAME, replacement.__code__.co_names)
            self.assertIsNot(replacement.__globals__, original.__globals__)
            for name, value in original_globals.items():
                self.assertIs(replacement.__globals__[name], value)
            self.assertIs(replacement.__builtins__, original.__builtins__)
            self.assertEqual(original.__globals__, original_globals)
            self.assertIs(untouched._apply_action.__func__, original)
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(backend.calls[0].values, (0.02, 0.02, 0.06))
            self.assertEqual(backend.calls[1].values, (0.004, 0.004, 0.003))
            validate_cache(env)
            with self.assertRaises(TypeError):
                install(env)
        finally:
            restore()
            restore()
        self.assertNotIn("_apply_action", vars(env))
        self.assertIs(env._apply_action.__func__, original)
        with self.assertRaises(TypeError):
            validate_cache(env)

    def test_existing_instance_override_is_restored(self):
        env, _ = bare_environment(ConstructorSpy())
        old = MethodType(env._apply_action.__func__, env)
        env._apply_action = old
        restore = install(env)
        restore()
        self.assertIs(env._apply_action, old)

    def test_invalid_bounds_are_rejected_without_installation(self):
        for invalid in ([], [1, 2], [1, 2, 3, 4], [True, 1, 1], [0, 1, 1],
                        [-1, 1, 1], [float("nan"), 1, 1], [float("inf"), 1, 1]):
            with self.subTest(invalid=invalid):
                env, _ = bare_environment(ConstructorSpy())
                env.cfg.ctrl.pos_action_bounds = invalid
                with self.assertRaises(ValueError):
                    install(env)
                self.assertNotIn("_apply_action", vars(env))

    def test_phase_boundary_validation_requires_reinstall_after_change(self):
        for change in ("bounds", "scalar_type", "error", "device", "dtype"):
            with self.subTest(change=change):
                backend = ConstructorSpy()
                env, _ = bare_environment(backend)
                if change == "scalar_type":
                    env.cfg.ctrl.pos_action_bounds = [1, 1, 1]
                restore = install(env)
                if change == "bounds":
                    env.cfg.ctrl.pos_action_bounds[0] = 0.03
                elif change == "scalar_type":
                    env.cfg.ctrl.pos_action_bounds = [1.0, 1.0, 1.0]
                elif change == "error":
                    env.cfg.hold.max_z_error_m = 0.002
                elif change == "device":
                    env.device = "cuda:0"
                else:
                    backend.dtype = "float64"
                with self.assertRaisesRegex(RuntimeError, "restore and reinstall"):
                    validate_cache(env)
                restore()
                restore_new = install(env)
                validate_cache(env)
                restore_new()


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((cr * cp * cy + sr * sp * sy,
                        sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy), dim=-1)


class FakeEnv:
    def __init__(self, count, dtype, mode="residual", enabled=True):
        self.device = "cpu"
        self.num_envs = count
        self.cfg = SimpleNamespace(
            hold=SimpleNamespace(enabled=enabled, downward_margin_m=0.0003,
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
        generator = torch.Generator().manual_seed(37)

        def random(*shape):
            return torch.randn(shape, generator=generator, dtype=dtype)

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
        self.applied_wrench = torch.cat(((target_pos - self.fingertip_midpoint_pos) * 10000,
                                        torch.zeros_like(target_pos)), dim=-1)


@contextmanager
def default_dtype(dtype):
    original = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original)


@unittest.skipIf(torch is None, "PyTorch absent; run CPU numerical checks before GPU benchmarking")
class NumericalOracleTests(unittest.TestCase):
    def setUp(self):
        FakeEnv._apply_action = original_method(torch)

    def assert_bits_equal(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def assert_state_equal(self, candidate, reference):
        for name in ("success_latched", "success_hold_pos", "success_hold_quat", "command_pos",
                     "command_quat", "applied_wrench", "evaluation_interval_peak_force"):
            self.assert_bits_equal(getattr(candidate, name), getattr(reference, name))
        self.assertEqual(candidate.calls, reference.calls)

    def test_modes_latches_dtypes_and_repeated_steps_match_original_bitwise(self):
        for dtype in (torch.float32, torch.float64):
            with default_dtype(dtype):
                for count in (1, 4, 128):
                    for mode in ("residual", "zero_residual", "spiral"):
                        for enabled in (False, True):
                            for mask_kind in ("none", "mixed", "all"):
                                for stale in (False, True):
                                    with self.subTest(dtype=dtype, count=count, mode=mode,
                                                      hold=enabled, mask=mask_kind, stale=stale):
                                        reference = FakeEnv(count, dtype, mode, enabled)
                                        reference.success_latched[:] = (
                                            torch.arange(count) % 3 == 1 if mask_kind == "mixed"
                                            else mask_kind == "all")
                                        reference.current_success[:] = (
                                            torch.arange(count) % 2 == 0 if mask_kind == "mixed"
                                            else mask_kind == "all")
                                        reference.last_update_timestamp = 0 if stale else 1
                                        candidate = copy.deepcopy(reference)
                                        restore = install(candidate)
                                        try:
                                            for _ in range(2):
                                                reference._apply_action()
                                                candidate._apply_action()
                                                self.assert_state_equal(candidate, reference)
                                        finally:
                                            restore()

    def test_constructor_dtype_is_inferred_independently_of_state_dtype(self):
        for dtype in (torch.float32, torch.float64):
            with default_dtype(dtype):
                env = FakeEnv(4, torch.float64 if dtype == torch.float32 else torch.float32)
                restore = install(env)
                try:
                    namespace = env._apply_action.__func__.__globals__
                    for key, value in ((BOUNDS_NAME, env.cfg.ctrl.pos_action_bounds),
                                       (ERROR_NAME, [0.004, 0.004, 0.006])):
                        self.assert_bits_equal(namespace[key], torch.tensor(value, device=env.device))
                        self.assertEqual(namespace[key].dtype, dtype)
                finally:
                    restore()
                env.cfg.ctrl.pos_action_bounds = [1, 1, 1]
                restore = install(env)
                self.assertEqual(env._apply_action.__func__.__globals__[BOUNDS_NAME].dtype, torch.int64)
                restore()

    def test_hot_path_has_zero_constructors_and_preserves_signed_zero_and_clamp_order(self):
        reference = FakeEnv(4, torch.float32, enabled=True)
        reference.last_update_timestamp = 1
        reference.fingertip_midpoint_pos[:] = torch.tensor([-0.0, 0.0, 0.0])
        reference.actions.zero_()
        reference.fixed_pos_obs_frame[:] = torch.tensor([1.0, -1.0, 0.0])
        reference.init_fixed_pos_obs_noise.zero_()
        reference.current_success[:] = True
        candidate = copy.deepcopy(reference)
        restore = install(candidate)
        try:
            constructor = torch.tensor
            with patch.object(torch, "tensor", wraps=constructor) as calls:
                reference._apply_action()
                self.assertEqual(calls.call_count, 2)
            with patch.object(torch, "tensor", wraps=constructor) as calls:
                candidate._apply_action()
                self.assertEqual(calls.call_count, 0)
            self.assert_state_equal(candidate, reference)
        finally:
            restore()

    def test_partial_state_reset_does_not_modify_cached_constants(self):
        reference = FakeEnv(4, torch.float32)
        candidate = copy.deepcopy(reference)
        restore = install(candidate)
        namespace = candidate._apply_action.__func__.__globals__
        cached = {key: namespace[key].clone() for key in (BOUNDS_NAME, ERROR_NAME)}
        try:
            for env in (reference, candidate):
                env._apply_action()
                # Exercise the dynamic subset touched by real reset. This is
                # not a substitute for an Isaac simulator reset validation.
                env.success_latched[::2] = False
                env.success_hold_pos[::2] = 0.0
                env.success_hold_quat[::2] = 0.0
                env.evaluation_interval_peak_force[::2] = 0.0
                env.fingertip_midpoint_pos[::2] += 0.001
                env._apply_action()
            self.assert_state_equal(candidate, reference)
            for key, before in cached.items():
                self.assert_bits_equal(namespace[key], before)
            validate_cache(candidate)
        finally:
            restore()


if __name__ == "__main__":
    unittest.main()
