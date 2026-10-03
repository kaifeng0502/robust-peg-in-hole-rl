"""Pinned-source and CPU numerical checks; no Isaac Sim or GPU is imported.

Without local PyTorch the numerical tests skip and must run on AutoDL before
this candidate is benchmarked. A passing CPU test is not simulation validation.
"""

from __future__ import annotations

from contextlib import contextmanager
import builtins
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import CodeType, FunctionType, MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

from inverse_reuse import EXPECTED_SOURCE_SHA256, ORIGINAL_EXPRESSION, _replace_expression, build_candidate, install

try:
    import torch
except ModuleNotFoundError:
    torch = None


def source_path():
    checkout = Path(os.environ.get("ISAACLAB_PATH", Path(__file__).resolve().parents[1] / "IsaacLab"))
    return checkout / "source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py"


def factory_fixture():
    """Execute the actual function and gain helper without Isaac-only imports.

    Pose-error inputs use a shared deterministic fixture: pose extraction is
    unchanged by this experiment and is outside these controller-math tests.
    """
    path = source_path()
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == EXPECTED_SOURCE_SHA256
    module_code = compile(source, str(path), "exec", dont_inherit=True)
    codes = [constant for constant in module_code.co_consts
             if isinstance(constant, CodeType)
             and constant.co_name in ("compute_dof_torque", "_apply_task_space_gains")]

    def pose_error(**kwargs):
        return (kwargs["ctrl_target_fingertip_midpoint_pos"] - kwargs["fingertip_midpoint_pos"],
                kwargs["ctrl_target_fingertip_midpoint_quat"][:, 1:]
                - kwargs["fingertip_midpoint_quat"][:, 1:])

    namespace = {"__builtins__": builtins.__dict__, "__name__": "controller_fixture",
                 "torch": torch, "math": math, "get_pose_error": pose_error}
    for code in codes:
        namespace[code.co_name] = FunctionType(code, namespace, code.co_name,
                                             (None,) if code.co_name == "compute_dof_torque" else None)
    return SimpleNamespace(__file__=str(path), compute_dof_torque=namespace["compute_dof_torque"], marker=object())


def environment_fixture(module):
    namespace = {"factory_control": module}
    exec("def generate_ctrl_signals(self, *args, **kwargs):\n"
         "    return factory_control.compute_dof_torque(*args, **kwargs)\n", namespace)
    cls = type("Environment", (), {"generate_ctrl_signals": namespace["generate_ctrl_signals"]})
    return cls(), cls()


class SourceAndInstallationTests(unittest.TestCase):
    def test_replacement_is_exactly_one_expression(self):
        self.assertEqual(_replace_expression(ORIGINAL_EXPRESSION),
                         "jacobian @ arm_mass_matrix_inv @ jacobian_T")
        for source in ("", ORIGINAL_EXPRESSION + "\n" + ORIGINAL_EXPRESSION):
            with self.assertRaises(ValueError):
                _replace_expression(source)

    def test_wrong_source_hash_is_rejected(self):
        module = factory_fixture()
        with patch("inverse_reuse.Path.read_bytes", return_value=b"modified source"):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                build_candidate(module)

    def test_runtime_wrapper_is_rejected(self):
        module = factory_fixture()
        module.compute_dof_torque = lambda: None
        with self.assertRaises(TypeError):
            build_candidate(module)

    def test_runtime_rejection_reports_precise_code_differences(self):
        module = factory_fixture()
        original = module.compute_dof_torque.__code__
        for field, altered in (
            ("co_flags", original.co_flags | 0x1000000),
            ("co_firstlineno", original.co_firstlineno + 1),
            ("co_consts", (*original.co_consts, "altered")),
            ("co_linetable", b""),
        ):
            with self.subTest(field=field):
                module.compute_dof_torque.__code__ = original.replace(**{field: altered})
                with self.assertRaises(TypeError) as error:
                    build_candidate(module)
                differences = json.loads(str(error.exception).split(": ", 1)[1])
                self.assertIn(field, differences)
                self.assertIn("runtime", differences[field])
                self.assertIn("trusted", differences[field])

    def test_install_is_instance_local_and_restore_idempotent(self):
        module = factory_fixture()
        env, untouched = environment_fixture(module)
        original = env.generate_ctrl_signals
        compute = module.compute_dof_torque
        restore = install(env)
        proxy = env.generate_ctrl_signals.__func__.__globals__["factory_control"]
        self.assertIsNot(proxy, module)
        self.assertIs(proxy.marker, module.marker)
        self.assertIsNot(proxy.compute_dof_torque, compute)
        self.assertEqual(inspect.signature(proxy.compute_dof_torque), inspect.signature(compute))
        self.assertIs(module.compute_dof_torque, compute)
        self.assertIs(untouched.generate_ctrl_signals.__func__, original.__func__)
        self.assertIs(original.__func__.__globals__["factory_control"], module)
        with self.assertRaises(TypeError):
            install(env)
        restore()
        restore()
        self.assertNotIn("generate_ctrl_signals", vars(env))
        self.assertEqual(env.generate_ctrl_signals, original)

    def test_existing_instance_override_is_restored(self):
        module = factory_fixture()
        env, _ = environment_fixture(module)
        override = MethodType(env.generate_ctrl_signals.__func__, env)
        env.generate_ctrl_signals = override
        restore = install(env)
        restore()
        self.assertIs(env.generate_ctrl_signals, override)


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def inputs(count, dtype, dead_zone):
    generator = torch.Generator().manual_seed(173)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=dtype)

    factor = random(count, 7, 7)
    mass = factor @ factor.transpose(1, 2) + torch.eye(7, dtype=dtype) * 2
    jacobian = random(count, 6, 7) * 0.2
    jacobian[:, :, :6] += torch.eye(6, dtype=dtype) * 2
    return {
        "cfg": SimpleNamespace(scene=SimpleNamespace(num_envs=count), ctrl=SimpleNamespace(
            default_dof_pos_tensor=[0.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6], kp_null=0.2, kd_null=0.1)),
        "dof_pos": random(count, 9) * 0.1,
        "dof_vel": random(count, 9) * 0.01,
        "fingertip_midpoint_pos": random(count, 3) * 0.01,
        "fingertip_midpoint_quat": random(count, 4) * 0.01,
        "fingertip_midpoint_linvel": random(count, 3) * 0.01,
        "fingertip_midpoint_angvel": random(count, 3) * 0.01,
        "jacobian": jacobian,
        "arm_mass_matrix": mass,
        "ctrl_target_fingertip_midpoint_pos": random(count, 3) * 0.01,
        "ctrl_target_fingertip_midpoint_quat": random(count, 4) * 0.01,
        "task_prop_gains": random(count, 6).abs() + 10,
        "task_deriv_gains": random(count, 6).abs() + 1,
        "device": "cpu",
        "dead_zone_thresholds": torch.full((count, 6), 0.02, dtype=dtype) if dead_zone else None,
    }


@unittest.skipIf(torch is None, "PyTorch is absent locally; run these CPU checks on AutoDL before benchmarking")
class InverseReuseNumericalTests(unittest.TestCase):
    def test_torque_wrench_inputs_and_inverse_counts(self):
        module = factory_fixture()
        candidate = build_candidate(module)
        inverse = torch.inverse
        for count in (1, 8, 128):
            for dtype in (torch.float32, torch.float64):
                for dead_zone in (False, True):
                    with self.subTest(count=count, dtype=dtype, dead_zone=dead_zone), default_dtype(dtype):
                        kwargs = inputs(count, dtype, dead_zone)
                        originals = {key: value.clone() for key, value in kwargs.items()
                                     if isinstance(value, torch.Tensor)}
                        with patch.object(torch, "inverse", wraps=inverse) as calls:
                            expected = module.compute_dof_torque(**kwargs)
                            self.assertEqual(calls.call_count, 3)
                        with patch.object(torch, "inverse", wraps=inverse) as calls:
                            actual = candidate(**kwargs)
                            self.assertEqual(calls.call_count, 2)
                        for reference, result in zip(expected, actual):
                            self.assertEqual(result.dtype, dtype)
                            self.assertEqual(result.shape, reference.shape)
                            self.assertTrue(torch.equal(result.contiguous().view(torch.uint8),
                                                        reference.contiguous().view(torch.uint8)))
                        for key, original in originals.items():
                            self.assertTrue(torch.equal(kwargs[key], original), key)

    def test_singular_matrix_exception_semantics(self):
        module = factory_fixture()
        candidate = build_candidate(module)
        for singular in ("arm_mass_matrix", "jacobian"):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(singular=singular, dtype=dtype), default_dtype(dtype):
                    kwargs = inputs(8, dtype, False)
                    kwargs[singular][3].zero_()
                    errors = []
                    for function in (module.compute_dof_torque, candidate):
                        with self.assertRaises(torch.linalg.LinAlgError) as error:
                            function(**kwargs)
                        errors.append(error.exception)
                    self.assertEqual(type(errors[0]), type(errors[1]))
                    self.assertEqual(str(errors[0]), str(errors[1]))


if __name__ == "__main__":
    unittest.main()
