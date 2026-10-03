"""Compare two controllers on a frozen real simulator state, without profiling.

This measures controller calls only: it does not measure environment stepping,
PhysX, reset, policy evaluation, PPO throughput, or learning effectiveness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import time
from types import MethodType


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new directory")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calls-per-block", type=int, default=100)
    parser.add_argument("--warmup-calls", type=int, default=20)
    parser.add_argument("--simulation-warmup-steps", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new directory")
    if not (args.project / "local_insertion/envs.py").is_file():
        parser.error("--project must contain the insertion environment")
    if args.num_envs < 1 or args.calls_per_block < 1 or args.warmup_calls < 1:
        parser.error("Environment and call counts must be positive")
    if not 1 <= args.simulation_warmup_steps < 150:
        parser.error("Simulation warm-up must remain within one 150-step episode")
    if not args.device.startswith("cuda"):
        parser.error("This microbenchmark requires a CUDA device")
    return args


def snapshot_kwargs(base, target_pos, target_quat, torch):
    """Copy exactly the inputs forwarded by the pinned generate_ctrl_signals."""
    values = {
        "cfg": base.cfg,
        "dof_pos": base.joint_pos,
        "dof_vel": base.joint_vel,
        "fingertip_midpoint_pos": base.fingertip_midpoint_pos,
        "fingertip_midpoint_quat": base.fingertip_midpoint_quat,
        "fingertip_midpoint_linvel": base.fingertip_midpoint_linvel,
        "fingertip_midpoint_angvel": base.fingertip_midpoint_angvel,
        "jacobian": base.fingertip_midpoint_jacobian,
        "arm_mass_matrix": base.arm_mass_matrix,
        "ctrl_target_fingertip_midpoint_pos": target_pos,
        "ctrl_target_fingertip_midpoint_quat": target_quat,
        "task_prop_gains": base.task_prop_gains,
        "task_deriv_gains": base.task_deriv_gains,
        "device": base.device,
        "dead_zone_thresholds": base.dead_zone_thresholds,
    }
    return {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in values.items()}


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(args.project.resolve()))
    sys.path.insert(0, str(args.project.resolve() / "scripts"))
    from evaluate_insertion import file_sha256, runtime_versions, serializable_config, source_fingerprint, validate_cuda_device
    from inverse_reuse import build_candidate

    validate_cuda_device(args.device)
    from isaaclab.app import AppLauncher

    args.output.mkdir(parents=True, exist_ok=False)
    app = None
    env = None
    try:
        app = AppLauncher(headless=True, device=args.device).app
        import gymnasium as gym
        import torch
        import local_insertion  # noqa: F401
        from local_insertion.env_cfg import LocalInsertionRLEnvCfg
        from isaaclab_tasks.direct.factory import factory_control

        code_revision = source_fingerprint()
        factory_sha = file_sha256(inspect.getfile(factory_control))
        original = factory_control.compute_dof_torque
        candidate = build_candidate(factory_control)
        cfg = LocalInsertionRLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.episode_length_s = 10.066666666666666
        cfg.reward.success_contract = "legacy"
        assert cfg.sim.dt == 1 / 120 and cfg.decimation == 8
        env = gym.make("Isaac-LocalInsertion-RL-Direct-v0", cfg=cfg)
        base = env.unwrapped
        assert base.max_episode_length - 1 == 150
        env.reset(seed=args.seed)
        config_snapshot = serializable_config(cfg.to_dict())
        device = torch.device(base.device)
        properties = torch.cuda.get_device_properties(device)
        thread_settings = (torch.get_num_threads(), torch.get_num_interop_threads())
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "code_revision": code_revision,
            "runtime_versions": runtime_versions(base),
            "environment_config": config_snapshot,
            "factory_control_sha256": factory_sha,
            "script_sha256": file_sha256(__file__),
            "patch_sha256": file_sha256(Path(__file__).with_name("inverse_reuse.py")),
            "gpu": {"name": properties.name, "total_memory_bytes": properties.total_memory,
                    "multiprocessor_count": properties.multi_processor_count,
                    "compute_capability": [properties.major, properties.minor]},
            "torch_execution_settings": {
                "cpu_threads": thread_settings[0], "interop_threads": thread_settings[1],
                "default_dtype": str(torch.get_default_dtype()),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "scope": "Repeated compute_dof_torque calls on one frozen real state; no env.step speed claim.",
        }
        write_json(args.output / "metadata.json", metadata)

        # Capture the last physical substep's real command and state. The
        # temporary wrapper delegates unchanged and is removed before timing.
        captured = {}
        control_calls = 0
        bound_control = base.generate_ctrl_signals
        absent = object()
        previous = vars(base).get("generate_ctrl_signals", absent)

        def capture(self, ctrl_target_fingertip_midpoint_pos,
                    ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos):
            nonlocal captured, control_calls
            control_calls += 1
            if control_calls == self.cfg.decimation:
                captured = snapshot_kwargs(self, ctrl_target_fingertip_midpoint_pos,
                                           ctrl_target_fingertip_midpoint_quat, torch)
            return bound_control(ctrl_target_fingertip_midpoint_pos,
                                 ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos)

        with torch.no_grad():
            zero_action = torch.zeros((args.num_envs, 6), device=device)
            for _ in range(args.simulation_warmup_steps - 1):
                env.step(zero_action)
            base.generate_ctrl_signals = MethodType(capture, base)
            try:
                env.step(zero_action)
            finally:
                if previous is absent:
                    delattr(base, "generate_ctrl_signals")
                else:
                    base.generate_ctrl_signals = previous
            if control_calls != cfg.decimation or not captured:
                raise RuntimeError("Unexpected control call count during state capture")
            torch.cuda.synchronize(device)
            tensors = {key: value.clone() for key, value in captured.items()
                       if isinstance(value, torch.Tensor)}
            for key, value in tensors.items():
                if not bool(torch.isfinite(value).all()):
                    raise AssertionError(f"Nonfinite frozen input: {key}")
            metadata["frozen_inputs"] = {
                key: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
                for key, value in tensors.items()
            }
            write_json(args.output / "metadata.json", metadata)

            def require_inputs_unchanged():
                for key, value in tensors.items():
                    actual = captured[key]
                    if not torch.equal(actual, value) or not torch.equal(
                        actual.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)
                    ):
                        raise AssertionError(f"Controller changed input: {key}")
                if serializable_config(cfg.to_dict()) != config_snapshot:
                    raise AssertionError("Controller changed configuration")

            def require_same_outputs(expected, actual):
                for name, left, right in zip(("torque", "wrench"), expected, actual, strict=True):
                    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                        raise AssertionError(f"Nonfinite controller output: {name}")
                    if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
                        raise AssertionError(f"Controller output differs: {name}")
                    if not torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)):
                        raise AssertionError(f"Controller output bits differ: {name}")

            expected = original(**captured)
            require_inputs_unchanged()
            require_same_outputs(expected, candidate(**captured))
            require_inputs_unchanged()
            functions = {"A": original, "B": candidate}
            for _ in range(args.warmup_calls):
                original(**captured)
                candidate(**captured)
            torch.cuda.synchronize(device)
            require_inputs_unchanged()
            blocks = []
            stream = torch.cuda.current_stream(device)
            for index, label in enumerate("ABBAABBA"):
                function = functions[label]
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                start_event.record(stream)
                for _ in range(args.calls_per_block):
                    output = function(**captured)
                end_event.record(stream)
                torch.cuda.synchronize(device)
                wall_s = time.perf_counter() - start
                blocks.append({
                    "index": index, "variant": "reference" if label == "A" else "inverse_reuse",
                    "calls": args.calls_per_block, "wall_s": wall_s,
                    "wall_ms_per_call": wall_s * 1000 / args.calls_per_block,
                    "cuda_stream_elapsed_ms_per_call": start_event.elapsed_time(end_event) / args.calls_per_block,
                })
                require_same_outputs(expected, output)
                require_inputs_unchanged()
            require_same_outputs(expected, candidate(**captured))
            require_inputs_unchanged()

        assert source_fingerprint() == code_revision, "Project source changed during measurement"
        assert file_sha256(inspect.getfile(factory_control)) == factory_sha, "Controller source changed"
        assert (torch.get_num_threads(), torch.get_num_interop_threads()) == thread_settings
        summaries = {}
        for variant in ("reference", "inverse_reuse"):
            values = [block["wall_ms_per_call"] for block in blocks if block["variant"] == variant]
            summaries[variant] = {"median_wall_ms_per_call": statistics.median(values),
                                  "min_wall_ms_per_call": min(values), "max_wall_ms_per_call": max(values)}
        reference_ms = summaries["reference"]["median_wall_ms_per_call"]
        candidate_ms = summaries["inverse_reuse"]["median_wall_ms_per_call"]
        result = {
            "passed": True, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "block_order": "ABBAABBA", "labels": {"A": "reference", "B": "inverse_reuse"},
            "blocks": blocks, "summary": summaries,
            "median_controller_speedup": reference_ms / candidate_ms,
            "median_controller_wall_reduction_percent": (1 - candidate_ms / reference_ms) * 100,
            "checks": {"torque_wrench_bitwise_equal": True, "inputs_unchanged": True,
                       "configuration_unchanged": True, "finite_inputs_outputs": True,
                       "thread_settings_unchanged": True},
            "interpretation": "Controller only, one frozen state. CUDA event elapsed time includes stream idle gaps; "
                              "it is not a sum of kernel time. This is not an environment-step or PPO speedup.",
        }
        write_json(args.output / "result.json", result)
        print(json.dumps({"passed": True, "summary": summaries,
                          "controller_speedup": reference_ms / candidate_ms,
                          "scope": "controller only; frozen real state"}), flush=True)
    except Exception as error:
        write_json(args.output / "result.json", {"passed": False, "error_type": type(error).__name__,
                                                  "error": str(error)})
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
