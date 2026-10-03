"""Microbenchmark _apply_action with two cached constants on a frozen real state.

PhysX does not advance during measurement. This cannot establish full-trajectory
equivalence, environment-step throughput, or PPO training acceleration.
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


INPUT_NAMES = (
    "joint_pos", "joint_vel", "fingertip_midpoint_pos", "fingertip_midpoint_quat",
    "fingertip_midpoint_linvel", "fingertip_midpoint_angvel", "fingertip_midpoint_jacobian",
    "arm_mass_matrix", "task_prop_gains", "task_deriv_gains", "dead_zone_thresholds",
    "actions", "held_pos", "held_quat", "fixed_pos", "fixed_quat", "fixed_pos_obs_frame",
    "init_fixed_pos_obs_noise", "episode_length_buf",
)
OUTPUT_NAMES = (
    "joint_torque", "applied_wrench", "ctrl_target_joint_pos", "success_latched",
    "success_hold_pos", "success_hold_quat", "evaluation_interval_success",
    "evaluation_interval_ever_success", "evaluation_interval_peak_force", "evaluation_task_reward",
    "training_hold_samples", "training_ever_geometry_success", "training_ever_held_success",
)


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


def snapshot(base, names, torch):
    return {name: value.detach().clone() for name in names
            if isinstance(value := getattr(base, name, None), torch.Tensor)}


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(args.project.resolve()))
    sys.path.insert(0, str(args.project.resolve() / "scripts"))
    from evaluate_insertion import file_sha256, runtime_versions, serializable_config, source_fingerprint, validate_cuda_device
    from target_constants import install, validate_cache

    validate_cuda_device(args.device)
    from isaaclab.app import AppLauncher

    args.output.mkdir(parents=True, exist_ok=False)
    app = None
    env = None
    restore = None
    try:
        app = AppLauncher(headless=True, device=args.device).app
        import gymnasium as gym
        import torch
        import local_insertion  # noqa: F401
        from local_insertion.env_cfg import LocalInsertionRLEnvCfg
        from isaaclab_tasks.direct.factory import factory_control

        code_revision = source_fingerprint()
        factory_sha = file_sha256(inspect.getfile(factory_control))
        patch_path = Path(__file__).with_name("target_constants.py")
        patch_sha = file_sha256(patch_path)
        cfg = LocalInsertionRLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.episode_length_s = 10.066666666666666
        cfg.reward.success_contract = "legacy"
        assert cfg.sim.dt == 1 / 120 and cfg.decimation == 8
        assert cfg.controller_mode == "residual", "Frozen-state timing excludes stateful spiral search"
        env = gym.make("Isaac-LocalInsertion-RL-Direct-v0", cfg=cfg)
        base = env.unwrapped
        assert base.max_episode_length - 1 == 150
        env.reset(seed=args.seed)
        device = torch.device(base.device)
        default_dtype = torch.get_default_dtype()
        thread_settings = (torch.get_num_threads(), torch.get_num_interop_threads())
        config_snapshot = serializable_config(cfg.to_dict())
        properties = torch.cuda.get_device_properties(device)
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "code_revision": code_revision, "runtime_versions": runtime_versions(base),
            "environment_config": config_snapshot, "factory_control_sha256": factory_sha,
            "script_sha256": file_sha256(__file__), "patch_sha256": patch_sha,
            "gpu": {"name": properties.name, "total_memory_bytes": properties.total_memory,
                    "multiprocessor_count": properties.multi_processor_count},
            "torch_execution_settings": {
                "cpu_threads": thread_settings[0], "interop_threads": thread_settings[1],
                "default_dtype": str(default_dtype),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "scope": "Frozen real state; _apply_action only, no PhysX advancement during measurement.",
        }
        write_json(args.output / "metadata.json", metadata)

        def require_equal(name, expected, actual):
            if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.device != actual.device:
                raise AssertionError(f"Tensor metadata changed: {name}")
            if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()):
                raise AssertionError(f"Nonfinite value: {name}")
            if not torch.equal(expected, actual) or not torch.equal(
                expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
            ):
                raise AssertionError(f"Tensor bits changed: {name}")

        with torch.no_grad():
            zero_action = torch.zeros((args.num_envs, 6), device=device)
            for _ in range(args.simulation_warmup_steps):
                env.step(zero_action)
            original = base._apply_action
            # Resolve stale kinematics and latch/peak reductions to their fixed
            # point before comparing methods on this one unchanged scene.
            for _ in range(args.warmup_calls):
                original()
            torch.cuda.synchronize(device)
            inputs = snapshot(base, INPUT_NAMES, torch)
            expected_outputs = snapshot(base, OUTPUT_NAMES, torch)
            fixed_timestamp = base._robot._data._sim_timestamp
            fixed_update_timestamp = base.last_update_timestamp
            restore = install(base)
            candidate = base._apply_action
            validate_cache(base)

            def require_stable_state():
                for name, expected in inputs.items():
                    require_equal("input/" + name, expected, getattr(base, name))
                for name, expected in expected_outputs.items():
                    require_equal("output/" + name, expected, getattr(base, name))
                if (base._robot._data._sim_timestamp != fixed_timestamp
                        or base.last_update_timestamp != fixed_update_timestamp):
                    raise AssertionError("Simulation or kinematics timestamp changed")
                if torch.device(base.device) != device or torch.get_default_dtype() != default_dtype:
                    raise AssertionError("Device/default dtype changed")
                if serializable_config(cfg.to_dict()) != config_snapshot:
                    raise AssertionError("Configuration changed")
                if (torch.get_num_threads(), torch.get_num_interop_threads()) != thread_settings:
                    raise AssertionError("Thread settings changed")
                validate_cache(base)

            # The temporary hook only observes actual command arguments and
            # delegates untouched. It is removed before either timed variant.
            captured = {}
            capture_calls = 0
            bound_control = base.generate_ctrl_signals
            absent = object()
            previous = vars(base).get("generate_ctrl_signals", absent)

            def capture(self, ctrl_target_fingertip_midpoint_pos,
                        ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos):
                nonlocal captured, capture_calls
                capture_calls += 1
                captured = {"target_pos": ctrl_target_fingertip_midpoint_pos.detach().clone(),
                            "target_quat": ctrl_target_fingertip_midpoint_quat.detach().clone(),
                            "gripper_target": ctrl_target_gripper_dof_pos}
                return bound_control(ctrl_target_fingertip_midpoint_pos,
                                     ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos)

            base.generate_ctrl_signals = MethodType(capture, base)
            expected_targets = None
            try:
                for function in (original, candidate, original, candidate):
                    capture_calls = 0
                    function()
                    if capture_calls != 1:
                        raise AssertionError("Unexpected control call count during preflight")
                    if expected_targets is None:
                        expected_targets = captured
                    for name in ("target_pos", "target_quat"):
                        require_equal("command/" + name, expected_targets[name], captured[name])
                    if expected_targets["gripper_target"] != captured["gripper_target"]:
                        raise AssertionError("Gripper command changed")
                    require_stable_state()
            finally:
                if previous is absent:
                    delattr(base, "generate_ctrl_signals")
                else:
                    base.generate_ctrl_signals = previous
            metadata["frozen_inputs"] = {
                name: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
                for name, value in inputs.items()
            }
            metadata["compared_output_fields"] = sorted(expected_outputs)
            metadata["frozen_simulation_timestamp"] = float(fixed_timestamp)
            metadata["held_environment_count"] = int(base.success_latched.sum().item())
            write_json(args.output / "metadata.json", metadata)
            for _ in range(args.warmup_calls):
                original()
                candidate()
            torch.cuda.synchronize(device)
            require_stable_state()
            functions = {"A": original, "B": candidate}
            stream = torch.cuda.current_stream(device)
            blocks = []
            for index, label in enumerate("ABBAABBA"):
                function = functions[label]
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                start_event.record(stream)
                for _ in range(args.calls_per_block):
                    function()
                end_event.record(stream)
                torch.cuda.synchronize(device)
                duration = time.perf_counter() - start
                blocks.append({"index": index, "variant": "reference" if label == "A" else "target_constants",
                               "calls": args.calls_per_block, "wall_s": duration,
                               "wall_ms_per_call": duration * 1000 / args.calls_per_block,
                               "cuda_stream_elapsed_ms_per_call": start_event.elapsed_time(end_event) / args.calls_per_block})
                require_stable_state()

        assert source_fingerprint() == code_revision, "Project source changed"
        assert file_sha256(inspect.getfile(factory_control)) == factory_sha, "Controller source changed"
        assert file_sha256(patch_path) == patch_sha, "Candidate patch changed"
        summaries = {}
        for variant in ("reference", "target_constants"):
            values = [block["wall_ms_per_call"] for block in blocks if block["variant"] == variant]
            summaries[variant] = {"median_wall_ms_per_call": statistics.median(values),
                                  "min_wall_ms_per_call": min(values), "max_wall_ms_per_call": max(values)}
        reference_ms = summaries["reference"]["median_wall_ms_per_call"]
        candidate_ms = summaries["target_constants"]["median_wall_ms_per_call"]
        result = {
            "passed": True, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "block_order": "ABBAABBA", "labels": {"A": "reference", "B": "target_constants"},
            "blocks": blocks, "summary": summaries, "median_apply_action_speedup": reference_ms / candidate_ms,
            "median_apply_action_wall_reduction_percent": (1 - candidate_ms / reference_ms) * 100,
            "checks": {"preflight_commands_bitwise_equal": True, "output_buffers_bitwise_equal": True,
                       "frozen_inputs_unchanged": True, "finite_inputs_outputs": True,
                       "configuration_device_dtype_unchanged": True, "no_simulation_advancement": True},
            "interpretation": "_apply_action microbenchmark at one frozen real state, including unchanged control "
                              "and geometry branches. CUDA stream elapsed time includes idle gaps. "
                              "No PhysX advancement, full-trajectory equivalence, env.step or PPO speedup claim.",
        }
        write_json(args.output / "result.json", result)
        print(json.dumps({"passed": True, "summary": summaries,
                          "apply_action_speedup": reference_ms / candidate_ms,
                          "scope": "_apply_action only; frozen real state; no PhysX advancement"}), flush=True)
    except Exception as error:
        write_json(args.output / "result.json", {"passed": False, "error_type": type(error).__name__,
                                                  "error": str(error)})
        raise
    finally:
        try:
            if restore is not None:
                restore()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if app is not None:
                    app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
