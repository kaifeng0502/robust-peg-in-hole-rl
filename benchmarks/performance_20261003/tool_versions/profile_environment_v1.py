"""Measure insertion stepping without editing the measured project or Isaac Lab.

Use separate invocations for uninstrumented throughput and diagnostic tracing.
This is a synthetic action workload, not a PPO training speed measurement.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import uuid

from performance_probe import Probe


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--mode", choices=("throughput", "diagnostic"), default="throughput")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-mode", choices=("zero", "replay"), default="replay")
    parser.add_argument("--reward-profile", choices=("legacy", "terminal_hold_v2"), default="legacy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace-last-steps", type=int, default=0)
    parser.add_argument("--barrier", type=Path, help="Fresh shared directory for concurrent workers")
    parser.add_argument("--barrier-token", help="Shared fresh UUID for this concurrent measurement")
    parser.add_argument("--participants", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new directory")
    if not (args.project / "local_insertion" / "envs.py").is_file():
        parser.error("--project must contain the insertion environment")
    if args.num_envs < 1 or args.steps < 450 or args.steps % 150:
        parser.error("Use a positive environment count and at least 450 steps, in multiples of 150")
    if args.warmup_steps < 150 or args.warmup_steps % 150:
        parser.error("Warm-up must include complete 150-step episodes")
    if not 0 <= args.trace_last_steps <= args.steps:
        parser.error("Invalid trace window")
    if args.trace_last_steps and args.mode != "diagnostic":
        parser.error("Tracing is diagnostic only; throughput runs must be uninstrumented")
    if args.participants < 1 or not 0 <= args.worker_id < args.participants:
        parser.error("Invalid barrier participant count or worker ID")
    if (args.participants > 1) != (args.barrier is not None):
        parser.error("A shared barrier is required only for concurrent workers")
    if args.barrier is not None:
        try:
            uuid.UUID(args.barrier_token or "")
        except ValueError:
            parser.error("Concurrent workers require the same fresh --barrier-token UUID")
    elif args.barrier_token is not None:
        parser.error("--barrier-token requires --barrier")
    return args


def wait_for_peers(args):
    """Coordinate after warm-up, outside the measurement window."""
    if args.barrier is None:
        return
    deadline = time.monotonic() + 600.0
    manifest = args.barrier / "manifest.json"
    expected_manifest = {"token": args.barrier_token, "participants": args.participants}
    if args.worker_id == 0:
        args.barrier.mkdir(parents=True, exist_ok=False)
        temporary = args.barrier / "manifest.tmp"
        write_json(temporary, expected_manifest)
        temporary.rename(manifest)
    while not manifest.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("Benchmark leader did not create a fresh barrier")
        time.sleep(0.1)
    if json.loads(manifest.read_text()) != expected_manifest:
        raise ValueError("Barrier belongs to a different benchmark run")
    go = args.barrier / "go.json"
    if go.exists():
        raise ValueError("Refusing a barrier with an existing start signal")
    ready = args.barrier / f"worker_{args.worker_id}.ready"
    with ready.open("x") as stream:
        stream.write(str(args.output.resolve()))
    expected = [args.barrier / f"worker_{i}.ready" for i in range(args.participants)]
    while not all(path.is_file() for path in expected):
        if time.monotonic() > deadline:
            raise TimeoutError("Other performance workers did not become ready")
        time.sleep(0.1)
    if args.worker_id == 0:
        temporary = args.barrier / "go.tmp"
        with temporary.open("x") as stream:
            json.dump({"start_at_monotonic_s": time.monotonic() + 3.0}, stream)
        temporary.rename(go)
    while not go.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("No benchmark start signal")
        time.sleep(0.05)
    start_at = json.loads(go.read_text())["start_at_monotonic_s"]
    while time.monotonic() < start_at:
        time.sleep(min(0.01, max(0.0, start_at - time.monotonic())))


def attach_regions(base, probe, factory_control):
    methods = {
        "_pre_physics_step": "action_preparation",
        "_apply_action": "apply_action",
        "_compute_intermediate_values": "kinematics",
        "insertion_geometry": "insertion_geometry",
        "generate_ctrl_signals": "control_signals",
        "_get_dones": "done_checks",
        "_get_rewards": "reward",
        "_get_observations": "observations",
        "_reset_idx": "reset",
        "_sample_and_apply_friction": "friction_reset",
        "randomize_initial_state": "randomized_reset",
        "set_pos_inverse_kinematics": "reset_ik",
        "close_gripper_in_place": "gripper_close",
        "step_sim_no_action": "reset_sim_step",
    }
    for method, label in methods.items():
        probe.wrap(base, method, label)
    probe.wrap(base.scene, "write_data_to_sim", "scene_write")
    probe.wrap(base.scene, "update", "scene_update")
    probe.wrap(base.sim, "step", "sim_step")
    probe.wrap(base.sim, "forward", "sim_forward")
    probe.wrap(factory_control, "compute_dof_torque", "control_torque")


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(args.project.resolve()))
    sys.path.insert(0, str(args.project.resolve() / "scripts"))
    from evaluate_insertion import (
        file_sha256, runtime_versions, serializable_config, source_fingerprint, validate_cuda_device,
    )

    validate_cuda_device(args.device)
    from isaaclab.app import AppLauncher

    args.output.mkdir(parents=True, exist_ok=False)
    app = None
    env = None
    probe = None
    profiler_stack = ExitStack()
    try:
        app = AppLauncher(headless=True, device=args.device).app
        import gymnasium as gym
        import torch
        import local_insertion  # noqa: F401
        from local_insertion.env_cfg import LocalInsertionRLEnvCfg
        from isaaclab_tasks.direct.factory import factory_control

        code_revision = source_fingerprint()
        cfg = LocalInsertionRLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.episode_length_s = 10.066666666666666
        cfg.reward.success_contract = args.reward_profile
        assert cfg.sim.dt == 1.0 / 120.0 and cfg.decimation == 8
        env = gym.make("Isaac-LocalInsertion-RL-Direct-v0", cfg=cfg)
        base = env.unwrapped
        assert base.max_episode_length - 1 == 150
        observation, _ = env.reset(seed=args.seed)
        # A separate CPU generator leaves the environment's randomization RNG
        # untouched. Reuse the exact same action stream for every comparison.
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 100000)
        actions_cpu = torch.zeros((args.warmup_steps + args.steps, args.num_envs, 6))
        if args.action_mode == "replay":
            actions_cpu.uniform_(-0.2, 0.2, generator=generator)
        action_sha = hashlib.sha256(actions_cpu.numpy().tobytes()).hexdigest()
        actions = actions_cpu.to(base.device)

        def synchronize():
            if str(base.device).startswith("cuda"):
                torch.cuda.synchronize(base.device)

        write_json(args.output / "metadata.json", {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "code_revision": code_revision,
            "runtime_versions": runtime_versions(base),
            "environment_config": serializable_config(cfg.to_dict()),
            "action_sha256": action_sha,
            "measurement_script_sha256": file_sha256(__file__),
            "probe_script_sha256": file_sha256(Path(__file__).with_name("performance_probe.py")),
            "factory_control_sha256": file_sha256(inspect.getfile(factory_control)),
            "torch_execution_settings": {
                "cpu_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "note": "Synthetic stepping workload, not policy evaluation or PPO training.",
        })
        print("Warming up complete episodes", flush=True)
        with torch.no_grad():
            for step in range(args.warmup_steps):
                observation, reward, _, _, _ = env.step(actions[step])
        synchronize()

        if args.mode == "diagnostic":
            probe = Probe(region_factory=torch.profiler.record_function if args.trace_last_steps else None)
            attach_regions(base, probe, factory_control)
        measured_done = torch.zeros((), dtype=torch.int64, device=base.device)
        finite = torch.ones((), dtype=torch.bool, device=base.device)
        wait_for_peers(args)
        synchronize()
        start_wall = time.time()
        start_monotonic = time.monotonic()
        start = time.perf_counter()
        trace = None
        # No scalar tensor reads, per-step logging, or explicit synchronization
        # occur inside this loop. GPU reductions are identical across variants.
        with torch.no_grad():
            for step in range(args.steps):
                if args.trace_last_steps and step == args.steps - args.trace_last_steps:
                    trace = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False, with_stack=False, profile_memory=False,
                    )
                    profiler_stack.enter_context(trace)
                context = torch.profiler.record_function("measured_env_step") if trace is not None else nullcontext()
                with context:
                    observation, reward, terminated, truncated, _ = env.step(actions[args.warmup_steps + step])
                measured_done += (terminated | truncated).sum()
                finite &= torch.isfinite(observation["policy"]).all() & torch.isfinite(reward).all()
                if trace is not None:
                    trace.step()
        synchronize()
        duration = time.perf_counter() - start
        end_monotonic = time.monotonic()
        end_wall = time.time()
        if trace is not None:
            profiler_stack.close()
            trace.export_chrome_trace(str(args.output / "trace.json"))
            (args.output / "operator_cpu_times.txt").write_text(
                trace.key_averages().table(sort_by="self_cpu_time_total", row_limit=60)
            )
            (args.output / "operator_device_times.txt").write_text(
                trace.key_averages().table(sort_by="self_device_time_total", row_limit=60)
            )
        assert bool(finite.item()), "Non-finite observation or reward"
        assert int(measured_done.item()) == args.num_envs * (args.steps // 150)
        assert source_fingerprint() == code_revision, "Measured source changed during the run"
        result = {
            "passed": True,
            "mode": args.mode,
            "wall_time_s": duration,
            "measurement_start_unix_s": start_wall,
            "measurement_end_unix_s": end_wall,
            "measurement_start_monotonic_s": start_monotonic,
            "measurement_end_monotonic_s": end_monotonic,
            "vector_steps": args.steps,
            "environment_transitions": args.steps * args.num_envs,
            "environment_transitions_per_s": args.steps * args.num_envs / duration,
            "mean_vector_step_ms": 1000.0 * duration / args.steps,
            "policy_physics_substeps": args.steps * cfg.decimation,
            "completed_episodes": int(measured_done.item()),
            "includes_reset_physics": True,
            "code_revision": code_revision,
            "action_sha256": action_sha,
            "cpu_regions": probe.summary() if probe else None,
            "interpretation": (
                "No-profiler synchronized-boundary wall time, including reset and fixed lightweight "
                "GPU validity checks; no PPO updates."
                if args.mode == "throughput" else
                "Diagnostic run includes instrumentation overhead; CPU region times include enqueue/wait time, "
                "not isolated GPU duration. Nested inclusive times must not be added. "
                "Trace records visible PyTorch CUDA work; PhysX kernels may require a system profiler."
            ),
        }
        write_json(args.output / "result.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "cpu_regions"}), flush=True)
    finally:
        try:
            profiler_stack.close()
        finally:
            try:
                if probe is not None:
                    probe.restore()
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    if app is not None:
                        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
