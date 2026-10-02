"""Evaluate spiral, zero-residual or PPO control against frozen insertion cases.

The reference evaluator uses one environment to make per-case reset provenance
independent of batching. It stops before Isaac Lab's auto-reset boundary.
Run with the Python interpreter from the Isaac Lab 2.2.1 environment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Metrics and manifest commands remain usable without importing Isaac Sim.
sys.path.insert(0, str(PROJECT_ROOT / "local_insertion"))
from evaluation_cases import read_case_set  # noqa: E402
from evaluation_loop import ControlSample, run_case  # noqa: E402
from evaluation_metrics import EvaluationCriteria, summarize_trials  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["spiral", "zero_residual", "ppo"], required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--agent-config", type=Path, help="Original RL-Games YAML saved by the training run.")
    parser.add_argument("--horizon-s", type=float, default=10.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    for name in ("horizon_s", "hold_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.hold_s > args.horizon_s:
        parser.error("--hold-s cannot exceed --horizon-s")
    if args.output.exists():
        parser.error("--output must not already exist")
    if args.method == "ppo":
        if args.checkpoint is None or args.agent_config is None:
            parser.error("PPO requires --checkpoint and the training run's --agent-config")
        for path in (args.checkpoint, args.agent_config):
            if not path.is_file():
                parser.error(f"Missing file: {path}")
    elif args.checkpoint is not None or args.agent_config is not None:
        parser.error("Checkpoint and agent configuration apply only to PPO")
    return args


def validate_cuda_device(device):
    """Fail before simulator startup when a requested CUDA device is unusable."""
    if not device.startswith("cuda"):
        # CPU and any other device handling remain the simulator's responsibility.
        return
    match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
    if match is None:
        raise ValueError(f"Invalid CUDA device {device!r}; use 'cuda' or 'cuda:N' with a nonnegative index")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA preflight requires PyTorch; use the configured Isaac Lab Python environment") from exc

    access_hint = (
        "Check nvidia-smi and GPU access inside this container. "
        "If access was lost after startup, restart the Pod/container."
    )
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
    except (RuntimeError, AssertionError, OSError) as exc:
        raise RuntimeError(f"CUDA preflight could not query GPU access for {device!r}. {access_hint}") from exc
    if not available or count == 0:
        raise RuntimeError(f"No accessible CUDA GPU for requested device {device!r}. {access_hint}")

    requested_index = int(match[1]) if match[1] is not None else None
    if requested_index is not None and requested_index >= count:
        raise ValueError(
            f"Invalid CUDA device index {requested_index}: {count} device(s) are visible; "
            f"valid indices are 0 through {count - 1}"
        )
    try:
        index = requested_index if requested_index is not None else torch.cuda.current_device()
        # Availability queries can succeed without initializing CUDA. Exercise
        # the requested device before paying for simulator/asset initialization.
        torch.empty(1, device=f"cuda:{index}")
    except (RuntimeError, AssertionError, OSError) as exc:
        raise RuntimeError(
            f"CUDA device {device!r} was reported available but could not allocate a tensor. {access_hint}"
        ) from exc


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_fingerprint():
    """Include uncommitted source content and work without a .git directory."""
    paths = []
    for directory in ("local_insertion", "scripts"):
        paths.extend((PROJECT_ROOT / directory).rglob("*.py"))
        paths.extend((PROJECT_ROOT / directory).rglob("*.yaml"))
    return json_sha256({str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in sorted(paths)})


def serializable_config(value):
    if isinstance(value, dict):
        return {str(k): serializable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable_config(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return str(value)


def write_json(path, value):
    # Only used for files inside this run's newly created directory.
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def runtime_versions(base):
    from isaaclab.envs import DirectRLEnv
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
    import torch

    result = {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda}
    for package in ("isaaclab", "isaacsim", "rl-games", "gymnasium"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "distribution-metadata-unavailable"
    result["factory_source_sha256"] = file_sha256(inspect.getfile(FactoryEnv))
    result["direct_env_source_sha256"] = file_sha256(inspect.getfile(DirectRLEnv))
    result["device"] = str(base.device)
    if str(base.device).startswith("cuda"):
        result["gpu"] = torch.cuda.get_device_name(base.device)
    return result


def make_player(env, checkpoint, agent_config, device):
    """Follow the pinned Isaac Lab RL-Games play runner, including normalizers."""
    import yaml
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    config = yaml.safe_load(agent_config.read_text())
    if not isinstance(config, dict) or "params" not in config:
        raise ValueError("Expected the original RL-Games agent YAML with a params section")
    parameters = config["params"]
    if not parameters["config"].get("ppo", False):
        raise ValueError("This evaluation entry point expects a PPO checkpoint")
    clip_obs = float(parameters.get("env", {}).get("clip_observations", math.inf))
    clip_actions = float(parameters.get("env", {}).get("clip_actions", math.inf))
    if math.isnan(clip_obs) or clip_obs <= 0.0:
        raise ValueError("Invalid observation clipping in agent configuration")
    if clip_actions != 1.0:
        raise ValueError("The frozen benchmark requires the trained action clipping to equal 1.0")
    parameters["load_checkpoint"] = True
    parameters["load_path"] = str(checkpoint.resolve())
    parameters["config"]["device"] = device
    parameters["config"]["device_name"] = device
    parameters["config"]["num_actors"] = 1
    parameters["config"]["env_name"] = "rlgpu"
    wrapped = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    runner = Runner()
    runner.load(config)
    player = runner.create_player()
    player.restore(str(checkpoint.resolve()))
    player.model.eval()
    return player, clip_obs, config


class SimulationAdapter:
    """Convert tensors at the simulator boundary; the episode loop is pure Python."""

    def __init__(self, env, simulation_app, player=None, clip_obs=math.inf):
        self.env = env
        self.base = env.unwrapped
        self.simulation_app = simulation_app
        self.player = player
        self.clip_obs = clip_obs

    def initial_state(self, obs):
        base = self.base
        tensors = {
            "robot_joint_pos": base._robot.data.joint_pos,
            "robot_joint_vel": base._robot.data.joint_vel,
            "robot_root_state": base._robot.data.root_state_w,
            "held_root_state": base._held_asset.data.root_state_w,
            "fixed_root_state": base._fixed_asset.data.root_state_w,
            "episode_friction": base.episode_friction,
            "fixed_pos_obs_noise": base.init_fixed_pos_obs_noise,
            "fixed_pos_obs_frame": base.fixed_pos_obs_frame,
            "tool_position": base.fingertip_midpoint_pos,
            "tool_quaternion": base.fingertip_midpoint_quat,
            "tool_linear_velocity": base.fingertip_midpoint_linvel,
            "tool_angular_velocity": base.fingertip_midpoint_angvel,
            "policy_observation": obs["policy"],
            "proportional_gains": base.task_prop_gains,
            "derivative_gains": base.task_deriv_gains,
            "actions": base.actions,
            "held_material_properties": base._held_asset.root_physx_view.get_material_properties(),
            "fixed_material_properties": base._fixed_asset.root_physx_view.get_material_properties(),
        }
        import torch
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite initial state: {name}")
        return {name: tensor[0].detach().cpu().tolist() for name, tensor in tensors.items()}

    def reset(self, seed):
        if self.player is not None:
            self.player.reset()
        obs, _ = self.env.reset(seed=seed)
        if int(self.base.episode_length_buf[0].item()) != 0:
            raise RuntimeError("Reset did not clear the episode counter")
        if self.player is not None:
            self.player.get_batch_size(obs["policy"], 1)
            if self.player.is_rnn:
                self.player.init_rnn()
        return obs, self.initial_state(obs)

    def act(self, obs):
        import torch
        if not torch.isfinite(obs["policy"]).all():
            raise RuntimeError("Non-finite policy observation")
        if self.player is None:
            return torch.zeros((1, 6), device=self.base.device)
        actor_obs = torch.clamp(obs["policy"], -self.clip_obs, self.clip_obs).to(self.player.device)
        with torch.inference_mode():
            action = self.player.get_action(self.player.obs_to_torch(actor_obs), is_deterministic=True)
        if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite policy action")
        if tuple(action.shape) != (1, 6):
            raise RuntimeError(f"Expected batched six-dimensional actions, got {tuple(action.shape)}")
        return torch.clamp(action, -1.0, 1.0).to(self.base.device)

    def step(self, action):
        import torch
        if not self.simulation_app.is_running():
            raise RuntimeError("Simulation closed before the evaluation horizon")
        # Simulator state survives across cases and is mutated by reset().
        # Do not turn its persistent buffers into inference-only tensors.
        with torch.no_grad():
            obs, reward, terminated, truncated, _ = self.env.step(action)
        # Do not refresh kinematics here: Factory._get_dones already did so.
        xy, depth, _ = self.base.insertion_geometry()
        return ControlSample(
            observation=obs,
            xy_error_m=float(xy[0].item()),
            depth_m=float(depth[0].item()),
            reward=float(reward[0].item()),
            task_reward=float(self.base.evaluation_task_reward[0].item()),
            commanded_force_n=float(self.base.evaluation_interval_peak_force[0].item()),
            interval_success=bool(self.base.evaluation_interval_success[0].item()),
            interval_ever_success=bool(self.base.evaluation_interval_ever_success[0].item()),
            terminated=bool(terminated[0].item()),
            truncated=bool(truncated[0].item()),
            episode_step=int(self.base.episode_length_buf[0].item()),
            action=action[0].detach().cpu().tolist(),
        )


def main(argv=None):
    args = parse_args(argv)
    cases = read_case_set(args.cases)
    validate_cuda_device(args.device)
    # AppLauncher must start before importing Isaac Sim task modules.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=args.headless, device=args.device)
    simulation_app = launcher.app
    env = None
    output_created = False
    metadata = {"schema_version": 1, "run_complete": False, "method": args.method}
    started = time.monotonic()
    try:
        import gymnasium as gym
        sys.path.insert(0, str(PROJECT_ROOT))
        import local_insertion  # noqa: F401
        from local_insertion.env_cfg import LocalInsertionRLEnvCfg

        cfg = LocalInsertionRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = cases["cases"][0]["seed"]
        cfg.controller_mode = "residual" if args.method == "ppo" else args.method
        dt = cfg.sim.dt * cfg.decimation
        horizon_steps = round(args.horizon_s / dt)
        if not math.isclose(horizon_steps * dt, args.horizon_s, abs_tol=1e-9, rel_tol=0.0):
            raise ValueError("Evaluation horizon must be a whole number of control steps")
        geometry = cfg.evaluation_geometry
        criteria = EvaluationCriteria(dt, args.hold_s, geometry.xy_tolerance_m, geometry.depth_tolerance_m)
        if horizon_steps < criteria.required_hold_samples:
            raise ValueError("Evaluation horizon is shorter than the hold requirement")
        # Factory ends a trajectory at max_episode_length-1. Guard by two steps
        # and fail if any done/reset is observed during the scored horizon.
        cfg.episode_length_s = (horizon_steps + 2) * dt
        env = gym.make("Isaac-LocalInsertion-RL-Direct-v0", cfg=cfg)
        base = env.unwrapped
        if base.max_episode_length - 1 <= horizon_steps:
            raise RuntimeError("The simulator auto-reset boundary is inside the evaluation horizon")
        player, clip_obs, restored_config = None, math.inf, None
        if args.method == "ppo":
            player, clip_obs, restored_config = make_player(env, args.checkpoint, args.agent_config, args.device)
        adapter = SimulationAdapter(env, simulation_app, player, clip_obs)

        common_cfg = serializable_config(base.cfg.to_dict())
        common_cfg.pop("controller_mode")
        protocol = {
            "name": "local_insertion_held_v1",
            "horizon_steps": horizon_steps,
            "criteria": asdict(criteria),
            "initialization": "single_environment_seeded_reset_with_realized_state_gate",
            "success_sampling": "all_physics_boundaries_including_control_interval_endpoints",
            "force_sampling": "maximum_command_magnitude_over_all_physics_substeps",
            "shared_environment_config": common_cfg,
        }
        metadata.update({
            "case_set": cases,
            "case_set_id": cases["case_set_id"],
            "trial_count": len(cases["cases"]),
            "protocol_id": json_sha256(protocol),
            "protocol": protocol,
            "criteria": asdict(criteria),
            "code_revision": source_fingerprint(),
            "runtime_versions": runtime_versions(base),
            "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
            "agent_config_sha256": file_sha256(args.agent_config) if args.agent_config else None,
            "deterministic_policy": True,
            "policy_observation_clip": clip_obs if math.isfinite(clip_obs) else None,
            "control_uses_simulator_success_signal": True,
            "force_note": "Commanded impedance wrench, not sensor-measured contact force.",
        })
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
            metadata["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        except OSError:
            metadata["git_commit"] = None
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        write_json(args.output / "metadata.json", metadata)
        if restored_config is not None:
            write_json(args.output / "restored_agent_config.json", serializable_config(restored_config))
        rows = []
        with (args.output / "trajectories.jsonl").open("x") as traces, (args.output / "trials.jsonl").open("x") as trials:
            def write_trace(sample):
                traces.write(json.dumps(sample, allow_nan=False) + "\n")

            for case in cases["cases"]:
                row = run_case(
                    case, criteria, float(base.cfg_task.fixed_asset_cfg.height), horizon_steps,
                    adapter.reset, adapter.act, adapter.step, write_trace,
                )
                row["method"] = args.method
                rows.append(row)
                trials.write(json.dumps(row, allow_nan=False) + "\n")
                trials.flush()
                traces.flush()
                print(f"[{len(rows)}/{len(cases['cases'])}] {case['case_id']}: held_success={row['held_success']}", flush=True)
        write_json(args.output / "trials.json", rows)
        summary = summarize_trials(rows)
        summary["mean_task_episode_reward"] = sum(row["task_episode_reward"] for row in rows) / len(rows)
        summary["method"] = args.method
        write_json(args.output / "summary.json", summary)
        metadata["run_complete"] = True
        metadata["wall_time_s"] = time.monotonic() - started
        write_json(args.output / "metadata.json", metadata)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        if output_created:
            metadata["run_complete"] = False
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            write_json(args.output / "metadata.json", metadata)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
