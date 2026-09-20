"""Evaluate impedance control plus fixed spiral search on randomized peg insertion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trials", type=int, default=128, help="Total randomized trials.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", action="store_true", help="Record the first trial batch.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.trials <= 0 or args.num_envs <= 0:
    parser.error("--trials and --num_envs must be positive")
if args.output.exists() and any(args.output.iterdir()):
    parser.error("--output must be empty")
if args.video:
    args.enable_cameras = True
    args.num_envs = 1

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaacsim.core.utils.torch as torch_utils  # noqa: E402

import local_insertion  # noqa: F401, E402
from local_insertion.env_cfg import SpiralBaselineEnvCfg  # noqa: E402


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    cfg = SpiralBaselineEnvCfg()
    cfg.scene.num_envs = min(args.num_envs, args.trials)
    cfg.scene.clone_in_fabric = not args.video
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.seed = args.seed
    if args.video:
        cfg.viewer.eye = (0.95, 0.35, 0.30)
        cfg.viewer.lookat = (0.60, 0.0, 0.085)
        cfg.viewer.resolution = (960, 720)

    env = gym.make(
        "Isaac-LocalInsertion-Spiral-Direct-v0",
        cfg=cfg,
        render_mode="rgb_array" if args.video else None,
    )
    base = env.unwrapped
    env.reset()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    frames = []
    completed = 0
    batch_index = 0
    actions = torch.zeros((base.num_envs, 6), device=base.device)

    while completed < args.trials:
        base._compute_intermediate_values(base.physics_dt)
        initial_xy, initial_depth, _ = base.insertion_geometry()
        start_friction = base.episode_friction.clone()
        observation_error_xy = torch.linalg.vector_norm(base.init_fixed_pos_obs_noise[:, :2], dim=1)
        roll, pitch, _ = torch_utils.get_euler_xyz(base.fingertip_midpoint_quat)
        roll_error = torch.atan2(torch.sin(roll - torch.pi), torch.cos(roll - torch.pi))
        pitch_error = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        initial_tilt = torch.sqrt(torch.square(roll_error) + torch.square(pitch_error))
        done = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)

        for _ in range(base.max_episode_length + 1):
            if not simulation_app.is_running():
                raise RuntimeError("Simulation closed before evaluation completed")
            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(actions)
            if args.video and batch_index == 0:
                frames.append(env.render())
            done = torch.logical_or(terminated, truncated)
            if bool(torch.all(done)):
                break
        else:
            raise RuntimeError("Baseline episode exceeded configured horizon")

        take = min(base.num_envs, args.trials - completed)
        for env_index in range(take):
            succeeded = bool(base.ep_succeeded[env_index].item())
            success_step = int(base.ep_success_times[env_index].item())
            rows.append(
                {
                    "trial": completed + env_index,
                    "seed": args.seed,
                    "success": int(succeeded),
                    "success_time_s": success_step * base.step_dt if succeeded else "",
                    "initial_xy_error_m": float(initial_xy[env_index].item()),
                    "initial_depth_m": float(initial_depth[env_index].item()),
                    "hole_observation_xy_error_m": float(observation_error_xy[env_index].item()),
                    "initial_tool_tilt_rad": float(initial_tilt[env_index].item()),
                    "friction": float(start_friction[env_index].item()),
                    "max_depth_m": float(base.completed_max_depth[env_index].item()),
                    "peak_commanded_force_n": float(
                        base.completed_peak_commanded_force[env_index].item()
                    ),
                }
            )
        completed += take
        batch_index += 1

    fieldnames = list(rows[0])
    with (args.output / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successes = [row for row in rows if row["success"]]
    summary = {
        "controller": "Cartesian impedance + fixed Archimedean spiral search",
        "task": "Isaac Lab Factory 8 mm peg insertion",
        "trials": len(rows),
        "seed": args.seed,
        "difficulty_profile": "challenge_v2",
        "success_rate": len(successes) / len(rows),
        "mean_success_time_s": _mean_or_none([float(row["success_time_s"]) for row in successes]),
        "mean_peak_commanded_force_n": _mean_or_none(
            [float(row["peak_commanded_force_n"]) for row in rows]
        ),
        "mean_max_depth_m": _mean_or_none([float(row["max_depth_m"]) for row in rows]),
        "randomization": {
            "target_position_m": cfg.task.fixed_asset_init_pos_noise,
            "hand_position_m": cfg.task.hand_init_pos_noise,
            "hand_orientation_rad": cfg.task.hand_init_orn_noise,
            "grasp_position_m": cfg.task.held_asset_pos_noise,
            "hole_observation_std_m": cfg.obs_rand.fixed_asset_pos,
            "friction": [cfg.randomization.friction_min, cfg.randomization.friction_max],
        },
        "force_note": "Commanded impedance wrench, not a force/torque sensor measurement.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.video and frames:
        import imageio.v2 as imageio

        imageio.mimsave(args.output / "spiral_baseline.mp4", frames, fps=round(1.0 / base.step_dt))

    print(json.dumps(summary, indent=2), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close(wait_for_replicator=False)
