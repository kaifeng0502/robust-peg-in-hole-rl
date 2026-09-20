"""Short dynamic validation of task registration, reset, randomization, and stepping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=["baseline", "rl"], required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--action-scale", type=float, default=0.2)
parser.add_argument("--disable-pose-randomization", action="store_true")
parser.add_argument("--disable-friction-randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import local_insertion  # noqa: E402, F401
from local_insertion.env_cfg import LocalInsertionRLEnvCfg, SpiralBaselineEnvCfg  # noqa: E402


def main() -> int:
    if args.task == "baseline":
        task_id = "Isaac-LocalInsertion-Spiral-Direct-v0"
        cfg = SpiralBaselineEnvCfg()
    else:
        task_id = "Isaac-LocalInsertion-RL-Direct-v0"
        cfg = LocalInsertionRLEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.randomization.pose_enabled = not args.disable_pose_randomization
    cfg.randomization.friction_enabled = not args.disable_friction_randomization

    print(f"[smoke] creating {task_id} with {args.num_envs} environments", flush=True)
    env = gym.make(task_id, cfg=cfg)
    print("[smoke] environment created; resetting", flush=True)
    observation, _ = env.reset()
    print("[smoke] reset complete; stepping", flush=True)
    base = env.unwrapped
    actions = torch.zeros((base.num_envs, 6), device=base.device)
    rewards = []
    _, initial_depth, _ = base.insertion_geometry()
    max_depth = initial_depth.clone()
    for step in range(args.steps):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed during smoke test")
        if args.task == "rl":
            actions.uniform_(-args.action_scale, args.action_scale)
        with torch.inference_mode():
            observation, reward, _, _, _ = env.step(actions)
        if not torch.isfinite(observation["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError("Non-finite observation or reward")
        rewards.append(float(reward.mean().item()))
        _, depth, _ = base.insertion_geometry()
        max_depth = torch.maximum(max_depth, depth)
        if (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"[smoke] completed step {step + 1}/{args.steps}", flush=True)

    friction = base.episode_friction
    low = cfg.randomization.friction_min
    high = cfg.randomization.friction_max
    passed = bool(torch.all((friction >= low) & (friction <= high)))
    result = {
        "passed": passed,
        "task": task_id,
        "num_envs": base.num_envs,
        "steps": args.steps,
        "policy_observation_shape": list(observation["policy"].shape),
        "mean_reward": sum(rewards) / len(rewards),
        "initial_mean_depth_m": float(initial_depth.mean().item()),
        "final_mean_depth_m": float(depth.mean().item()),
        "max_depth_m": float(max_depth.max().item()),
        "success_count": int(base._get_curr_successes(base.cfg_task.success_threshold).sum().item()),
        "friction_min_observed": float(friction.min().item()),
        "friction_max_observed": float(friction.max().item()),
        "friction_expected_range": [low, high],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close(wait_for_replicator=False)
