"""Verify the candidate's timeout, terminal metrics and reset in Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_insertion import runtime_versions, source_fingerprint  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--action-contract", choices=["absolute_residual_v1", "relative_z_v1"], default="relative_z_v1")
parser.add_argument("--dense-profile", choices=["legacy", "entry_v1"], default="entry_v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error("--output must be a new file")
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import local_insertion  # noqa: E402, F401
from local_insertion.env_cfg import LocalInsertionRLEnvCfg  # noqa: E402
from local_insertion.evaluation_metrics import EpisodeMetrics, EvaluationCriteria  # noqa: E402


def main():
    code_revision = source_fingerprint()
    cfg = LocalInsertionRLEnvCfg()
    cfg.scene.num_envs = 1024
    count = cfg.scene.num_envs
    cfg.action_contract = args.action_contract
    cfg.reward.dense_profile = args.dense_profile
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.episode_length_s = 10.066666666666666
    cfg.reward.success_contract = "terminal_hold_v2"
    cfg.randomization.pose_enabled = False
    cfg.randomization.friction_enabled = False
    env = gym.make("Isaac-LocalInsertion-RL-Direct-v0", cfg=cfg)
    try:
        base = env.unwrapped
        assert base.max_episode_length - 1 == 150, base.max_episode_length
        criteria = EvaluationCriteria(dt_s=base.step_dt, hold_duration_s=1.0)
        assert base.required_training_hold_samples == criteria.required_hold_samples == 15
        original_rewards = base._get_rewards
        observed = {}

        def capture_rewards():
            reward = original_rewards()
            xy, depth, _ = base.insertion_geometry()
            observed.update({
                "xy": xy.detach().cpu().tolist(),
                "depth": depth.detach().cpu().tolist(),
                "interval": base.evaluation_interval_success.detach().cpu().tolist(),
                "ever_interval": base.evaluation_interval_ever_success.detach().cpu().tolist(),
                "counter": base.training_hold_samples.detach().cpu().tolist(),
                "episode_step": base.episode_length_buf.detach().cpu().tolist(),
                "metrics": {key: float(value) for key, value in base.extras.get("episode", {}).items()},
                "terminal_bonus": float(base.extras["logs_rew_terminal_hold"]),
            })
            return reward

        base._get_rewards = capture_rewards
        obs, _ = env.reset(seed=42)
        actions = torch.zeros((count, 6), device=base.device)
        episodes = []
        for episode in range(2):
            metrics = [EpisodeMetrics(criteria, base.cfg_task.fixed_asset_cfg.height) for _ in range(count)]
            for step in range(1, 151):
                if not simulation_app.is_running():
                    raise RuntimeError("Simulation closed during training contract check")
                if cfg.action_contract == "relative_z_v1":
                    # Explicitly exercise upward/downward authority, then insert.
                    actions[:, 2] = -1.0
                    if step == 1:
                        actions[:count//2, 2] = 1.0
                    if step == 2:
                        actions[:, 2] = 1.0
                    if step == 3:
                        actions[:, 2] = -1.0
                start_z = base.fingertip_midpoint_pos[:, 2].clone()
                with torch.no_grad():
                    obs, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert observed["episode_step"] == [step] * count
                if step == 1 and cfg.action_contract == "relative_z_v1":
                    command_delta = base.relative_z_command - start_z
                    assert torch.all(command_delta[:count//2] > 0)
                    assert torch.all(command_delta[count//2:] < 0)
                    assert torch.all(command_delta.abs() <= cfg.hold.max_z_error_m + 1e-7)
                assert torch.all((terminated | truncated) == (step == 150))
                for index, accumulator in enumerate(metrics):
                    accumulator.update(
                        observed["xy"][index], observed["depth"][index], float(reward[index]), 0.0,
                        interval_success=observed["interval"][index],
                        interval_ever_success=observed["ever_interval"][index],
                    )
                    assert accumulator.result()["held_success"] == (observed["counter"][index] >= 15)
                if step < 150:
                    assert not observed["metrics"], "Terminal metrics leaked into a nonterminal step"
                    assert observed["terminal_bonus"] == 0.0
            rows = [accumulator.result() for accumulator in metrics]
            expected = {
                "terminal_held_success": sum(row["held_success"] for row in rows) / count,
                "ever_geometry_success": sum(row["ever_success"] for row in rows) / count,
                "final_geometry_success": sum(row["final_success"] for row in rows) / count,
                "ever_held_success": sum(row["had_hold_anytime"] for row in rows) / count,
            }
            for key, value in expected.items():
                assert observed["metrics"][f"terminal_hold_v2/{key}"] == value
            assert observed["terminal_bonus"] == cfg.reward.terminal_hold_bonus * expected["terminal_held_success"]
            assert not base.training_hold_samples.any()
            assert not base.training_ever_geometry_success.any()
            assert not base.training_ever_held_success.any()
            episodes.append({
                "episode": episode, "steps": 150, "terminal_metrics": expected,
                "terminal_bonus_mean": observed["terminal_bonus"],
            })
            print(f"training contract episode {episode + 1}/2 passed", flush=True)
        assert any(row["terminal_metrics"]["terminal_held_success"] > 0 for row in episodes), (
            "Nominal check did not exercise a positive held-success/reset path"
        )
        assert source_fingerprint() == code_revision, "Source changed during runtime check"
        result = {
            "passed": True, "num_envs": count, "action_contract": cfg.action_contract, "dense_profile": cfg.reward.dense_profile, "episodes": episodes,
            "max_episode_length": base.max_episode_length,
            "required_hold_samples": 15,
            "code_revision": code_revision,
            "runtime_versions": runtime_versions(base),
            "note": "Nominal runtime contract check, not randomized performance evidence.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close(wait_for_replicator=False)
