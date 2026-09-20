"""Configuration for the fixed-search baseline and randomized local RL task."""

from __future__ import annotations

import math

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.factory.factory_env_cfg import CtrlCfg, FactoryTaskPegInsertCfg, ObsRandCfg
from isaaclab_tasks.direct.factory.factory_tasks_cfg import PegInsert


@configclass
class PoseFrictionRandomizationCfg:
    """Episode-level uncertainty shared by baseline evaluation and RL training."""

    pose_enabled: bool = True
    friction_enabled: bool = True
    friction_min: float = 0.30
    friction_max: float = 1.20


@configclass
class SpiralControllerCfg:
    approach_duration_s: float = 1.0
    preload_m: float = 0.0030
    radial_speed_m_s: float = 0.0010
    turns_per_second: float = 1.25
    max_radius_m: float = 0.0045
    engage_depth_m: float = 0.0020
    insertion_depth_m: float = 0.0245
    max_xy_error_m: float = 0.0040
    max_z_error_m: float = 0.0030


@configclass
class LocalRewardCfg:
    keypoint_scale: float = 1.0
    alignment_scale: float = 0.5
    approach_scale: float = 1.5
    progress_scale: float = 3.0
    depth_scale: float = 2.0
    engage_scale: float = 2.0
    first_success_scale: float = 20.0
    success_hold_scale: float = 1.0
    action_scale: float = 0.01
    action_rate_scale: float = 0.02
    commanded_wrench_scale: float = 0.002
    alignment_sigma_m: float = 0.003
    approach_sigma_m: float = 0.010
    progress_normalizer_m: float = 0.002
    wrench_normalizer_n: float = 20.0


def _make_task() -> PegInsert:
    task = PegInsert()
    # The actual hole stays within the same robot workspace. Relative peg-hole
    # errors come from the hand pose, grasp, and observation uncertainty below.
    task.fixed_asset_init_pos_noise = [0.005, 0.005, 0.001]
    task.fixed_asset_init_orn_deg = 0.0
    task.fixed_asset_init_orn_range_deg = 10.0
    task.hand_init_pos_noise = [0.004, 0.004, 0.002]
    task.hand_init_orn_noise = [math.radians(3.0), math.radians(3.0), math.radians(5.0)]
    task.held_asset_pos_noise = [0.001, 0.001, 0.0005]
    task.action_penalty_ee_scale = 0.0
    task.action_grad_penalty_scale = 0.0
    return task


def _make_rl_task() -> PegInsert:
    """Start PPO at the local pre-insertion pose supplied by the global planner."""
    task = _make_task()
    # Factory's 47 mm hand offset leaves the peg base about 15 mm above the
    # socket.  The RL policy is the contact-stage controller, so place it about
    # 3 mm above the entrance instead of asking it to learn the free-space move.
    task.hand_init_pos = [0.0, 0.0, 0.035]
    task.hand_init_pos_noise = [0.004, 0.004, 0.001]
    return task


def _make_ctrl() -> CtrlCfg:
    ctrl = CtrlCfg()
    ctrl.ema_factor = 0.35
    # Factory applies one action across eight 120 Hz substeps. These values
    # therefore produce at most 2 mm and 2 degrees per 15 Hz policy step.
    ctrl.pos_action_threshold = [0.00025, 0.00025, 0.00025]
    ctrl.rot_action_threshold = [math.radians(0.25)] * 3
    ctrl.pos_action_bounds = [0.020, 0.020, 0.060]
    ctrl.rot_action_bounds = [math.radians(15.0)] * 3
    # Softer Z gain limits contact load while X/Y remain accurate enough to align.
    ctrl.default_task_prop_gains = [500.0, 500.0, 400.0, 30.0, 30.0, 30.0]
    return ctrl


def _make_obs_rand() -> ObsRandCfg:
    obs_rand = ObsRandCfg()
    obs_rand.fixed_asset_pos = [0.002, 0.002, 0.001]
    return obs_rand


@configclass
class SpiralBaselineEnvCfg(FactoryTaskPegInsertCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, clone_in_fabric=True)
    task: PegInsert = _make_task()
    ctrl: CtrlCfg = _make_ctrl()
    obs_rand: ObsRandCfg = _make_obs_rand()
    randomization: PoseFrictionRandomizationCfg = PoseFrictionRandomizationCfg()
    spiral: SpiralControllerCfg = SpiralControllerCfg()
    episode_length_s: float = 8.0
@configclass
class LocalInsertionRLEnvCfg(FactoryTaskPegInsertCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=True)
    task: PegInsert = _make_rl_task()
    ctrl: CtrlCfg = _make_ctrl()
    obs_rand: ObsRandCfg = _make_obs_rand()
    randomization: PoseFrictionRandomizationCfg = PoseFrictionRandomizationCfg()
    reward: LocalRewardCfg = LocalRewardCfg()
    # PPO outputs an absolute residual around the planner's nominal insertion
    # target.  The controller still clips instantaneous impedance error.
    residual_xy_span_m: float = 0.006
    residual_z_span_m: float = 0.003
    residual_roll_pitch_span_rad: float = math.radians(5.0)
    residual_yaw_span_rad: float = math.radians(10.0)
    nominal_tool_to_peg_base_m: float = 0.032
    nominal_insertion_depth_m: float = 0.0245
    episode_length_s: float = 10.0
