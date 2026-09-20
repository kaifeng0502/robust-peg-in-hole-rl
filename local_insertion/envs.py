"""Fixed-search baseline and local-contact RL environments for peg insertion."""

from __future__ import annotations

import math

import torch

import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from .env_cfg import LocalInsertionRLEnvCfg, SpiralBaselineEnvCfg


PHASE_APPROACH = 0
PHASE_SEARCH = 1
PHASE_INSERT = 2
PHASE_HOLD = 3


class RandomizedPegInsertEnv(FactoryEnv):
    """Factory peg insertion with episode-level contact-friction randomization."""

    cfg: SpiralBaselineEnvCfg | LocalInsertionRLEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        cfg.sim.render_interval = cfg.decimation
        if not cfg.randomization.pose_enabled:
            cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
            cfg.task.fixed_asset_init_orn_range_deg = 0.0
            cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
            cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
            cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
            cfg.obs_rand.fixed_asset_pos = [0.0, 0.0, 0.0]
        super().__init__(cfg, render_mode, **kwargs)
        self.episode_friction = torch.full(
            (self.num_envs,), self.cfg_task.held_asset_cfg.friction, device=self.device
        )
        self._sample_and_apply_friction(torch.arange(self.num_envs, device=self.device))

    def _sample_and_apply_friction(self, env_ids: torch.Tensor) -> None:
        rand_cfg = self.cfg.randomization
        if rand_cfg.friction_min <= 0.0 or rand_cfg.friction_max < rand_cfg.friction_min:
            raise ValueError("Invalid friction randomization range")
        if rand_cfg.friction_enabled:
            sample = rand_cfg.friction_min + torch.rand(len(env_ids), device=self.device) * (
                rand_cfg.friction_max - rand_cfg.friction_min
            )
        else:
            sample = torch.full(
                (len(env_ids),), self.cfg_task.held_asset_cfg.friction, device=self.device
            )
        self.episode_friction[env_ids] = sample
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        for asset in (self._held_asset, self._fixed_asset):
            materials = asset.root_physx_view.get_material_properties()
            sample_cpu = sample.detach().to(device=materials.device)
            materials[env_ids_cpu, :, 0] = sample_cpu[:, None]
            materials[env_ids_cpu, :, 1] = sample_cpu[:, None]
            asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _reset_idx(self, env_ids):
        if hasattr(self, "episode_friction"):
            self._sample_and_apply_friction(env_ids)
        super()._reset_idx(env_ids)

    def insertion_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return XY error, signed insertion depth, and target peg-base position."""
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos,
            self.held_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        target_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        xy_error = torch.linalg.vector_norm(held_base_pos[:, :2] - target_base_pos[:, :2], dim=1)
        hole_top_z = target_base_pos[:, 2] + self.cfg_task.fixed_asset_cfg.height
        depth = hole_top_z - held_base_pos[:, 2]
        return xy_error, depth, target_base_pos


class SpiralBaselineEnv(RandomizedPegInsertEnv):
    """Cartesian impedance controller with a deterministic spiral search."""

    cfg: SpiralBaselineEnvCfg

    def __init__(self, cfg: SpiralBaselineEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.baseline_phase = torch.full((self.num_envs,), PHASE_APPROACH, dtype=torch.long, device=self.device)
        self.baseline_time = torch.zeros(self.num_envs, device=self.device)
        self.baseline_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.baseline_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.baseline_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.baseline_tool_to_peg = torch.zeros((self.num_envs, 3), device=self.device)
        self.baseline_insert_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.episode_peak_commanded_force = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_depth = torch.full((self.num_envs,), -math.inf, device=self.device)
        self.completed_peak_commanded_force = torch.zeros(self.num_envs, device=self.device)
        self.completed_max_depth = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "baseline_ready"):
            self.baseline_phase[env_ids] = PHASE_APPROACH
            self.baseline_time[env_ids] = 0.0
            self.baseline_ready[env_ids] = False
            self.episode_peak_commanded_force[env_ids] = 0.0
            self.episode_max_depth[env_ids] = -math.inf

    def _initialize_baseline(self, env_ids: torch.Tensor) -> None:
        self.baseline_start[env_ids] = self.fingertip_midpoint_pos[env_ids]
        self.baseline_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]
        self.baseline_tool_to_peg[env_ids] = self.fingertip_midpoint_pos[env_ids] - self.held_pos[env_ids]
        self.baseline_insert_xy[env_ids] = self.fingertip_midpoint_pos[env_ids, :2]
        self.baseline_ready[env_ids] = True

    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        new_ids = torch.logical_not(self.baseline_ready).nonzero(as_tuple=False).squeeze(-1)
        if len(new_ids) > 0:
            self._initialize_baseline(new_ids)

        cfg = self.cfg.spiral
        sensed_hole_top = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        entrance_tool = sensed_hole_top + self.baseline_tool_to_peg
        target = entrance_tool.clone()

        approach = self.baseline_phase == PHASE_APPROACH
        approach_alpha = torch.clamp(self.baseline_time / cfg.approach_duration_s, 0.0, 1.0)
        target[approach] = self.baseline_start[approach] + approach_alpha[approach, None] * (
            entrance_tool[approach] - self.baseline_start[approach]
        )
        self.baseline_phase[torch.logical_and(approach, approach_alpha >= 1.0)] = PHASE_SEARCH

        search = self.baseline_phase == PHASE_SEARCH
        search_time = torch.clamp(self.baseline_time - cfg.approach_duration_s, min=0.0)
        radius = torch.clamp(search_time * cfg.radial_speed_m_s, max=cfg.max_radius_m)
        angle = 2.0 * math.pi * cfg.turns_per_second * search_time
        target[search, 0] += radius[search] * torch.cos(angle[search])
        target[search, 1] += radius[search] * torch.sin(angle[search])
        target[search, 2] -= cfg.preload_m

        _, depth, _ = self.insertion_geometry()
        newly_engaged = torch.logical_and(search, depth > cfg.engage_depth_m)
        if torch.any(newly_engaged):
            self.baseline_phase[newly_engaged] = PHASE_INSERT
            self.baseline_insert_xy[newly_engaged] = self.fingertip_midpoint_pos[newly_engaged, :2]

        inserting = self.baseline_phase == PHASE_INSERT
        target[inserting, :2] = self.baseline_insert_xy[inserting]
        target[inserting, 2] = entrance_tool[inserting, 2] - cfg.insertion_depth_m

        curr_success = self._get_curr_successes(self.cfg_task.success_threshold)
        self.baseline_phase[curr_success] = PHASE_HOLD
        holding = self.baseline_phase == PHASE_HOLD
        target[holding] = self.fingertip_midpoint_pos[holding]

        max_error = torch.tensor(
            [cfg.max_xy_error_m, cfg.max_xy_error_m, cfg.max_z_error_m], device=self.device
        )
        target = self.fingertip_midpoint_pos + torch.clamp(
            target - self.fingertip_midpoint_pos, min=-max_error, max=max_error
        )
        self.generate_ctrl_signals(target, self.baseline_quat, 0.0)
        self.baseline_time += self.physics_dt

    def _get_rewards(self):
        reward = super()._get_rewards()
        force = torch.linalg.vector_norm(self.applied_wrench[:, :3], dim=1)
        _, depth, _ = self.insertion_geometry()
        self.episode_peak_commanded_force = torch.maximum(self.episode_peak_commanded_force, force)
        self.episode_max_depth = torch.maximum(self.episode_max_depth, depth)
        completed = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(completed) > 0:
            self.completed_peak_commanded_force[completed] = self.episode_peak_commanded_force[completed]
            self.completed_max_depth[completed] = self.episode_max_depth[completed]
        return reward


class LocalInsertionRLEnv(RandomizedPegInsertEnv):
    """State-based local insertion task with hidden pose and friction variation."""

    cfg: LocalInsertionRLEnvCfg

    def __init__(self, cfg: LocalInsertionRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.previous_depth = torch.zeros(self.num_envs, device=self.device)
        self.previous_rl_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_hold_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.success_hold_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._compute_intermediate_values(self.physics_dt)
        _, self.previous_depth[:], _ = self.insertion_geometry()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "previous_depth"):
            _, depth, _ = self.insertion_geometry()
            self.previous_depth[env_ids] = depth[env_ids]
            self.previous_rl_action[env_ids] = 0.0
            self.success_latched[env_ids] = False
            self.success_hold_pos[env_ids] = 0.0
            self.success_hold_quat[env_ids] = 0.0

    def _apply_action(self):
        """Interpret actions as bounded residuals around an absolute insertion target."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        fixed_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        current_success = self._get_curr_successes(self.cfg_task.success_threshold)
        newly_success = current_success & ~self.success_latched
        if torch.any(newly_success):
            self.success_hold_pos[newly_success] = self.fingertip_midpoint_pos[newly_success]
            # Apply a small downward hold margin so contact compliance does not
            # let a barely successful peg climb back above the success plane.
            self.success_hold_pos[newly_success, 2] -= 0.0015
            self.success_hold_quat[newly_success] = self.fingertip_midpoint_quat[newly_success]
        self.success_latched |= current_success
        target_pos = fixed_action_frame.clone()
        target_pos[:, 2] += self.cfg.nominal_tool_to_peg_base_m - self.cfg.nominal_insertion_depth_m
        target_pos[:, :2] += self.actions[:, :2] * self.cfg.residual_xy_span_m
        target_pos[:, 2] += self.actions[:, 2] * self.cfg.residual_z_span_m

        bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        target_pos = fixed_action_frame + torch.clamp(target_pos - fixed_action_frame, -bounds, bounds)
        # Bound instantaneous impedance error, especially in Z at contact. The
        # absolute target remains fixed while the robot converges toward it.
        max_error = torch.tensor([0.004, 0.004, 0.003], device=self.device)
        target_pos = self.fingertip_midpoint_pos + torch.clamp(
            target_pos - self.fingertip_midpoint_pos, -max_error, max_error
        )

        target_roll = math.pi + self.actions[:, 3] * self.cfg.residual_roll_pitch_span_rad
        target_pitch = self.actions[:, 4] * self.cfg.residual_roll_pitch_span_rad
        target_yaw = self.actions[:, 5] * self.cfg.residual_yaw_span_rad
        target_quat = torch_utils.quat_from_euler_xyz(target_roll, target_pitch, target_yaw)
        # Once the peg reaches the success geometry, latch the current pose so
        # subsequent policy noise cannot pull it back out before episode end.
        if torch.any(self.success_latched):
            target_pos[self.success_latched] = self.success_hold_pos[self.success_latched]
            target_quat[self.success_latched] = self.success_hold_quat[self.success_latched]
        self.generate_ctrl_signals(target_pos, target_quat, 0.0)

    def _get_rewards(self):
        xy_error, depth, _ = self.insertion_geometry()
        curr_success = self._get_curr_successes(self.cfg_task.success_threshold)
        curr_engaged = self._get_curr_successes(self.cfg_task.engage_threshold)
        first_success = torch.logical_and(curr_success, torch.logical_not(self.ep_succeeded.bool()))
        factory_terms, _ = self._get_factory_rew_dict(curr_success)

        cfg = self.cfg.reward
        alignment = torch.exp(-torch.square(xy_error / cfg.alignment_sigma_m))
        keypoint = (
            factory_terms["kp_baseline"]
            + factory_terms["kp_coarse"]
            + factory_terms["kp_fine"]
        )
        # Dense vertical signal before contact. Multiplication by alignment
        # discourages simply pressing down while the peg is far from the hole.
        approach = alignment * torch.exp(-torch.clamp(-depth, min=0.0) / cfg.approach_sigma_m)
        progress = torch.clamp(
            (depth - self.previous_depth) / cfg.progress_normalizer_m, min=-1.0, max=1.0
        )
        depth_score = torch.clamp(depth / self.cfg_task.fixed_asset_cfg.height, min=0.0, max=1.0)
        action_cost = torch.linalg.vector_norm(self.actions, dim=1)
        action_rate_cost = torch.linalg.vector_norm(self.actions - self.previous_rl_action, dim=1)
        if hasattr(self, "applied_wrench"):
            wrench_cost = torch.linalg.vector_norm(self.applied_wrench[:, :3], dim=1) / cfg.wrench_normalizer_n
        else:
            wrench_cost = torch.zeros_like(depth)

        reward_terms = {
            "keypoint": cfg.keypoint_scale * keypoint,
            "alignment": cfg.alignment_scale * alignment,
            "approach": cfg.approach_scale * approach,
            "progress": cfg.progress_scale * progress,
            "depth": cfg.depth_scale * depth_score,
            "engaged": cfg.engage_scale * curr_engaged.float(),
            "first_success": cfg.first_success_scale * first_success.float(),
            "success_hold": cfg.success_hold_scale * curr_success.float(),
            "action_cost": -cfg.action_scale * action_cost,
            "action_rate_cost": -cfg.action_rate_scale * action_rate_cost,
            "commanded_wrench_cost": -cfg.commanded_wrench_scale * wrench_cost,
        }
        reward = torch.zeros_like(depth)
        for term in reward_terms.values():
            reward += term

        self._log_factory_metrics(reward_terms, curr_success)
        self.previous_depth[:] = depth
        self.previous_rl_action[:] = self.actions
        self.prev_actions = self.actions.clone()
        return reward
