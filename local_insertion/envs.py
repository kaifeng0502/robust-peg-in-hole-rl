"""Fixed-search baseline and local-contact RL environments for peg insertion."""

from __future__ import annotations

import math

import torch

import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from .env_cfg import LocalInsertionRLEnvCfg, SpiralBaselineEnvCfg
from .evaluation_metrics import EvaluationCriteria
from .reward_contract import advance_terminal_hold


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
        if cfg.controller_mode not in {"residual", "zero_residual", "spiral"}:
            raise ValueError(f"Unknown controller mode: {cfg.controller_mode}")
        if cfg.spiral.approach_duration_s <= 0.0:
            raise ValueError("Spiral approach duration must be positive")
        if min(cfg.hold.max_xy_error_m, cfg.hold.max_z_error_m) <= 0.0:
            raise ValueError("Position-error bounds must be positive")
        if cfg.reward.success_contract not in {"legacy", "terminal_hold_v2"}:
            raise ValueError(f"Unknown reward success contract: {cfg.reward.success_contract}")
        if cfg.reward.success_contract == "terminal_hold_v2":
            self.required_training_hold_samples = EvaluationCriteria(
                dt_s=cfg.sim.dt * cfg.decimation,
                hold_duration_s=cfg.reward.hold_duration_s,
            ).required_hold_samples
            if not math.isfinite(cfg.reward.terminal_hold_bonus) or cfg.reward.terminal_hold_bonus < 0.0:
                raise ValueError("Terminal hold bonus must be finite and nonnegative")
        super().__init__(cfg, render_mode, **kwargs)
        self.previous_depth = torch.zeros(self.num_envs, device=self.device)
        self.previous_rl_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_hold_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.success_hold_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.search_time = torch.zeros(self.num_envs, device=self.device)
        self.search_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.search_insert_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.search_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.evaluation_interval_success = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.evaluation_interval_ever_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.evaluation_interval_peak_force = torch.zeros(self.num_envs, device=self.device)
        self.evaluation_task_reward = torch.zeros(self.num_envs, device=self.device)
        if cfg.reward.success_contract == "terminal_hold_v2":
            self.training_hold_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.training_ever_geometry_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.training_ever_held_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, "held_pos") or self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(self.physics_dt)
        _, self.previous_depth[:], _ = self.insertion_geometry()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Manual benchmark resets must clear the history without relying on an
        # earlier timeout. Factory normally clears these during the next step.
        self._reset_buffers(env_ids)
        if hasattr(self, "previous_depth"):
            _, depth, _ = self.insertion_geometry()
            self.previous_depth[env_ids] = depth[env_ids]
            self.previous_rl_action[env_ids] = 0.0
            self.success_latched[env_ids] = False
            self.success_hold_pos[env_ids] = 0.0
            self.success_hold_quat[env_ids] = 0.0
            self.search_time[env_ids] = 0.0
            self.search_start[env_ids] = self.fingertip_midpoint_pos[env_ids]
            self.search_insert_xy[env_ids] = self.fingertip_midpoint_pos[env_ids, :2]
            self.search_engaged[env_ids] = False
            self.evaluation_interval_success[env_ids] = False
            self.evaluation_interval_ever_success[env_ids] = False
            self.evaluation_interval_peak_force[env_ids] = 0.0
            self.evaluation_task_reward[env_ids] = 0.0
        if hasattr(self, "training_hold_samples"):
            self.training_hold_samples[env_ids] = 0
            self.training_ever_geometry_success[env_ids] = False
            self.training_ever_held_success[env_ids] = False

    def _pre_physics_step(self, action):
        if self.cfg.controller_mode != "residual":
            action = torch.zeros_like(action)
        super()._pre_physics_step(action)
        self.evaluation_interval_success[:] = True
        self.evaluation_interval_ever_success[:] = False
        self.evaluation_interval_peak_force[:] = 0.0

    def evaluation_success_geometry(self):
        """Reject both insufficient depth and excessive penetration below the socket."""
        xy_error, depth, _ = self.insertion_geometry()
        geometry = self.cfg.evaluation_geometry
        return (xy_error <= geometry.xy_tolerance_m) & (
            torch.abs(depth - self.cfg_task.fixed_asset_cfg.height) <= geometry.depth_tolerance_m
        )

    def _sample_evaluation_geometry(self):
        success = self.evaluation_success_geometry()
        self.evaluation_interval_success &= success
        self.evaluation_interval_ever_success |= success

    def _spiral_target(self, fixed_action_frame):
        """Search from the same pre-insertion pose and nominal grasp as PPO."""
        cfg = self.cfg.spiral
        entrance = fixed_action_frame.clone()
        entrance[:, 2] += self.cfg.nominal_tool_to_peg_base_m
        alpha = torch.clamp(self.search_time / cfg.approach_duration_s, 0.0, 1.0)
        target = self.search_start + alpha[:, None] * (entrance - self.search_start)
        searching = self.search_time >= cfg.approach_duration_s
        elapsed = torch.clamp(self.search_time - cfg.approach_duration_s, min=0.0)
        radius = torch.clamp(elapsed * cfg.radial_speed_m_s, max=cfg.max_radius_m)
        angle = 2.0 * math.pi * cfg.turns_per_second * elapsed
        target[searching] = entrance[searching]
        target[searching, 0] += radius[searching] * torch.cos(angle[searching])
        target[searching, 1] += radius[searching] * torch.sin(angle[searching])
        target[searching, 2] -= cfg.preload_m
        _, depth, _ = self.insertion_geometry()
        newly_engaged = searching & (depth > cfg.engage_depth_m) & ~self.search_engaged
        self.search_insert_xy[newly_engaged] = self.fingertip_midpoint_pos[newly_engaged, :2]
        self.search_engaged |= newly_engaged
        target[self.search_engaged, :2] = self.search_insert_xy[self.search_engaged]
        target[self.search_engaged, 2] = (
            entrance[self.search_engaged, 2] - self.cfg.nominal_insertion_depth_m
        )
        roll = torch.full_like(self.search_time, math.pi)
        zero = torch.zeros_like(self.search_time)
        self.search_time += self.physics_dt
        return target, torch_utils.quat_from_euler_xyz(roll, zero, zero)

    def _apply_action(self):
        """Interpret actions as bounded residuals around an absolute insertion target."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        self._sample_evaluation_geometry()
        fixed_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        current_success = self._get_curr_successes(self.cfg_task.success_threshold)
        newly_success = current_success & ~self.success_latched & self.cfg.hold.enabled
        if torch.any(newly_success):
            self.success_hold_pos[newly_success] = self.fingertip_midpoint_pos[newly_success]
            # Apply a small downward hold margin so contact compliance does not
            # let a barely successful peg climb back above the success plane.
            self.success_hold_pos[newly_success, 2] -= self.cfg.hold.downward_margin_m
            self.success_hold_quat[newly_success] = self.fingertip_midpoint_quat[newly_success]
        self.success_latched |= newly_success
        target_pos = fixed_action_frame.clone()
        target_pos[:, 2] += self.cfg.nominal_tool_to_peg_base_m - self.cfg.nominal_insertion_depth_m
        target_pos[:, :2] += self.actions[:, :2] * self.cfg.residual_xy_span_m
        target_pos[:, 2] += self.actions[:, 2] * self.cfg.residual_z_span_m

        target_roll = math.pi + self.actions[:, 3] * self.cfg.residual_roll_pitch_span_rad
        target_pitch = self.actions[:, 4] * self.cfg.residual_roll_pitch_span_rad
        target_yaw = self.actions[:, 5] * self.cfg.residual_yaw_span_rad
        target_quat = torch_utils.quat_from_euler_xyz(target_roll, target_pitch, target_yaw)
        if self.cfg.controller_mode == "spiral":
            target_pos, target_quat = self._spiral_target(fixed_action_frame)
        # Once the peg reaches the success geometry, latch the current pose so
        # subsequent policy noise cannot pull it back out before episode end.
        if torch.any(self.success_latched):
            target_pos[self.success_latched] = self.success_hold_pos[self.success_latched]
            target_quat[self.success_latched] = self.success_hold_quat[self.success_latched]
        # Apply bounds AFTER every controller/hold target override. Otherwise a
        # displaced peg can receive an unbounded spring command during hold.
        bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        target_pos = fixed_action_frame + torch.clamp(target_pos - fixed_action_frame, -bounds, bounds)
        hold = self.cfg.hold
        max_error = torch.tensor(
            [hold.max_xy_error_m, hold.max_xy_error_m, hold.max_z_error_m], device=self.device
        )
        target_pos = self.fingertip_midpoint_pos + torch.clamp(
            target_pos - self.fingertip_midpoint_pos, -max_error, max_error
        )
        self.generate_ctrl_signals(target_pos, target_quat, 0.0)
        self.evaluation_interval_peak_force = torch.maximum(
            self.evaluation_interval_peak_force,
            torch.linalg.vector_norm(self.applied_wrench[:, :3], dim=1),
        )

    def _get_rewards(self):
        # _get_dones has already refreshed kinematics at the final substep.
        # Refreshing again here would erase the finite-difference velocities.
        self._sample_evaluation_geometry()
        xy_error, depth, _ = self.insertion_geometry()
        cfg = self.cfg.reward
        terminal_hold_v2 = cfg.success_contract == "terminal_hold_v2"
        curr_success = (
            self.evaluation_success_geometry()
            if terminal_hold_v2
            else self._get_curr_successes(self.cfg_task.success_threshold)
        )
        curr_engaged = self._get_curr_successes(self.cfg_task.engage_threshold)
        if terminal_hold_v2:
            hold_state = advance_terminal_hold(
                self.training_hold_samples,
                self.training_ever_geometry_success,
                self.training_ever_held_success,
                self.evaluation_interval_success,
                self.evaluation_interval_ever_success,
                curr_success,
                self.required_training_hold_samples,
            )
            self.training_hold_samples[:] = hold_state.consecutive_samples
            self.training_ever_geometry_success[:] = hold_state.ever_geometry_success
            self.training_ever_held_success[:] = hold_state.ever_held_success
            first_success = hold_state.first_held_success
            per_step_success = self.evaluation_interval_success & curr_success
        else:
            first_success = torch.logical_and(curr_success, torch.logical_not(self.ep_succeeded.bool()))
            per_step_success = curr_success
        factory_terms, _ = self._get_factory_rew_dict(curr_success)

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
            "success_hold": cfg.success_hold_scale * per_step_success.float(),
            "action_cost": -cfg.action_scale * action_cost,
            "action_rate_cost": -cfg.action_rate_scale * action_rate_cost,
            "commanded_wrench_cost": -cfg.commanded_wrench_scale * wrench_cost,
        }
        if terminal_hold_v2:
            reward_terms["terminal_hold"] = cfg.terminal_hold_bonus * (
                hold_state.held_success & self.reset_time_outs
            ).float()
        reward = torch.zeros_like(depth)
        for term in reward_terms.values():
            reward += term
        # Baselines have no residual-policy action penalty. Preserve a separate
        # task return so their moving scripted targets are not called zero-cost.
        self.evaluation_task_reward = reward - reward_terms["action_cost"] - reward_terms["action_rate_cost"]

        if terminal_hold_v2:
            # Episode metrics are emitted only at timeout. Do not carry the
            # prior episode dictionary into every subsequent RL-Games step.
            self.extras.pop("episode", None)
            completed = self.reset_time_outs
            if torch.any(completed):
                self.extras["episode"] = {
                    "terminal_hold_v2/terminal_held_success": hold_state.held_success[completed].float().mean(),
                    "terminal_hold_v2/ever_geometry_success": self.training_ever_geometry_success[completed].float().mean(),
                    "terminal_hold_v2/final_geometry_success": curr_success[completed].float().mean(),
                    "terminal_hold_v2/ever_held_success": self.training_ever_held_success[completed].float().mean(),
                }
            for name, term in reward_terms.items():
                self.extras[f"logs_rew_{name}"] = term.mean()
        else:
            self._log_factory_metrics(reward_terms, curr_success)
        self.previous_depth[:] = depth
        self.previous_rl_action[:] = self.actions
        self.prev_actions = self.actions.clone()
        return reward
