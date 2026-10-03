"""Experimental, instance-local target update with fixed-shape success selection.

Install only after the real environment has completed construction and reset.
Configuration, device and default floating-point dtype must then remain fixed.
This module does not modify the environment class, geometry, or force controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MethodType

import torch


@dataclass(frozen=True)
class TargetConstants:
    bounds: torch.Tensor
    max_error: torch.Tensor
    downward_margin: torch.Tensor


def make_constants(base) -> TargetConstants:
    """Match the original constant constructors, once outside measured stepping."""
    hold = base.cfg.hold
    return TargetConstants(
        bounds=torch.tensor(base.cfg.ctrl.pos_action_bounds, device=base.device),
        max_error=torch.tensor(
            [hold.max_xy_error_m, hold.max_xy_error_m, hold.max_z_error_m],
            device=base.device,
        ),
        downward_margin=torch.tensor(
            hold.downward_margin_m,
            device=base.success_hold_pos.device,
            dtype=base.success_hold_pos.dtype,
        ),
    )


def update_success_hold(
    current_success,
    success_latched,
    success_hold_pos,
    success_hold_quat,
    fingertip_pos,
    fingertip_quat,
    hold_enabled,
    downward_margin,
):
    """Preserve the legacy latch in-place without reading CUDA predicates on CPU."""
    newly_success = current_success & ~success_latched & hold_enabled
    # Subtract only in Z, just as the original assignment does. Subtracting a
    # three-vector of [0, 0, margin] could change signed-zero bits in X or Y.
    new_hold_pos = fingertip_pos.clone()
    new_hold_pos[:, 2] -= downward_margin
    success_hold_pos.copy_(torch.where(newly_success[:, None], new_hold_pos, success_hold_pos))
    success_hold_quat.copy_(torch.where(newly_success[:, None], fingertip_quat, success_hold_quat))
    success_latched |= newly_success


def select_and_bound_targets(
    target_pos,
    target_quat,
    success_latched,
    success_hold_pos,
    success_hold_quat,
    fixed_action_frame,
    fingertip_pos,
    bounds,
    max_error,
):
    """Apply hold, absolute position bounds, then impedance-error bounds."""
    target_pos = torch.where(success_latched[:, None], success_hold_pos, target_pos)
    target_quat = torch.where(success_latched[:, None], success_hold_quat, target_quat)
    target_pos = fixed_action_frame + torch.clamp(target_pos - fixed_action_frame, -bounds, bounds)
    target_pos = fingertip_pos + torch.clamp(target_pos - fingertip_pos, -max_error, max_error)
    return target_pos, target_quat


def install(base):
    """Replace one initialized instance's _apply_action; return idempotent restore.

    Install before performance instrumentation wraps this method. The original
    method's quaternion implementation is reused, so importing this module does
    not itself import or initialize Isaac Sim.
    """
    original = base._apply_action
    function = getattr(original, "__func__", None)
    utilities = function.__globals__.get("torch_utils") if function is not None else None
    if utilities is None or not hasattr(utilities, "quat_from_euler_xyz"):
        raise TypeError("Install on the original bound environment method before instrumentation")
    quat_from_euler_xyz = utilities.quat_from_euler_xyz
    constants = make_constants(base)
    absent = object()
    previous_override = vars(base).get("_apply_action", absent)

    def apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        self._sample_evaluation_geometry()
        fixed_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        current_success = self._get_curr_successes(self.cfg_task.success_threshold)
        update_success_hold(
            current_success,
            self.success_latched,
            self.success_hold_pos,
            self.success_hold_quat,
            self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat,
            self.cfg.hold.enabled,
            constants.downward_margin,
        )
        target_pos = fixed_action_frame.clone()
        target_pos[:, 2] += self.cfg.nominal_tool_to_peg_base_m - self.cfg.nominal_insertion_depth_m
        target_pos[:, :2] += self.actions[:, :2] * self.cfg.residual_xy_span_m
        target_pos[:, 2] += self.actions[:, 2] * self.cfg.residual_z_span_m

        target_roll = math.pi + self.actions[:, 3] * self.cfg.residual_roll_pitch_span_rad
        target_pitch = self.actions[:, 4] * self.cfg.residual_roll_pitch_span_rad
        target_yaw = self.actions[:, 5] * self.cfg.residual_yaw_span_rad
        target_quat = quat_from_euler_xyz(target_roll, target_pitch, target_yaw)
        if self.cfg.controller_mode == "spiral":
            target_pos, target_quat = self._spiral_target(fixed_action_frame)
        target_pos, target_quat = select_and_bound_targets(
            target_pos,
            target_quat,
            self.success_latched,
            self.success_hold_pos,
            self.success_hold_quat,
            fixed_action_frame,
            self.fingertip_midpoint_pos,
            constants.bounds,
            constants.max_error,
        )
        self.generate_ctrl_signals(target_pos, target_quat, 0.0)
        self.evaluation_interval_peak_force = torch.maximum(
            self.evaluation_interval_peak_force,
            torch.linalg.vector_norm(self.applied_wrench[:, :3], dim=1),
        )

    base._apply_action = MethodType(apply_action, base)
    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        if previous_override is absent:
            delattr(base, "_apply_action")
        else:
            base._apply_action = previous_override
        restored = True

    return restore
