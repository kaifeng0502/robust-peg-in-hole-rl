"""Opt-in per-physics-substep control diagnostics on the reference evaluator.

Uses one environment for seeded replay, not a training workload. No policy,
physics or reward changes. Reward terms are exposed by the isolated env copy.
"""
from pathlib import Path
import json
import sys
import torch
import evaluate_insertion as evaluation


class DiagnosticAdapter(evaluation.SimulationAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = False
        self.records = []
        self.layout = None
        self.base.capture_diagnostics = True
        original = self.base.generate_ctrl_signals

        def generate(ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos):
            target_pos, target_quat, gripper_pos = ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
            if not self.enabled:
                return original(target_pos, target_quat, gripper_pos)
            base = self.base
            frame = base.fixed_pos_obs_frame + base.init_fixed_pos_obs_noise
            raw = frame.clone()
            raw[:, 2] += base.cfg.nominal_tool_to_peg_base_m - base.cfg.nominal_insertion_depth_m
            raw[:, :2] += base.actions[:, :2] * base.cfg.residual_xy_span_m
            raw[:, 2] += base.actions[:, 2] * base.cfg.residual_z_span_m
            if base.cfg.action_contract == "relative_z_v1":
                raw[:, 2] = base.relative_z_command
            effective = torch.where(base.success_latched[:, None], base.success_hold_pos, raw)
            bounds = torch.tensor(base.cfg.ctrl.pos_action_bounds, device=base.device)
            bounded = frame + torch.clamp(effective - frame, -bounds, bounds)
            limits = torch.tensor([base.cfg.hold.max_xy_error_m]*2 + [base.cfg.hold.max_z_error_m], device=base.device)
            recomputed = base.fingertip_midpoint_pos + torch.clamp(bounded-base.fingertip_midpoint_pos, -limits, limits)
            raw_z_ends = frame[:, 2:3] + base.cfg.nominal_tool_to_peg_base_m - base.cfg.nominal_insertion_depth_m + torch.tensor([-1., 1.],device=base.device)*base.cfg.residual_z_span_m
            if base.cfg.action_contract == "relative_z_v1":
                # Here endpoints describe a fresh control decision at this pose,
                # not a change to the held command within the current interval.
                raw_z_ends = base.fingertip_midpoint_pos[:, 2:3] + torch.tensor([-1., 1.], device=base.device)*base.cfg.relative_z_step_m
            bounded_z_ends = frame[:, 2:3] + torch.clamp(raw_z_ends-frame[:, 2:3], -bounds[2], bounds[2])
            z_ends = base.fingertip_midpoint_pos[:, 2:3] + torch.clamp(bounded_z_ends-base.fingertip_midpoint_pos[:, 2:3], -limits[2], limits[2])
            fields = {
                'tool_position':base.fingertip_midpoint_pos,
                'tool_quaternion':base.fingertip_midpoint_quat,
                'peg_position':base.held_pos,
                'peg_quaternion':base.held_quat,
                'noisy_hole_top':frame,
                'filtered_action':base.actions,
                'nominal_target':raw,
                'effective_raw_target':effective,
                'workspace_bounded_target':bounded,
                'command_target':target_pos,
                'command_quaternion':target_quat,
                'recomputed_target_error':recomputed-target_pos,
                'impedance_clip_mask':(torch.abs(bounded-base.fingertip_midpoint_pos)>limits).float(),
                'legal_z_endpoint_commands':z_ends,
                'success_latched':base.success_latched[:,None].float(),
            }
            # Snapshot before control execution; no kinematic refresh or RNG calls.
            snapshots = {name:value.detach().clone() for name,value in fields.items()}
            result = original(target_pos, target_quat, gripper_pos)
            snapshots['commanded_wrench'] = base.applied_wrench.detach().clone()
            layout = [(name,value.shape[-1]) for name,value in snapshots.items()]
            if self.layout is None:
                self.layout = layout
            assert self.layout == layout
            self.records.append(torch.cat(list(snapshots.values()),dim=-1)[0])
            return result
        self.base.generate_ctrl_signals = generate

    def reset(self, seed):
        self.enabled = False
        self.records = []
        result = super().reset(seed)
        self.seed = seed
        self.enabled = True
        return result

    def step(self, action):
        self.records = []
        sample = super().step(action)
        if len(self.records) != self.base.cfg.decimation:
            raise RuntimeError('Unexpected number of recorded physics substeps')
        values = torch.stack(self.records).detach().cpu().tolist()
        substeps = []
        for row in values:
            offset, record = 0, {}
            for name,width in self.layout:
                record[name] = row[offset:offset+width]
                offset += width
            substeps.append(record)
        terms = self.base.diagnostic_reward_terms
        rewards = dict(zip(terms, torch.stack([v[0] for v in terms.values()]).detach().cpu().tolist()))
        if abs(sum(rewards.values())-sample.reward)>1e-4:
            raise RuntimeError('Reward components do not reconcile')
        record = {'seed':self.seed,'step':sample.episode_step,'substeps':substeps,
                  'reward_terms':rewards,'reward':sample.reward,
                  'observation_after':sample.observation['policy'][0].detach().cpu().tolist()}
        with (OUTPUT/'control_diagnostics.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        return sample


if __name__ == '__main__':
    args = evaluation.parse_args()
    if args.method != 'ppo':
        raise ValueError('This diagnostic reconstruction currently supports residual PPO only')
    OUTPUT = args.output
    evaluation.SimulationAdapter = DiagnosticAdapter
    raise SystemExit(evaluation.main())
