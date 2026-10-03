"""Control reachability and shaping checks using actual environment method bodies."""
import ast
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_insertion'))
sys.path.insert(0, str(ROOT/'tools/performance'))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required')
class ContactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from contact_contract import relative_z_target, entry_potentials
        from test_target_updates import FakeEnv, quat_from_euler_xyz
        cls.potentials = staticmethod(entry_potentials)
        class Factory:
            def _pre_physics_step(self, action):
                self.actions = 0.35*action+0.65*self.actions
        tree = ast.parse((ROOT/'local_insertion/envs.py').read_text())
        node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocalInsertionRLEnv')
        node = copy.deepcopy(node)
        node.bases = [ast.Name(id='Factory',ctx=ast.Load())]
        node.body = [n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('_pre_physics_step','_apply_action')]
        ns = dict(torch=torch, math=math, Factory=Factory, relative_z_target=relative_z_target,
                  torch_utils=SimpleNamespace(quat_from_euler_xyz=quat_from_euler_xyz))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/'local_insertion/envs.py'),'exec'),ns)
        cls.Env = type('Harness',(ns['LocalInsertionRLEnv'],FakeEnv),{})

    def make_env(self, contract, count=3):
        e=self.Env(count=count,hold=False)
        e.cfg.action_contract=contract
        e.cfg.relative_z_step_m=.003
        e.cfg.hold.max_z_error_m=.003
        e.cfg.ctrl.pos_action_bounds=[.020,.020,.060]
        e.cfg.nominal_tool_to_peg_base_m=.032
        e.cfg.nominal_insertion_depth_m=.0245
        e.cfg.residual_z_span_m=.003
        e.fixed_pos_obs_frame[:]=0
        e.init_fixed_pos_obs_noise[:]=0
        e.fingertip_midpoint_pos[:]=0
        e.fingertip_midpoint_pos[:,2]=.035
        e.last_update_timestamp=1
        e.current_success[:]=False
        e.actions[:]=0
        e.relative_z_command=torch.zeros(count)
        e.evaluation_interval_success=torch.zeros(count,dtype=torch.bool)
        e.evaluation_interval_ever_success=torch.zeros(count,dtype=torch.bool)
        return e

    def test_initial_z_actions_are_distinct_and_include_lift_after_real_bounds(self):
        action=torch.zeros(3,6); action[:,2]=torch.tensor([-1.,0.,1.])
        original=self.make_env('absolute_residual_v1')
        revised=self.make_env('relative_z_v1')
        for e in (original,revised):
            e._pre_physics_step(action); e._apply_action()
        torch.testing.assert_close(original.command_pos[:,2],torch.full((3,),.032))
        deltas=revised.command_pos[:,2]-.035
        self.assertLess(deltas[0],0); self.assertEqual(deltas[1],0); self.assertGreater(deltas[2],0)
        self.assertLessEqual(deltas.abs().max(),.003)

    def test_target_is_anchored_once_not_integrated_eight_times(self):
        e=self.make_env('relative_z_v1',1)
        a=torch.zeros(1,6); a[:,2]=1
        e._pre_physics_step(a)
        anchor=e.relative_z_command.clone()
        for _ in range(8):
            e.fingertip_midpoint_pos[:,2]+=.00005
            e._apply_action()
            torch.testing.assert_close(e.relative_z_command,anchor,rtol=0,atol=0)
            torch.testing.assert_close(e.command_pos[:,2],anchor)
        e._pre_physics_step(a)
        self.assertGreater(e.relative_z_command[0],anchor[0])

    def test_hold_override_and_final_bounds_still_dominate_policy(self):
        e=self.make_env('relative_z_v1',1)
        e.cfg.hold.enabled=True
        e.success_latched[:]=True
        e.success_hold_pos[:]=torch.tensor([[.5,.5,-.5]])
        e.relative_z_command[:]=.5
        e._apply_action()
        delta=e.command_pos-e.fingertip_midpoint_pos
        self.assertTrue(torch.all(delta.abs()<=torch.tensor([.004,.004,.003])+1e-8))
        self.assertLess(delta[0,2],0)

    def test_coarse_signal_at_six_mm_and_continuous_entry(self):
        xy=torch.tensor([.007,.006,.004,.002,0.],dtype=torch.float64)
        align, entry=self.potentials(xy,torch.zeros_like(xy),.003,.008,.01,.0025)
        self.assertTrue(torch.all(align[1:]>align[:-1]))
        self.assertGreater(align[1],.25)
        self.assertTrue(torch.isfinite(entry).all())
        d=torch.tensor([-1e-8,0,1e-8,.0025-1e-8,.0025,.0025+1e-8],dtype=torch.float64)
        _,p=self.potentials(torch.zeros_like(d),d,.003,.008,.01,.0025)
        self.assertLess(abs(p[0]-p[2]),1e-5)
        self.assertLess(abs(p[3]-p[5]),1e-5)

    def test_closed_search_path_cannot_accumulate_entry_progress(self):
        xy=torch.tensor([.006,.003,.001,.003,.006],dtype=torch.float64)
        d=torch.tensor([0,.001,.003,.001,0.],dtype=torch.float64)
        a,p=self.potentials(xy,d,.003,.008,.01,.0025)
        self.assertAlmostEqual(float(torch.diff(a).sum()),0,places=12)
        self.assertAlmostEqual(float(torch.diff(p).sum()),0,places=12)


if torch is not None:
    from test_reward_contract import ProductionRewardContractTest
    from contact_contract import entry_potentials

    class EntryRewardProductionTests(ProductionRewardContractTest):
        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            cls.environment_type._get_rewards.__globals__['entry_potentials'] = entry_potentials

        def test_stalled_approach_stops_paying_approach_component(self):
            env=self.make_env('terminal_hold_v2')
            env.cfg.reward.dense_profile='entry_v1'
            env.cfg.reward.approach_scale=1.5
            env.xy[:]=.006; env.depth[:]=0
            env.previous_xy_error[:]=env.xy; env.previous_depth[:]=env.depth
            env.evaluation_interval_success[:]=False
            reward=env._get_rewards()
            self.assertEqual(float(env.extras['logs_rew_approach']),0)
            self.assertEqual(float(env.extras['logs_rew_entry_progress']),0)
            self.assertEqual(float(reward),0)

        def test_coarse_alignment_improvement_is_positive_before_entry(self):
            env=self.make_env('terminal_hold_v2')
            env.cfg.reward.dense_profile='entry_v1'
            env.xy[:]=.005; env.previous_xy_error[:]=.007
            env.depth[:]=-.002; env.previous_depth[:]=-.002
            env.evaluation_interval_success[:]=False
            reward=env._get_rewards()
            self.assertGreater(float(env.extras['logs_rew_alignment_progress']),0)
            self.assertGreater(float(env.extras['logs_rew_entry_progress']),0)
            self.assertGreater(float(reward),0)

if __name__=='__main__':
    unittest.main()
