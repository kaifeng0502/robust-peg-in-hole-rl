"""Independent four-case behavioral gate; no simulator or torch dependency."""
from pathlib import Path
import hashlib, json, math, sys

root=Path(__file__).resolve().parent
out=root/(sys.argv[1] if len(sys.argv)>1 else 'gate_z_minus_half')
load=lambda p:json.loads(p.read_text())
lines=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
meta=load(out/'metadata.json')
assert meta['run_complete'] and meta['trial_count']==4
assert meta['protocol']['horizon_steps']==150
cfg=meta['protocol']['shared_environment_config']
assert cfg['sim']['dt']==1/120 and cfg['decimation']==8
assert cfg['action_contract']=='relative_z_v1'
assert meta['criteria']['hold_duration_s']==1.0
trials=load(out/'trials.json')
old={t['case_id']:t for t in load(root/'reference/trials.json')}
traces=lines(out/'trajectories.jsonl');diags=lines(out/'control_diagnostics.jsonl')
assert len(traces)==len(diags)==600 and len(trials)==4
assert {t['case_id'] for t in trials}==set(old)
def finite(value):
 if isinstance(value,dict):return all(finite(v) for v in value.values())
 if isinstance(value,list):return all(finite(v) for v in value)
 return not isinstance(value,float) or math.isfinite(value)
assert finite([traces,diags,trials])
result=[]
for trial in trials:
 case=trial['case_id'];previous=old[case]
 rows=[r for r in traces if r['case_id']==case]
 diagnostic=[r for r in diags if r['seed']==trial['reset_seed']]
 assert [r['step'] for r in rows]==list(range(1,151))
 assert [r['step'] for r in diagnostic]==list(range(1,151))
 assert all(len(r['substeps'])==8 for r in diagnostic)
 assert all(abs(r['time_s']-r['step']/15)<1e-9 for r in rows)
 for r,d in zip(rows,diagnostic):
  assert abs(sum(d['reward_terms'].values())-r['reward'])<1e-4
  assert all(abs(v)<1e-7 for s in d['substeps'] for v in s['recomputed_target_error'])
 # Strict final fifteen full control intervals, not just last pose.
 held=all(r['interval_success'] for r in rows[-15:])
 assert held==trial['held_success']
 assert abs(rows[-1]['depth_m']-trial['final_depth_m'])<1e-9
 peak=max(r['commanded_force_n'] for r in rows)
 assert abs(peak-trial['peak_commanded_force_n'])<1e-9
 unlatched=[d['substeps'][0] for d in diagnostic if not d['substeps'][0]['success_latched'][0]]
 delta=[s['command_target'][2]-s['tool_position'][2] for s in unlatched]
 depth=max(r['depth_m'] for r in rows)
 entry=next((r['time_s'] for r in rows if r['depth_m']>=-.001),None)
 result.append({'case_id':case,'initial_state_exact':trial['initial_state']==previous['initial_state'],
  'held_success':held,'old_held_success':previous['held_success'],'all_unlatched_initial_commands_down':bool(delta) and max(delta)<0,
  'command_delta_z_range_m':[min(delta),max(delta)],'max_depth_m':depth,'reached_entrance':depth>=-.001,
  'first_within_1mm_of_entrance_s':entry,'final_depth_m':rows[-1]['depth_m'],
  'initial_xy_error_m':rows[0]['xy_error_m'],'final_xy_error_m':rows[-1]['xy_error_m'],
  'peak_commanded_force_n':peak,'old_peak_commanded_force_n':previous['peak_commanded_force_n'],
  'force_regression_check':peak<=2*previous['peak_commanded_force_n'],
  'failure_category':trial['failure_category']})
checks={'matched_initial_states':all(r['initial_state_exact'] for r in result),
 'all_approach_commands_down':all(r['all_unlatched_initial_commands_down'] for r in result),
 'all_reach_entrance':all(r['reached_entrance'] for r in result),
 'original_success_retained':all(r['held_success'] for r in result if r['old_held_success']),
 'force_regression_bound':all(r['force_regression_check'] for r in result)}
report={'passed':all(checks.values()),'checks':checks,'cases':result,'control_rows':600,'physics_substeps':4800,'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},'scope':'Four selected development cases for debugging only, not a success-rate estimate.'}
(root/(out.name+'_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
