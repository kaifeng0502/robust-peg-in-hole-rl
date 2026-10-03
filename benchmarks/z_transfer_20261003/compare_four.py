"""Descriptive paired four-case report; no representative-rate inference."""
from pathlib import Path
import json,re,statistics
r=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
lines=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
labels=['reference','gate_z_minus_half','post_train_four']
groups={label:{x['case_id']:x for x in load(r/label/'trials.json')} for label in labels}
traces={label:lines(r/label/'trajectories.jsonl') for label in labels}
diagnostics={label:lines(r/label/'control_diagnostics.jsonl') for label in labels}
assert all(set(groups[l])==set(groups['reference']) for l in labels)
assert load(r/'gate_z_minus_half/metadata.json')['protocol_id']==load(r/'post_train_four/metadata.json')['protocol_id']
cases=[]
for case in groups['reference']:
 assert all(groups[label][case]['initial_state']==groups['reference'][case]['initial_state'] for label in labels)
 row={'case_id':case,'stages':{}}
 for label in labels:
  trial=groups[label][case];ts=[x for x in traces[label] if x['case_id']==case]
  ds=[x for x in diagnostics[label] if x['seed']==trial['reset_seed']]
  assert len(ts)==len(ds)==150
  assert abs(sum(x['reward'] for x in ts)-trial['episode_reward'])<1e-5
  assert abs(sum(x['task_reward'] for x in ts)-trial['task_episode_reward'])<1e-5
  assert all(x['final_success']==(x['xy_error_m']<=.0025 and abs(x['depth_m']-.025)<=.001) for x in ts)
  actions=[t['action'][2] for t,d in zip(ts,ds) if not d['substeps'][0]['success_latched'][0]]
  row['stages'][label]={key:trial[key] for key in ['held_success','final_xy_error_m','final_depth_m','first_hold_time_s','peak_commanded_force_n','episode_reward','task_episode_reward','failure_category']}
  row['stages'][label].update(unlatched_raw_z_mean=statistics.mean(actions),unlatched_raw_z_positive_fraction=sum(a>0 for a in actions)/len(actions),unlatched_raw_z_lower_clip_fraction=sum(a<=-.999 for a in actions)/len(actions),max_depth_m=max(t['depth_m'] for t in ts))
 cases.append(row)
log_path=r/'short_train/training.log'
if not log_path.exists():log_path=r/'short_train/training_runtime.txt'
log=log_path.read_text()
fps=[int(x) for x in re.findall(r'fps total: (\d+)',log)]
epochs=[int(x) for x in re.findall(r'epoch: (\d+)/121',log)];assert epochs==list(range(114,122))
memory={}
for sample in lines(r/'short_train/gpu_samples.jsonl'):
 if sample['query_exit']==0:
  memory.setdefault(sample['phase'],[]).append(int(sample['sample'].split(',')[0]))
status=load(r/'short_train/status.json')
report={'cases':cases,'paired':{'both_held_success':sum(x['stages'][labels[1]]['held_success'] and x['stages'][labels[2]]['held_success'] for x in cases),'gained_successes':sum(not x['stages'][labels[1]]['held_success'] and x['stages'][labels[2]]['held_success'] for x in cases),'lost_successes':sum(x['stages'][labels[1]]['held_success'] and not x['stages'][labels[2]]['held_success'] for x in cases)},'stage_summary':{label:load(r/label/'summary.json') for label in labels},'training':{'logged_epochs':epochs,'logged_total_fps':fps,'median_logged_total_fps':statistics.median(fps),'training_wall_s':status['training_elapsed_s'],'four_case_wall_s':status['four_case_postcheck_elapsed_s'],'device_sampled_peak_mib':{k:max(v) for k,v in memory.items()},'memory_note':'Device-wide samples every 10 seconds, not allocator peaks.'},'limits':['Four reused selected development cases, not a generalization estimate.','Legacy absolute-interface reference is historical context, not an isolated action-only comparison.','Identical physical reset states do not imply identical closed-loop trajectories.','Hold audit uses recorded full-interval geometry flags; substep pose records and reward/target reconstruction are preserved.','No new successful case; one XY error improves and the other worsens.','The remaining depth failure is deeper than the calibrated initializer but shallower than the original absolute-interface reference.']}
(r/'four_case_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'paired':report['paired'],'training':report['training'],'Z':{x['case_id']:x['stages']['post_train_four']['unlatched_raw_z_mean'] for x in cases}},indent=2))
