"""Independent read-only CPU checkpoint/config/TensorBoard audit."""
from pathlib import Path
import hashlib,json,sys
import torch,yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

torch.set_num_threads(1)
r=Path(sys.argv[1])
run=r/'short_train/training_backup'
initial=torch.load(r/'z_calibrated_epoch113.pth',map_location='cpu',weights_only=False)
paths=list((run/'nn').glob('last_LocalInsertion_ep_121_*.pth'));assert len(paths)==1
final=torch.load(paths[0],map_location='cpu',weights_only=False)
assert (int(initial['epoch']),int(final['epoch']))==(113,121)
assert int(final['frame'])-int(initial['frame'])==1024*128*8==1048576
assert set(initial['model'])==set(final['model'])
assert all(final['model'][k].shape==initial['model'][k].shape for k in final['model'])
assert all(torch.isfinite(v).all() for v in final['model'].values() if v.is_floating_point())
env=yaml.load((run/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
agent=yaml.safe_load((run/'params/agent.yaml').read_text())
assert int(env['scene']['num_envs'])==agent['params']['config']['num_actors']==1024
assert agent['params']['config']['horizon_length']==128 and agent['params']['seed']==42
assert float(env['sim']['dt'])==1/120 and int(env['decimation'])==8
assert float(env['episode_length_s'])==151/15 and float(env['reward']['hold_duration_s'])==1
assert env['action_contract']=='relative_z_v1'
assert env['reward']['dense_profile']=='legacy' and env['reward']['success_contract']=='terminal_hold_v2'
assert int(env['sim']['physx']['max_position_iteration_count'])==192
solver=[]
def walk(v,path=''):
 if isinstance(v,dict):
  for key,val in v.items():
   if key=='solver_position_iteration_count':
    assert int(val)==192;solver.append(path+'/'+key)
   walk(val,path+'/'+key)
 elif isinstance(v,list):
  for i,val in enumerate(v):walk(val,path+f'/{i}')
walk(env);assert len(solver)>=4
meta=json.loads((r/'post_train_four/metadata.json').read_text())
sha=hashlib.sha256(paths[0].read_bytes()).hexdigest()
assert meta['checkpoint_sha256']==sha
events=EventAccumulator(str(run/'summaries'),size_guidance={'scalars':0});events.Reload()
curves={tag:[{'step':e.step,'value':e.value,'wall_time':e.wall_time} for e in events.Scalars(tag)] for tag in events.Tags()['scalars']}
assert not torch.cuda.is_initialized()
report={'passed':True,'cpu_only':True,'initial_epoch':113,'final_epoch':121,'initial_frame':int(initial['frame']),'final_frame':int(final['frame']),'added_transitions':1048576,'num_envs':1024,'horizon_length':128,'solver_paths_verified':solver,'finite_weights':True,'checkpoint_sha256':sha,'checkpoint_used_in_evaluation':True,'initial_checkpoint_sha256':hashlib.sha256((r/'z_calibrated_epoch113.pth').read_bytes()).hexdigest(),'tensorboard_scalars':curves}
print(json.dumps(report,indent=2,allow_nan=False))
