"""User-authorized 1024 x 64 x 500 continuation, with no intermediate eval."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, shutil, signal, subprocess, sys, time
import torch, yaml

root=Path(__file__).resolve().parent
source=root/'source_save100';out=root/'run_save100'
lock=(root/'training_save100.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not out.exists(), 'Existing run: inspect it, never launch a duplicate'
plan=json.loads((root/'plan_save100.json').read_text())
sys.path.insert(0,str(source/'scripts'))
from evaluate_insertion import source_fingerprint
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fingerprint=source_fingerprint()
assert fingerprint=='1d0ecd6cdd7fb7f79fe864bd2275cb857058039c192ad176b1b02d22293aa67f'
checkpoint=Path(plan['initial_checkpoint'])
assert sha(checkpoint)==plan['initial_checkpoint_sha256']
initial=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert int(initial['epoch'])==133 and int(initial['frame'])==3686400
assert all(torch.isfinite(v).all() for v in initial['model'].values() if v.is_floating_point())
assert plan['num_envs']*plan['horizon_length']*plan['additional_rollouts']==31981568
name='calibrated_ppo_h64_500_save100_20261003'
directory=source/'logs/rl_games/LocalInsertion'/name
assert not directory.exists()
out.mkdir()
status={'complete':False,'phase':'prepared','supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initial_epoch':133,'final_epoch':621,'max_added_transitions':31981568,'num_envs':1024,'horizon_length':64,'source_fingerprint':fingerprint,'initial_checkpoint_sha256':sha(checkpoint),'training_config_verified':False}
def write(name,data):
 p=out/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def verify_config():
 p=directory/'params/agent.yaml';q=directory/'params/env.yaml'
 if not p.exists() or not q.exists():return False
 a=yaml.safe_load(p.read_text());e=yaml.load(q.read_text(),Loader=yaml.BaseLoader)
 c=a['params']['config']
 assert int(e['scene']['num_envs'])==c['num_actors']==1024
 assert c['horizon_length']==64 and c['max_epochs']==621
 assert c['minibatch_size']==2048 and c['mini_epochs']==4
 assert c['save_frequency']==100 and c['save_best_after']==1000000000 and a['params']['seed']==42
 assert c['gamma']==.99 and c['tau']==.95
 assert e['action_contract']=='relative_z_v1'
 assert e['reward']['dense_profile']=='legacy' and e['reward']['success_contract']=='terminal_hold_v2'
 assert float(e['episode_length_s'])==151/15 and float(e['reward']['hold_duration_s'])==1
 assert float(e['sim']['dt'])==1/120 and int(e['decimation'])==8
 assert int(e['sim']['physx']['max_position_iteration_count'])==192
 def check(v):
  if isinstance(v,dict):
   for key,val in v.items():
    if key=='solver_position_iteration_count':assert int(val)==192
    check(val)
  elif isinstance(v,list):
   for val in v:check(val)
 check(e)
 return True
def run(command,phase,timeout):
 write(phase+'_command.json',command)
 log=out/(phase+'.log')
 with log.open('x') as stream:
  p=subprocess.Popen(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  status.update(phase=phase,pid=p.pid,phase_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write('status.json',status)
  begin=time.monotonic()
  try:
   while p.poll() is None:
    elapsed=time.monotonic()-begin
    if elapsed>timeout:raise TimeoutError(phase)
    if phase=='training':
     if not status['training_config_verified']:status['training_config_verified']=verify_config()
     epochs=re.findall(r'fps total: (\d+) epoch: (\d+)/621',log.read_text(errors='replace'))
     if epochs:
      fps,epoch=map(int,epochs[-1]);assert 133<epoch<=621
      status.update(last_logged_epoch=epoch,logged_new_rollouts=epoch-133,last_logged_total_fps=fps,derived_added_samples=(epoch-133)*65536)
    sample=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    with (out/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'phase':phase,'sample':sample.stdout.strip(),'query_exit':sample.returncode})+'\n')
    status['phase_elapsed_s']=elapsed;write('status.json',status)
    time.sleep(30)
   status[phase+'_exit']=p.returncode;status[phase+'_elapsed_s']=time.monotonic()-begin
   if p.returncode:raise RuntimeError(f'{phase} exited {p.returncode}')
  finally:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   write('status.json',status)
 assert source_fingerprint()==fingerprint

try:
 shutil.copy(checkpoint,out/'initial_checkpoint.pth')
 write('plan.json',plan)
 cmd=[sys.executable,'scripts/train_rl.py','--task','Isaac-LocalInsertion-RL-Direct-v0','--num_envs','1024','--seed','42','--headless','--checkpoint',str(checkpoint),'--max_iterations','621','env.episode_length_s=10.066666666666666','env.action_contract=relative_z_v1','env.reward.success_contract=terminal_hold_v2','env.reward.dense_profile=legacy','agent.params.config.horizon_length=64','agent.params.config.save_frequency=100','agent.params.config.save_best_after=1000000000',f'+agent.params.config.full_experiment_name={name}']
 run(cmd,'training',12*3600)
 assert verify_config()
 files=list((directory/'nn').glob('last_LocalInsertion_ep_621_rew_*.pth'));assert len(files)==1
 final=torch.load(files[0],map_location='cpu',weights_only=False)
 added=int(final['frame'])-int(initial['frame'])
 assert final['epoch']==621 and added==31981568 and final['frame']==35667968
 assert all(torch.isfinite(v).all() for v in final['model'].values() if v.is_floating_point())
 write('training_budget.json',{'initial_epoch':133,'final_epoch':621,'initial_frame':int(initial['frame']),'final_frame':int(final['frame']),'added_transitions':added,'checkpoint':str(files[0]),'checkpoint_sha256':sha(files[0])})
 shutil.copytree(directory,out/'training_backup')
 run([sys.executable,'scripts/diagnose_insertion.py','--method','ppo','--headless','--action-contract','relative_z_v1','--cases',str(root/'diagnostic_cases.json'),'--checkpoint',str(files[0]),'--agent-config',str(directory/'params/agent.yaml'),'--output',str(out/'final_four_cases')],'final_four_case_check',1200)
 status.update(phase='remote_complete_awaiting_local_audit',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_added_transitions=added)
except BaseException as exc:
 status.update(phase='failed',error=repr(exc));raise
finally:write('status.json',status)
