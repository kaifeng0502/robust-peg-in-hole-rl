"""One bounded 1024x128x8 PPO continuation after the audited four-case gate."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time
import torch, yaml

root=Path(__file__).resolve().parent
source=root/'source'
out=root/'short_train'
lock=(root/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not out.exists()
assert json.loads((root/'gate_z_minus_half_audit.json').read_text())['passed']
sys.path[:0]=[str(source/'scripts')]
from evaluate_insertion import source_fingerprint
fingerprint=source_fingerprint()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checkpoint=root/'z_calibrated_epoch113.pth'
assert sha(checkpoint)==json.loads((root/'calibration_audit.json').read_text())['migrated_sha256']
initial=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert initial['epoch']==113
out.mkdir()
status={'complete':False,'phase':'prepared','supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'max_new_transitions':1048576,'num_envs':1024,'source_fingerprint':fingerprint,'initial_checkpoint_sha256':sha(checkpoint)}
def write(name,value):
 p=out/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def run(command,phase,timeout):
 write(phase+'_command.json',command)
 with (out/(phase+'.log')).open('x') as stream:
  p=subprocess.Popen(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  status.update(phase=phase,pid=p.pid,phase_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write('status.json',status)
  begin=time.monotonic()
  try:
   while p.poll() is None:
    if time.monotonic()-begin>timeout:raise TimeoutError(phase)
    memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    with (out/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'phase':phase,'sample':memory.stdout.strip(),'query_exit':memory.returncode})+'\n')
    time.sleep(10)
   status[phase+'_exit']=p.returncode
   status[phase+'_elapsed_s']=time.monotonic()-begin
   if p.returncode:raise RuntimeError(f'{phase} exited {p.returncode}')
  finally:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   write('status.json',status)
 assert source_fingerprint()==fingerprint

try:
 name='z_calibrated_1024_8rollouts_20261003'
 directory=source/'logs/rl_games/LocalInsertion'/name
 assert not directory.exists()
 command=[sys.executable,'scripts/train_rl.py','--task','Isaac-LocalInsertion-RL-Direct-v0','--num_envs','1024','--seed','42','--headless','--checkpoint',str(checkpoint),'--max_iterations','121','env.episode_length_s=10.066666666666666','env.action_contract=relative_z_v1','env.reward.success_contract=terminal_hold_v2','env.reward.dense_profile=legacy','agent.params.config.horizon_length=128',f'+agent.params.config.full_experiment_name={name}']
 run(command,'training',1800)
 agent=yaml.safe_load((directory/'params/agent.yaml').read_text())
 env=yaml.load((directory/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
 assert int(env['scene']['num_envs'])==agent['params']['config']['num_actors']==1024
 assert agent['params']['config']['horizon_length']==128
 assert env['action_contract']=='relative_z_v1' and env['reward']['dense_profile']=='legacy'
 assert env['reward']['success_contract']=='terminal_hold_v2'
 assert float(env['episode_length_s'])==151/15 and float(env['sim']['dt'])==1/120 and int(env['decimation'])==8
 files=list((directory/'nn').glob('last_LocalInsertion_ep_121_rew_*.pth'));assert len(files)==1
 final=torch.load(files[0],map_location='cpu',weights_only=False)
 added=int(final['frame'])-int(initial['frame'])
 assert final['epoch']==121 and added==1048576
 assert all(torch.isfinite(v).all() for v in final['model'].values() if v.is_floating_point())
 write('training_budget.json',{'initial_frame':int(initial['frame']),'final_frame':int(final['frame']),'new_transitions':added,'initial_epoch':113,'final_epoch':121,'checkpoint':str(files[0]),'checkpoint_sha256':sha(files[0])})
 shutil.copytree(directory,out/'training_backup')
 run([sys.executable,'scripts/diagnose_insertion.py','--method','ppo','--headless','--action-contract','relative_z_v1','--cases',str(root/'diagnostic_cases.json'),'--checkpoint',str(files[0]),'--agent-config',str(directory/'params/agent.yaml'),'--output',str(root/'post_train_four')],'four_case_postcheck',1200)
 status.update(phase='remote_complete_awaiting_local_audit',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
except BaseException as exc:
 status.update(phase='failed',error=repr(exc))
 raise
finally:write('status.json',status)
