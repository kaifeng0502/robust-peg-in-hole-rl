"""Run four development cases only; no training or broad evaluation queue."""
from pathlib import Path
import datetime, json, os, signal, subprocess, sys, time
root=Path(__file__).resolve().parent
out=root/'gate_z_minus_half'
assert not out.exists()
status={'phase':'four_case_gate','complete':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_samples_added':0}
def write():
 p=root/'status.tmp';p.write_text(json.dumps(status,indent=2)+'\n');p.replace(root/'status.json')
command=[sys.executable,'scripts/diagnose_insertion.py','--method','ppo','--headless','--action-contract','relative_z_v1','--cases',str(root/'diagnostic_cases.json'),'--checkpoint',str(root/'z_calibrated_epoch113.pth'),'--agent-config','/root/autodl-tmp/insertion/results/held_v2_pilot_20261003/training/held_v2/params/agent.yaml','--output',str(out)]
(root/'gate_command.json').write_text(json.dumps(command,indent=2)+'\n')
with (root/'gate.log').open('x') as stream:
 child=subprocess.Popen(command,cwd=root/'source',stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 status.update(pid=child.pid,supervisor_pid=os.getpid());write()
 try:
  code=child.wait(timeout=1200)
  status.update(phase='gate_awaiting_audit' if code==0 else 'gate_failed',exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  status.update(phase='gate_timeout',exit_code=child.returncode)
 finally:write()
