"""CPU-only, immutable-output Z head migration and paired-observation audit."""
from pathlib import Path
import copy, hashlib, json, sys
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
source = Path(sys.argv[1])
destination = root / 'z_calibrated_epoch113.pth'
assert not destination.exists()
old = torch.load(source, map_location='cpu', weights_only=False)
new = copy.deepcopy(old)
model = new['model']
weight, bias = 'a2c_network.mu.weight', 'a2c_network.mu.bias'
model[weight][2].zero_()
model[bias][2] = -0.5
# Verify the checkpoint's parameter order and shapes before touching moments.
names = [k for k in model if k.startswith('a2c_network.')]
ids = new['optimizer']['param_groups'][0]['params']
assert len(ids) == len(names) == 12
for name, ident in zip(names, ids):
    state = new['optimizer']['state'][ident]
    assert state['exp_avg'].shape == model[name].shape
    assert state['exp_avg_sq'].shape == model[name].shape
    if name in (weight, bias):
        for key in ('exp_avg', 'exp_avg_sq'):
            state[key][2].zero_()
for name in model:
    if name in (weight, bias):
        assert torch.equal(model[name][[0,1,3,4,5]],old['model'][name][[0,1,3,4,5]])
    else:
        assert torch.equal(model[name],old['model'][name]), name
records = [json.loads(line) for line in (root/'reference/control_diagnostics.jsonl').read_text().splitlines()]
obs = torch.tensor([r['observation_after'] for r in records])
assert obs.shape == (600, 19)
def infer(state):
    x = torch.clamp((obs-state['running_mean_std.running_mean']) / torch.sqrt(state['running_mean_std.running_var']+1e-5), -5.,5.).float()
    for i in (0,2,4):
        x = F.elu(F.linear(x,state[f'a2c_network.actor_mlp.{i}.weight'],state[f'a2c_network.actor_mlp.{i}.bias']))
    return F.linear(x,state[weight],state[bias]), F.linear(x,state['a2c_network.sigma.weight'],state['a2c_network.sigma.bias'])
before, before_sigma = infer(old['model'])
after, after_sigma = infer(model)
assert torch.equal(before[:,[0,1,3,4,5]],after[:,[0,1,3,4,5]])
assert torch.equal(before_sigma,after_sigma)
assert torch.equal(after[:,2],torch.full((600,),-0.5))
mapping=[]
for seed in sorted({r['seed'] for r in records}):
    selected=[r['substeps'][0] for r in records if r['seed']==seed and not r['substeps'][0]['success_latched'][0]]
    # Invert the old bounded target into a relative filtered command at the
    # identical pre-physics pose. This audit cannot remove old saturation.
    relative=[(s['command_target'][2]-s['tool_position'][2])/.003 for s in selected]
    mapping.append({'seed':seed,'unlatched_controls':len(relative),'relative_command_min':min(relative),'relative_command_max':max(relative),'saturated_down_controls':sum(x<=-.999 for x in relative),'degenerate_endpoints_controls':sum(abs(s['legal_z_endpoint_commands'][1]-s['legal_z_endpoint_commands'][0])<1e-7 for s in selected)})
assert old['epoch']==new['epoch']==113 and old['frame']==new['frame']
assert not torch.cuda.is_initialized()
torch.save(new,destination)
report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'migrated_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'changed_model_rows':{weight:2,bias:2},'paired_recorded_observations':600,'other_mean_outputs_bitwise_equal':True,'all_sigma_outputs_bitwise_equal':True,'new_mean_z':-.5,'old_z_mean_range':[before[:,2].min().item(),before[:,2].max().item()],'inverted_legacy_mapping':mapping,'optimizer_Z_mu_moments_cleared':True,'training_samples_added':0,'limits':['Paired observations are recorded legacy states, not new closed-loop states.','Z head initialization changes the policy. Closed-loop XY can change as states and previous-Z observations change.','Sigma network and shared Adam step retained; optimizer migration is partial, not fresh training.']}
(root/'calibration_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
