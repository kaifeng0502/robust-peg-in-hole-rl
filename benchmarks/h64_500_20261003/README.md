# Calibrated PPO continuation: 64-step rollouts, 500 new iterations

Status: launched; results pending. The previous calibration and eight-rollout
screen are documented in `../z_transfer_20261003`. This experiment continues
that calibrated epoch-121 model; it does not reinitialize Z again.

| Setting | Value |
|---|---|
| Parallel environments | 1,024 |
| Rollout length per environment | 64 control intervals |
| New rollout iterations | 500 |
| Epoch range | 121 to 621 |
| New transitions | 32,768,000 |
| Checkpoint frame range | 2,899,968 to 35,667,968 |
| Minibatch size / passes | 2,048 / 4 |
| Seed | 42 |
| Action contract | relative_z_v1 |
| Dense / success reward | legacy / terminal_hold_v2 |
| Save frequency | every 100 absolute epochs, plus final 621 |

Model, optimizer and normalizers are restored from the verified checkpoint
whose SHA is in `plan.json`. Physics remains 1/120 s with decimation eight,
192 position-solver iterations, 150 scored intervals per 10-second episode and
a strict continuous one-second hold. Sampling 64 steps does not truncate or
reset the episode. Nonterminal rollout ends bootstrap from the value network.
The unchanged reward and action interface isolate this as continued learning
with a different rollout schedule, not a new reward comparison.

The current `run_training_save100.py` checks the starting checkpoint, source fingerprint, saved
runtime configuration and final sample budget. It does not stop for short-term
reward fluctuations, add budget, or launch intermediate evaluations. A 12-hour
watchdog bounds a stuck run; process/configuration failures retain evidence.
The final epoch-621 checkpoint is selected regardless of its training reward.
After backing up models, TensorBoard and configurations, the supervisor checks
only the four existing diagnostic cases. No 64-case evaluation is queued; the
final 500-case holdout remains sealed. These four cases cannot establish a
generalization success rate. The paid instance is not shut down automatically.

The stage is isolated from all preceding experiments. Actual model provenance,
training throughput, curve interpretation and final behavior require completion
and independent audit; launch alone is not validation of the outcome.

## User-requested saving-frequency change

The running RL-Games agent reads its saving interval only at initialization.
Following the user's request for every 100 epochs, the original run was paused,
its latest complete checkpoint at epoch 133 was verified and copied, and both
old processes were stopped. An isolated `source_save100` checkout with identical
source fingerprint resumes from epoch 133. `plan_save100.json` and
`run_training_save100.py` supersede the initial plan/supervisor for execution;
the originals remain historical provenance. Current status is
`h64_500_20261003/run_save100/status.json`.

The saved runtime configuration verifies `save_frequency=100` and
`save_best_after=1000000000`, disabling extra writes on reward improvements.
Periodic saves use absolute epochs 200, 300, 400, 500 and 600; the final 621
checkpoint is also saved. This changes checkpoint scheduling, not policy or
physics calculations. There are 488 logical rollouts remaining after the
verified epoch-133 checkpoint, adding 31,981,568 retained transitions. The
original 500-rollout target and final frame 35,667,968 remain unchanged.

The original log already recorded epoch 134, but its weights were not saved.
One complete rollout (65,536 transitions) must therefore be recomputed, plus
possibly up to 65,536 transitions from the interrupted partial rollout. This
restart cost is recorded separately: actual executed samples can exceed the
32,768,000 retained logical samples by 65,536–131,072. Checkpoint frame deltas
must not be presented as all computation spent. `recovery_save100.json`
records the preserved checkpoint SHA, frames and discarded-work bounds.
The checkpoint restore retains policy, optimizer and normalizers; simulator
state is reset on process startup, so this is not a bitwise uninterrupted run.
