# Calibrated PPO continuation: 64-step rollouts, 500 new iterations

Status: completed and independently audited. The previous calibration and
eight-rollout screen are documented in `../z_transfer_20261003`. This
experiment continued that calibrated epoch-121 model; it did not reinitialize
Z again.

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

The final checkpoint is absolute epoch 621, frame 35,667,968, SHA-256
`b4b9e428af688f73c1c845d5b4088e7f6ea54ce6fddb86f7b08386f1ba347524`.
CPU-only audit loaded all six saved checkpoints (epochs 200, 300, 400, 500,
600 and 621), checked frames, finite weights, model shapes, source checkpoint,
saved configuration and TensorBoard. Each of the 500 logical new rollouts
contains 1,024 × 64 = 65,536 transitions; retained new transitions total
32,768,000. Post-resume median logged total throughput was 2,104
transitions/s. The full elapsed wall clock includes a user-requested pause
and is not an active-training speed measurement.

The training TensorBoard series ended with a 98.05% terminal-held-success
fraction (last ten reported points averaged 98.37%). This describes sampled
training episodes, not performance on held-out initial states. The fixed
epoch-621 checkpoint was evaluated only on the four previously selected
diagnostic cases. It held success on **2/4**. The matched epoch-300 checkpoint
had **3/4**: two cases remained successful, case `000010` remained misaligned,
and case `000028` regressed from a 24.92 mm insertion and 1 s hold to a 5.15
mm final XY error with no insertion. There were zero paired gains and one
paired loss. All four initial states matched exactly; each replay had 150
control intervals and 1,200 physical substeps. Successful cases met the
final 15-interval continuous-hold criterion. These four reused cases cannot estimate generalization or
support a 95% success claim. The final 500-case holdout remains sealed.

The full training backup, TensorBoard file, two log segments and both raw
four-case replays are in `results/` archives. `training_audit.json` and
`evaluation_audit.json` record the independent checks; `audit_training.py`
and `audit_results.py` reproduce them after extraction into the same
directory. `files_sha256.json` verifies the published files, while
`remote_artifact_sha256.txt` records the 22 remote file hashes, all of which
matched the local backup. The force field is
commanded impedance wrench, not sensor-measured contact force. The epoch-621
checkpoint is retained as an experimental result; its four-case regression
does not justify replacing the epoch-300 checkpoint or expanding training
without a separate decision.

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
