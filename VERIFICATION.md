# RunPod verification record

Verified on 2026-09-17 with NVIDIA L4 (23,034 MiB), Isaac Lab 2.2.1 commit `0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20`, Isaac Sim 5.0, physics at 120 Hz, and policy control at 15 Hz.

## Dynamic checks

Both registered environments passed a four-environment, 12-step dynamic smoke test. Policy observations had shape `(4, 19)`, observations and rewards stayed finite, and sampled friction values stayed inside the configured 0.30–1.20 interval.

RL-Games completed a one-epoch training smoke test with 32 environments and wrote a 1.29 MB checkpoint. The full environment, PPO update, TensorBoard logging, and checkpoint pipeline are therefore operational.

## Fixed-search calibration

All rows contain 128 trials at seed 42 with identical controller settings.

| Distribution | Hole XY observation noise | Success | Mean successful time | Mean peak commanded force |
|---|---:|---:|---:|---:|
| Easy calibration | 1.0 mm std. | 96.1% (123/128) | 2.87 s | 1.33 N |
| Challenge v1 | 2.5 mm std. | 50.0% (64/128) | 4.00 s | 2.33 N |
| Challenge v2, selected | 2.0 mm std. | 64.8% (83/128) | 3.92 s | 2.18 N |

Challenge v2 is the selected training and evaluation distribution. In its successful trials, the mean hole XY observation error was 1.86 mm; in failures it was 3.33 mm. Initial tilt and friction were similar across both groups, so the selected comparison primarily measures recovery from localization error while retaining friction and pose variation.

The force values above are impedance-controller wrench commands, not sensor-measured contact forces.

## First pilot finding

The initial seed-42 pilot stopped at epoch 25 because its cumulative reward crossed an incorrectly low `score_to_win` threshold. Its true insertion success remained 0%, and all depth, engagement, and success reward terms remained zero. The policy had only increased its XY alignment reward. This checkpoint is retained as evidence of a reward-design failure and must not be reported as a successful policy.

The corrected configuration disables reward-based early stopping and adds the upstream Factory multi-scale keypoint reward plus an alignment-gated vertical approach reward. This supplies dense XY, Z, and orientation feedback before contact while keeping insertion depth and task success as separate metrics.

A second diagnostic run showed that the RL task still inherited Factory's 47 mm hand offset, leaving the peg base about 15 mm above the socket. That setup makes PPO learn the global free-space approach again. The RL task now starts at a separate 35 mm hand offset, placing the peg base about 3 mm above the entrance; the fixed-search baseline keeps its original scripted approach. This matches the intended MoveIt2-to-pre-insertion plus local-RL architecture.

That pre-insertion diagnostic still produced no positive depth in ten epochs because zero-mean exploration had to discover sustained downward motion while simultaneously aligning the peg. The local controller is therefore formulated as residual RL: impedance control contributes a contact-seeking Z preload while PPO learns XY, Z, and orientation recovery.

An initial normalized Z bias was too weak: in a zero-action nominal test it produced no insertion because the spring component was only about 0.015 N. A 3 mm current-relative preload then reached only 1.1 mm depth because zero policy action did not hold an absolute XY target. The controller now uses the observed hole pose and nominal grasp to define an absolute insertion target, with PPO producing bounded 6D residuals around it. Instantaneous impedance error remains force-limited as in the baseline.

Cloud paths:

- project: `/workspace/projects/baseline_rl`
- fixed-search results: `/workspace/results/baseline_rl/spiral_challenge_v2_seed42`
- pilot log: `/workspace/results/baseline_rl/train_pilot_seed42.log`
- TensorBoard run: `/workspace/projects/baseline_rl/logs/rl_games/LocalInsertion/2026-09-17_03-11-11`

## Curriculum training result

The corrected absolute-residual controller was trained with seed 42 and 128 parallel environments. The curriculum was continued from nominal to medium, intermediate, and then full pose-plus-friction randomization:

| Stage | Environment steps | Final/peak success in training | Run |
|---|---:|---:|---|
| Nominal, no pose or friction randomization | 393,216 | 100% nominal smoke success | `06-49-54` |
| Medium randomization | 573,440 additional | 96.9% | `07-10-53` |
| Intermediate randomization | 245,760 additional | 98.4% peak, 96.9% final | `07-27-18` |
| Full randomization, resumed from intermediate | 1,130,496 total in resumed run | 82.0% peak/final batch | `07-47-07` |

The full stage used the configured pose and friction randomization together: hole and grasp pose variation, 2 mm XY observation noise, and friction sampled from 0.30–1.20. Its TensorBoard success series was `[71.1%, 68.0%, 66.4%, 69.5%, 72.7%, 71.9%, 71.9%, 64.1%, 66.4%, 66.4%, 74.2%, 79.7%, 75.8%, 75.8%, 75.8%, 81.3%, 75.0%, 70.3%, 82.0%]`; the best checkpoint is `/workspace/projects/baseline_rl/logs/rl_games/LocalInsertion/2026-09-17_07-47-07/nn/LocalInsertion.pth`. This is a training-seed result, not a multi-seed generalization claim.

The fixed-search challenge-v2 baseline is 64.8% (83/128), so the full randomized RL training run exceeds that reference in its held-in training distribution. A separate `play_rl.py` inference launch loaded the checkpoint and constructed the 128-environment policy successfully, but Isaac Sim later hit a known headless Vulkan/Carb mutex assertion before a long independent rollout completed; the 82.0% figure above therefore comes from the training episode success metric.

## Independent deterministic evaluation

The first finite-horizon deterministic evaluation exposed a serious gap between the training scalar and deployable policy behavior. Using the full-randomization checkpoint from `07-47-07`, 128 environments, 149 policy steps (one 10 s episode without triggering the reset), and fresh seeds produced:

| Evaluation seed | Success | Final insertion result |
|---:|---:|---|
| 42 | 0/128 (0.0%) | No episode reached the success geometry |
| 123 | 0/128 (0.0%) | No episode reached the success geometry |

The earlier 62/128 and 59/128 counts came from the cumulative `ep_succeeded` buffer before the evaluation script was corrected; they are not valid success measurements. The corrected metric recomputes the Factory success predicate from the final geometry. This means the 82.0% TensorBoard series must not be treated as deployment success yet. The next engineering step is to trace the deterministic action output and reward/termination timing, then fix the evaluation and policy pipeline before changing the reward or curriculum.

## Success-hold correction

The trace showed that the policy could enter the success geometry briefly and then pull the peg back out. The RL environment now latches the first successful world pose, commands that stored pose on subsequent steps, and adds a 1.5 mm downward hold margin. A nominal deterministic check changed from 0/1 to 1/1 with 24.1 mm final depth.

The full-randomization continuation was launched from the `07-47-07` checkpoint, but its final artifacts were not preserved after the compute instance stopped. No terminal success claim is made from that continuation. A future full-randomization result must be generated with the corrected final-geometry evaluator and retained alongside its checkpoint and evaluation seeds.
