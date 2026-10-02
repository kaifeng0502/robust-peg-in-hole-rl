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

The legacy fixed-search challenge-v2 rate is 64.8% (83/128), using a cumulative success buffer. It is not a fair terminal-hold comparison with PPO: metric semantics, start height, and horizon differed. A separate `play_rl.py` inference launch loaded the checkpoint and constructed the 128-environment policy successfully, but Isaac Sim later hit a headless Vulkan/Carb mutex assertion before that independent rollout completed. The 82.0% figure above comes from the training episode success metric, not that playback run.

## Independent deterministic evaluation

The first finite-horizon deterministic evaluation exposed a serious gap between the training scalar and deployable policy behavior. Using the full-randomization checkpoint from `07-47-07`, 128 environments, 149 policy steps (one 10 s episode without triggering the reset), and fresh seeds produced:

| Evaluation seed | Success | Final insertion result |
|---:|---:|---|
| 42 | 0/128 (0.0%) | No episode retained success geometry at the final evaluated sample |
| 123 | 0/128 (0.0%) | No episode retained success geometry at the final evaluated sample |

The earlier 62/128 and 59/128 counts came from the cumulative `ep_succeeded` buffer before the evaluation script was corrected; they describe ever-success and are not terminal success measurements. The corrected diagnostic recomputed the Factory predicate from final geometry. The difference from the 82.0% TensorBoard series remains unresolved; deterministic vs stochastic action selection, normalization, horizon, and reset timing must be verified before attributing it to a single cause.

On 2026-10-02, review of `_log_factory_metrics` at the pinned Isaac Lab commit confirmed that `extras["successes"]` logs **current geometry when the training episode times out**, while `ep_succeeded` separately accumulates whether success ever occurred. Earlier documentation incorrectly assumed the 82.0% scalar necessarily used the latter buffer. The original TensorBoard tag-to-logger path still needs to be checked alongside preserved run artifacts. None of these historical scalars proves one-second terminal-hold success.

## Success-hold correction

The trace showed that the policy could enter the success geometry briefly and then pull the peg back out. The RL environment now latches the first successful world pose, commands that stored pose on subsequent steps, and adds a 1.5 mm downward hold margin. A nominal deterministic check changed from 0/1 to 1/1 with 24.1 mm final depth.

The full-randomization continuation was launched from the `07-47-07` checkpoint. Its artifacts were recovered from persistent storage on 2026-10-02 after the Pod restarted. The earlier connection failure did not indicate loss of the saved run.

## Recovered continuation — 2026-10-02

The restarted Pod reports an NVIDIA L4 with driver `570.195.03`. The recovered run directory is:

```text
/workspace/projects/baseline_rl/logs/rl_games/LocalInsertion/2026-09-17_11-39-10
```

The preserved training log confirms epoch `100/100`, logged frames `1622016`, reward `898.63434`, and the saved final checkpoint `nn/last_LocalInsertion_ep_100_rew_898.63434.pth`.

Reading the original best training-reward checkpoint, `nn/LocalInsertion.pth`, recovered these fields:

| Checkpoint field | Value |
|---|---:|
| `epoch` | 93 |
| `frame` | 1523712 |
| `last_mean_rewards` | 953.5416 |

That original `LocalInsertion.pth` is selected for the upcoming integration check and unified evaluation **before observing results on the new cases**. The selection uses the saved training-reward checkpoint, not new evaluation outcomes. The checkpoint and original agent configuration will be hashed in evaluation metadata.

The selected checkpoint SHA256 is `9811f2bfa13dab366531324fbab5bb758a7e031753a6378b5d318ad99d843721`; its original agent configuration SHA256 is `10cfe0b4e3974952b02b565054a47ebcc2cf68a5ff21ca5fa316f448f7dfbe7e`.

These records establish that the continuation completed and its weights survived. They do not establish deterministic inference correctness or terminal-hold success. The pinned Isaac Sim 5.0 extension-cache packages were restored after a workspace-quota interruption. The subsequent simulation attempt stopped before evaluating any cases because the container lost GPU access: `nvidia-smi` reported `Failed to initialize NVML: Unknown Error`, PyTorch reported zero CUDA devices, and opening `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm` returned `EPERM`, despite the device nodes being present. The same session had successfully detected the L4 earlier.

These observations are consistent with the container-update GPU-access failure documented by [NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error); the precise host-side trigger was not established. Container GPU access must be restored before simulation integration can proceed. No new manipulation results have been produced.

## Unified evaluation revision — 2026-10-02

Implemented locally:

- A common `LocalInsertionRLEnv` for spiral, zero-residual, and PPO evaluation, with the same pre-insertion pose distribution, nominal grasp offset, reward, control gains, and hold logic.
- Position-error clipping after the hold override, closing the previous unbounded-hold target path.
- A 10 s finite-horizon evaluator with a guarded simulator timeout, deterministic PPO inference, original training configuration, and restored policy normalizers.
- Success checks at every 120 Hz physics boundary. The headline criterion requires the final 1 s to remain inside XY ≤2.5 mm and absolute depth error ≤1 mm. The two-sided depth check also rejects excessive penetration.
- Recorded initial states, frozen reset manifests, checkpoint/source/configuration identities, per-step traces, and rejection of incomplete or mismatched comparisons.
- Dependency-free tests for metric semantics, reset contamination, manifest integrity, and comparison validation.

Local tests and source checks do not execute Isaac Sim or validate policy inference. Checkpoint metadata has now been recovered, but GPU integration, physical-state repeatability, checkpoint inference, and all new success-rate measurements remain pending restoration of GPU device access. The new shared spiral implementation is a new benchmark controller, not a remeasurement of the historical 83/128 result.

Local validation on 2026-10-02: all 68 unit/regression tests passed, Python compilation passed, and Ruff reported no issues. The test suite uses deterministic synthetic backends for lifecycle and metric checks; those outcomes are not manipulation experiment results.

Pod validation at `2df57de`: all 69 tests passed with PyTorch installed, including two consecutive reset/step cycles through the actual simulation adapter with a tensor-backed test environment. This regression verifies that persistent buffers remain mutable after stepping; it does not execute Isaac Sim physics or prove GPU inference.

After the device-access failure, a CUDA preflight was added before simulator startup. Ten mocked tests cover missing CUDA access, invalid device indices, query/allocation errors and preflight ordering. The resulting local suite ran 79 tests: 78 passed and the PyTorch-dependent adapter test was skipped because local PyTorch is unavailable. Ruff and compilation passed. The preflight is an early diagnostic, not a repair for host-managed device permissions.
