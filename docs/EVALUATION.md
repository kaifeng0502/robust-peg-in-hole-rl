# Unified insertion evaluation

## Status and scope

This revision implements the common evaluator and regression tests. The GPU instance restarted on 2026-10-02 with an NVIDIA L4 and driver `570.195.03`; the preserved continuation run and its checkpoints were recovered, and 69 tests passed on the Pod. Isaac Sim execution and checkpoint inference have **not** yet been validated for this revision. Extension installation completed, but the container then reported NVML failure and denied access to its GPU device nodes. GPU access must be restored before proceeding. The commands below are the next integration checks, not completed experiments.

The reference task is local insertion under the existing `challenge_v2` pose and friction distribution. All three methods use `LocalInsertionRLEnv` with identical initialization, gains, observation inputs, nominal grasp geometry, position bounds, and success-pose hold:

| Method | Target generator |
|---|---|
| `spiral` | Scripted approach, Archimedean search, then insertion |
| `zero_residual` | Nominal insertion target with all policy residuals set to zero |
| `ppo` | The same nominal target plus deterministic PPO residuals |

All methods start at the same local pre-insertion distribution and receive the same 10 s budget. The spiral's approach consumes part of that budget. The old standalone spiral evaluator is retained solely for reproducing historical calibration; its rates and times are not part of this comparison.

## Success contract

- Physics timestep: 1/120 s; one control interval contains eight physics steps.
- Scored horizon: exactly 150 control intervals (10 s).
- XY error: at most 2.5 mm.
- Depth error: absolute difference from the geometric socket-bottom target at most 1 mm. Excessive penetration does not count as success.
- Headline success: valid geometry throughout the **final continuous 1 s**, including every physics boundary in each of the last 15 control intervals.

This is a discrete simulation criterion at 120 Hz. It does not establish continuous real-world contact or grasp stability beyond the measured geometry.

The report separately records:

| Field | Meaning |
|---|---|
| `ever_success` | Any valid geometry during the scored episode |
| `final_success` | Valid geometry at the final sample |
| `had_hold_anytime` | A qualifying hold occurred at some point |
| `held_success` / `success` | A qualifying hold is still active at the horizon |
| `success_time_s` | Time when the final uninterrupted successful run first met the hold requirement; null for unsuccessful trials |

A peg that inserts and subsequently withdraws is a failed terminal-hold trial, even if it previously held for one second. Separate success intervals never add up to a hold. A peg that only reaches the target at the final instant is also unsuccessful under this contract.

## Reset and state integrity

The reference evaluator deliberately uses one environment. A frozen manifest identifies each case by an independent reset seed, so changing parallel batch size cannot silently alter the benchmark.

Seeds alone do not prove identical physical starting states. Every run records robot joint/root positions and velocities, peg/socket root states, sampled and applied material properties, target observation noise, fingertip state, gains, and initial policy observations. The comparison requires identical case identities, reset seeds, keys/shapes and matching finite numeric state with absolute tolerance `1e-6` and zero relative tolerance.

This verifies the recorded state, not hidden PhysX solver/contact caches. It is a checked seeded-reset protocol, not a claim of bitwise simulator replay. If states differ, comparison stops. Inspect the mismatches and fix reset repeatability before producing a headline comparison; do not loosen tolerance simply to obtain a table.

Factory automatically resets at its timeout. The evaluator reserves two extra simulator control steps but scores only the requested 150. Every sample must have no termination/truncation flag and must increment the episode counter exactly once. Unexpected resets abort the run before scoring reset observations. Failed or partial runs retain `run_complete: false` and cannot be compared.

## Reproduce on the GPU instance

Use the configured Isaac Lab 2.2.1 / Isaac Sim 5.0 Python environment and the same source tree for all methods. Preserve a trained checkpoint and its **original** `params/agent.yaml`; a YAML with only a matching network shape is insufficient evidence of equivalent preprocessing.

The evaluator checks CUDA availability, the requested device index, and a small tensor allocation before starting Isaac Sim. If this check fails, restore container GPU access first; changing task code or interpreting a startup failure as an unsuccessful insertion would be incorrect. CPU device requests bypass this CUDA check, but CPU task execution has not been validated.

On the recovered Pod, the three `isaacsim-extscache-*==5.0.0.0` packages were reinstalled successfully. Their 446 extension directories were copied to `/workspace/runtime/insertion_isaacsim_extscache_5_0`, and the environment's `isaacsim/extscache` link now resolves to that persistent location. This prevents the restored extensions from depending on the temporary container disk after restart. The L4 must still pass the CUDA check before any benchmark resumes.

The recovered run is `/workspace/projects/baseline_rl/logs/rl_games/LocalInsertion/2026-09-17_11-39-10`. Its training log confirms completion at epoch 100/100 with logged frames `1622016` and reward `898.63434`, saving `nn/last_LocalInsertion_ep_100_rew_898.63434.pth`. The original best training-reward checkpoint, `nn/LocalInsertion.pth`, records epoch `93`, frame `1523712`, and `last_mean_rewards` of `953.5416`.

Use that original best training-reward checkpoint for the upcoming integration check and evaluation. This choice was made before observing the new cases' outcomes. Retain its checkpoint/configuration hashes in each run; training reward and completed epochs are provenance evidence, not success-rate evidence.

Start with the committed eight-case integration manifest:

```bash
python scripts/evaluate_insertion.py \
  --method spiral --cases configs/evaluation_smoke.json --headless \
  --output /workspace/results/evaluation_v1/spiral_smoke

python scripts/evaluate_insertion.py \
  --method zero_residual --cases configs/evaluation_smoke.json --headless \
  --output /workspace/results/evaluation_v1/zero_smoke

python scripts/evaluate_insertion.py \
  --method ppo --cases configs/evaluation_smoke.json --headless \
  --checkpoint /workspace/projects/baseline_rl/logs/rl_games/LocalInsertion/2026-09-17_11-39-10/nn/LocalInsertion.pth \
  --agent-config /workspace/projects/baseline_rl/logs/rl_games/LocalInsertion/2026-09-17_11-39-10/params/agent.yaml \
  --output /workspace/results/evaluation_v1/ppo_smoke

python scripts/compare_evaluations.py \
  --runs /workspace/results/evaluation_v1/spiral_smoke \
         /workspace/results/evaluation_v1/zero_smoke \
         /workspace/results/evaluation_v1/ppo_smoke \
  --output /workspace/results/evaluation_v1/comparison_smoke.json
```

The paths to the checkpoint and agent YAML must identify the actual retained training run. The evaluator restores the RL-Games model and input normalizer, sets evaluation mode, initializes batched/recurrent state, and uses deterministic actions. The expected action interface is six clipped residuals in `[-1, 1]`.

After integration and tuning on development cases, freeze the checkpoint and use `configs/evaluation_holdout.json` for the 500-case comparison, changing output directory names accordingly. Do not tune on the final holdout and still describe it as unseen evaluation. For further tuning, create a separate development manifest with a different seed:

```bash
python scripts/make_evaluation_cases.py \
  --trials 64 --seed 20261003 --output results/development_cases.json
```

All output paths must be new. A failed output directory remains available for inspection; use a new directory for the next attempt.

## Artifacts and interpretation

Each completed evaluation produces:

- `metadata.json`: complete case manifest, protocol and source identity, package/upstream source versions, checkpoint/configuration hashes, hold criteria, and completion status.
- `trials.json` and `trials.jsonl`: initial-state snapshot and metrics for every case. JSONL is flushed after each completed episode for crash recovery.
- `trajectories.jsonl`: per-control-step actions, geometry, rewards, interval success flags, and commanded-force peaks.
- `summary.json`: held-success rate and Wilson 95% interval, other success definitions, successful completion times, rewards, depths and failure categories.
- `restored_agent_config.json` for PPO: the effective agent configuration after runtime device and actor-count overrides.

Force magnitude is sampled after every generated physics-step command, and each control sample records the maximum over its eight physics substeps. This is **commanded force**, not measured contact force. Depth traces and maximum-depth summaries are sampled at the 15 Hz control rate.

`episode_reward` includes the existing residual-action and action-rate penalties. `task_episode_reward` removes those two penalties, retaining common task progress, geometry and commanded-wrench terms. The scripted controllers issue zero residual actions despite generating moving targets; therefore regularized episode returns alone must not rank control quality.

Wilson intervals describe the proportion over evaluated cases. Variation across independently trained PPO seeds is a different quantity and must be reported separately. Synthetic regression-test outcomes are never simulation success-rate evidence.

The comparison checks complete coverage, protocol/configuration identity, runtime identity, source identity and matched initial states. It never compares an incomplete run with a completed one or silently drops failed cases. Failure categories describe observed terminal geometry/hold behavior; they do not diagnose physical jamming or sensor failure.

## Deployment boundary and subsequent milestones

The current engagement and success-hold decisions use simulator truth. That assumption is shared and explicitly recorded for this algorithm comparison. Ground-truth rewards/evaluation and online controller observations serve different purposes; a future deployment controller needs observable contact/completion feedback.

After this evaluation is validated:

1. Establish the PPO vs zero-residual vs spiral result and inspect failures; then run training-seed and pose/friction ablations.
2. Replace online truth-based decisions with observable feedback and validate command limits, completion detection and failure detection.
3. Integrate ROS 2 / MoveIt 2 pre-insertion planning, controller handover, bounded retreat/retry and event logs. Report first-attempt, recovery and final system success separately.
4. Freeze in-distribution results before testing larger pose errors, unseen friction intervals and delayed observations as separate generalization sets.

PPO/SAC/TD3 algorithm comparisons are outside this revision.

## Local verification

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q local_insertion scripts tests
ruff check .
```

Tests cover transient success, hold loss/recovery, excessive depth, physical-interval flags, timeout/reset contamination, case provenance, incomplete runs and invalid numeric data. Simulator adapters still require the GPU integration run above.
