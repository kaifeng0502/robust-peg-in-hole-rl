# AutoDL integration replication — 2026-10-03

All three methods completed the same eight frozen integration cases on an NVIDIA GeForce RTX 4090 D. The paired comparison passed on AutoDL and independently on the local machine. Recorded initial-state components were exactly equal across the three methods and matched the historical RunPod resets.

This is a replication of the existing integration cases, not eight additional independent test cases. Do not pool the two platforms' results to increase the statistical sample size. The 64-case development set and 500-case holdout remain unused at this checkpoint.

## Results

Each episode ran 150 control intervals / 10 seconds at 120 Hz physics and 15 Hz policy control. Success requires valid insertion geometry throughout the final continuous one second, sampled at every physics boundary.

| Method | Held successes | Wilson 95% interval | Mean task episode reward |
|---|---:|---:|---:|
| Zero residual | 2/8 (25.0%) | 7.1–59.1% | 487.9222 |
| Spiral search | 4/8 (50.0%) | 21.5–78.5% | 496.1660 |
| PPO | 3/8 (37.5%) | 13.7–69.4% | 539.1577 |

PPO successes were cases 1, 2 and 6. Case 7 first reached valid geometry at 9.8 seconds and held it for only 0.2 seconds, correctly failing the terminal-hold criterion. The other four PPO failures were classified as misalignment. Higher task reward again did not produce more held successes than spiral search. This small integration set establishes neither a PPO advantage nor the 95% target.

## Replication checks

- All 24 episodes completed; all 3,600 trace records reconcile with their saved per-case metrics and summaries.
- The comparison checks complete coverage, runtime, protocol, source fingerprints and realized initial states. Its existing tolerance was retained; the observed maximum initial-state difference was zero.
- For zero residual and spiral, `summary.json`, `trials.json` and `trajectories.jsonl` are byte-identical to the corresponding [RunPod evidence](../gpu_integration_20261002/README.md).
- PPO has the same per-case success/failure decisions but is not numerically identical. Mean task reward changed from 542.9137 to 539.1577; mean completion time over its three successful cases changed from 4.4889 to 4.5778 seconds.
- PPO first-step actions differ by a maximum of approximately 1.49e-8 to 5.96e-8 per case despite identical initial observations and checkpoint/configuration hashes. Recorded geometry first diverges within steps 1–6. The cause is not isolated: the GPU changed and the original RL-Games source commit is unavailable.

Do not rank controller speed from averages over different successful-case subsets. Force fields are commanded controller wrench magnitudes, not measured contact forces.

## Runtime and validation

The runtime is Ubuntu 22.04.5, driver 595.71.05, Python 3.11.13, PyTorch 2.7.0+cu128, Isaac Sim 5.0.0.0, pinned Isaac Lab 2.2.1, and RL-Games 1.6.1. Source identities, artifact hashes, environment activation and dependency constraints are described in [the AutoDL runbook](../../docs/AUTODL.md).

Dependency consistency passed. All 79 regression tests passed, three fresh-process CUDA computation checks passed, and the four-environment / 12-step simulator smoke check passed. A further fresh CUDA process and NVML query succeeded after the complete evaluation; the evaluation processes had exited. No GPU access loss was observed during these checks. This does not establish indefinite host reliability.

The epoch-93 checkpoint was retained without retraining. The original training `env.yaml` / `env.pkl` and original RL-Games commit remain unavailable; the saved agent YAML does not replace the missing training environment snapshot. No claim of a fully reconstructed historical training environment is made.

## Artifacts

Each method directory contains provenance, trials, summary and full trajectories. `comparison_smoke.json` is the accepted AutoDL comparison. `runtime/` retains the complete package snapshot, RL-Games source/wheel provenance, GPU version, CUDA checks, smoke result and regression log. The setup and comparison exit code is zero.

Next performance work uses development cases before freezing a checkpoint for the final holdout. Multi-seed training, randomization ablations, ROS 2 / MoveIt 2 and recovery remain subsequent stages.
