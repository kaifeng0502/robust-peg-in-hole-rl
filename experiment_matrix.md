# Evaluation matrix

This is the experiment protocol, not a completed result table. First validate the unified evaluator on the GPU using the eight committed integration cases. Then freeze the checkpoint and evaluate the 500 cases in `configs/evaluation_holdout.json`, verifying the realized initial state for each paired case.

The primary criterion is XY error ≤2.5 mm and absolute depth error ≤1 mm, maintained throughout the final continuous 1 s of a common 10 s horizon. All three methods use the same `LocalInsertionRLEnv`, pre-insertion initialization, gains, action units, hold logic and final position limits. See [docs/EVALUATION.md](docs/EVALUATION.md).

## Main control comparison

| Method | Learned residual | Purpose |
|---|---|---|
| Impedance + spiral | none | Scripted search reference |
| Impedance + nominal insertion | zero | Isolate the contribution of the common target and hold controller |
| Impedance + PPO | trained | Measure the contribution of learned local adjustment |

The legacy standalone spiral results used different initialization, horizon and success semantics. Re-evaluate every method with the common evaluator before reporting improvement.

## PPO randomization study

| Method | Pose randomization in training | Friction randomization in training | Evaluation distribution |
|---|---:|---:|---|
| PPO nominal | off | off | pose + friction |
| PPO pose only | on | off | pose + friction |
| PPO friction only | off | on | pose + friction |
| PPO full | on | on | pose + friction |

Report these task metrics for every method:

- terminal-hold success rate with a Wilson 95% interval, with ever/final success shown separately;
- mean and median completion time among successful trials;
- maximum insertion depth sampled at the control rate;
- peak commanded force (clearly labeled as a command, until a real contact sensor is added).

Report these learning metrics for every PPO row:

- environment steps to 50%, 70%, and 80% held-out success;
- evaluation success versus environment steps;
- mean and standard deviation across seeds 42, 43, and 44;
- task return, regularized training return, and critic loss curves for diagnosing instability;
- wall-clock training time and total transitions, including every earlier curriculum stage.

Use distinct development and final-test manifests. Each frozen PPO checkpoint is compared against both non-learned controls on the same final cases. Across training seeds 42, 43 and 44, report variation in final evaluation success separately from within-run binomial intervals. An unachieved sample-efficiency threshold is reported as not reached within budget.

Reaching at least 95% terminal-hold success on the declared distribution is an engineering target, not a measured result or a guarantee. Larger pose errors, unseen friction intervals, and observation delays belong to a later, separately reported generalization suite.
