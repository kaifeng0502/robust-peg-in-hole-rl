# Matched GPU integration run — 2026-10-02

This is an eight-case integration check, not the 500-case final evaluation. Each method used the same frozen seeds, 150 control intervals, and 10-second horizon. Success requires valid geometry throughout the final continuous second, sampled at every 120 Hz physics boundary.

| Method | Held successes | Wilson 95% interval | Successful case suffixes |
|---|---:|---:|---|
| Zero residual | 2/8 (25.0%) | 7.1–59.1% | 2, 6 |
| Spiral search | 4/8 (50.0%) | 21.5–78.5% | 0, 2, 6, 7 |
| PPO | 3/8 (37.5%) | 13.7–69.4% | 1, 2, 6 |

All 24 episodes completed. The comparison gate accepted protocol, source, runtime, case coverage, and initial-state identity. Every recorded initial-state component was exactly equal across methods; this does not establish identity of hidden simulator caches. Replaying all 3,600 control-step records through the metric implementation reproduces the saved episode and summary metrics within floating-point roundoff.

The original best training-reward checkpoint was selected before these cases were evaluated. Its SHA256 is `9811f2bfa13dab366531324fbab5bb758a7e031753a6378b5d318ad99d843721`. The simulator ran source commit `089ccae435a00d3b4bd84c90c969e3cf6eae7374`; full configuration, upstream source hashes, runtime versions and checkpoint provenance are in each method's metadata.

## Interpretation

PPO succeeds on case 1 where both scripted methods fail. Spiral succeeds on cases 0 and 7 where PPO fails the hold criterion. Cases 3, 4 and 5 fail under all methods. This integration sample does not establish a reliable PPO advantage or support a 95% success claim.

PPO case 7 first reaches valid geometry at 9.8 seconds and holds for only 0.2 seconds. Its final XY error is about 0.481 mm and depth 24.655 mm, but its insufficient hold correctly makes the episode unsuccessful. PPO cases 0, 3, 4 and 5 remain near the socket entrance, with maximum sampled depths between −0.023 and 0.030 mm and final XY errors between 3.12 and 4.64 mm.

In those four PPO failures, the logged roll action reaches its clip bound in 35–39 of 150 intervals, and yaw in 25–30 intervals. These are policy inputs before action filtering and target limits, not measured rotations. Effective targets and contact measurements are needed before assigning a mechanical cause.

Mean task return ranks PPO (542.91) above spiral (496.17), while terminal-hold success ranks spiral above PPO. Future checkpoint selection should prioritize development-set terminal-hold success over training reward. Successful-time averages use different subsets and must not be interpreted as a speed ranking. On the shared successful cases 2 and 6, both PPO and spiral average 4.57 seconds.

Force values are commanded impedance-wrench magnitudes, not sensor-measured contact forces. Initial reset states have nonzero arm velocities despite zero initial finite-difference velocity observations; their exact equality across controllers supports this comparison, not a stationary deployment handover.

## Reproduce the comparison offline

From the repository root:

```bash
python3 scripts/compare_evaluations.py \
  --runs benchmarks/gpu_integration_20261002/zero_residual \
         benchmarks/gpu_integration_20261002/spiral \
         benchmarks/gpu_integration_20261002/ppo \
  --output /tmp/insertion_comparison.json
```

Use a new output path. The saved `comparison_smoke.json` was produced on the Pod; platform roundoff in Wilson bounds may be approximately `1e-16` when regenerated locally.

## Runtime and next experiment

The Pod intermittently denied new processes access to GPU device nodes and reported NVML errors. Initialized simulation processes completed the recorded episodes; subsequent application starts required container recovery. The specific host-side trigger has not been established. All 79 regression tests passed on the Pod, including the early CUDA-access diagnostic.

The 64-case development manifest is prepared with seeds disjoint from these integration cases and the 500-case holdout. It has not been run. Before further training, compare the recovered training environment snapshot with this evaluation configuration, instrument effective targets and hold transitions, and align development checkpoint selection with terminal-hold success. Keep the 500-case holdout unused while diagnosing and tuning. Multi-seed training, randomization ablations, ROS 2 integration and recovery remain subsequent work.
