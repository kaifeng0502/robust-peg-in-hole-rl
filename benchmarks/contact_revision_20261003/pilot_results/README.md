# Completed 1,024-environment contact-transfer pilot

**Decision: do not adopt either transferred checkpoint or the new dense reward
on the evidence of this pilot.** Both fixed final policies failed all 64 matched
development cases. Keep the original epoch-113 checkpoint and its absolute
residual action contract available; do not automatically increase this budget.

| Metric | Revised Z + legacy dense reward | Revised Z + entry dense reward |
|---|---:|---:|
| Terminal held successes | 0/64 | 0/64 |
| Ever reached valid geometry | 0/64 | 0/64 |
| Terminal misalignment failures | 41 | 43 |
| Terminal insufficient-depth failures | 23 | 21 |
| Mean final signed depth | -27.8245 mm | -27.7892 mm |
| Mean peak commanded force | 1.81184 N | 1.80824 N |
| Maximum commanded force | 3.25732 N | 3.24902 N |
| Mean common task return | 38.92050 | 38.64407 |
| Mean common regularized return | 36.05026 | 35.73136 |
| Success completion time | Not available: no successes | Not available: no successes |
| New training transitions | 1,048,576 | 1,048,576 |
| Median logged total training throughput | 2,369.5 transitions/s | 2,437.0 transitions/s |

Both policies used every 150-interval, 10-second horizon. There are no paired
wins or losses: all 64 cases failed for both. Failure categories describe final
geometry; they do not establish physical jamming. The descriptive Wilson 95%
interval for each 0/64 proportion is [0, 5.6624%], not a generalization guarantee
on this reused development set.

## Failure mechanism and limits

In the complete evaluation traces, raw Z actions were positive in 99.7708% and
99.7292% of control intervals, respectively. Their means were +0.94565 and
+0.93790. All 64 cases in each arm had positive mean Z. Reconstructing the action
EMA gives similarly persistent positive commands. With `relative_z_v1`, positive
Z lifts; the final peg base is 25.8–31.0 mm above the entrance. Thus the measured
failure is sustained lifting away from insertion, not an overly strict hold
counter or a missing success flag.

The source checkpoint's *original* evaluation had positive Z in all 64 initial
controls and 98.0208% of the first-second controls (mean +0.65860). Under the old
absolute target those commands still requested downward motion at the entrance.
Changing the action meaning and directly restoring the old actor, value model,
normalizers and optimizer therefore introduced a transfer mismatch. The new
rollouts and reward change did not correct it within the declared budget. This
supports a concrete mechanism, not proof that action transfer is the only cause
or that longer training could never recover.

Both arms' recorded training geometry-success and held-success scalar series
are zero. Training reward totals differ in definition and are not compared as
performance. The small common-evaluation return difference does not establish
an improvement when both arms fail every case.

The earlier 1,024-environment nominal scripted check correctly demonstrated
lift/lower authority and the simulator's success/reset contract. It did **not**
validate the migrated neural policy. The missing safeguard was a short actual
policy transfer check before the reward experiment. Future transfer work should
first calibrate or initialize the Z output under the new action meaning, verify
approach/entry and intentional unloading with the actual initialized policy,
and assess value/optimizer/normalization transfer separately. Only then should
a new fixed-budget reward comparison be proposed. All future training must use
1,024 environments; no additional training has been launched here.

## Verification and runtime evidence

- Both arms start at epoch 113 and end at the predeclared epoch 121, with
  1,024 actors × 128 controls × 8 rollouts each. Independent CPU loading verifies
  checkpoint frame counters, finite model tensors and identical model shapes.
  Saved configurations differ only in dense reward profile and run name.
- A separate standard-library reconstruction verifies all 19,200 control rows,
  128 case outcomes, final geometry, interval flags, 15-interval terminal hold,
  times, returns, depths and commanded-force aggregates. All recorded paired
  initial states are exactly equal, stricter than the declared 1e-6 tolerance.
  Raw 120 Hz geometry is not saved, so interval flags cannot be independently
  reconstructed from substep positions in these evaluations.
- Both evaluator exits are 0. Remote completion was 2026-10-03 08:38:08 UTC;
  the evaluation stage took about 3,808 seconds. Source fingerprint is
  `1d0ecd6cdd7fb7f79fe864bd2275cb857058039c192ad176b1b02d22293aa67f`.
- Device-wide 10-second memory samples peak at 5,380 MiB in each training arm
  and 9,072 MiB during simultaneous evaluation. These are sampled device peaks,
  not allocator peaks. Both 1,024-environment PPO runs completed without OOM.
- The initial native startup abort, its same-config retry, and shared
  Warp/Fabric/runtime warnings remain documented. No NVML GPU loss occurred
  during the completed retry. No performance patch was enabled.
- The full 41-file archive and a separate completed-training archive are
  downloaded and SHA256 verified. Full model binaries remain in those backups;
  this directory publishes traces, metadata, saved YAML, TensorBoard, logs,
  comparison, independent audits and checkpoint digests.
- Physics stays at 1/120 s, decimation eight, 192 position iterations, 10 seconds
  of scored interaction and one second of final hold. The final 500-case holdout
  remains sealed. No paid instance was shut down.

See `comparison.json`, `independent_completed_pilot_audit.json`,
`independent_training_artifact_audit.json`, `final_diagnosis.json`,
`training_budget.json`, `backup_provenance.json` and `SHA256SUMS.json`.
