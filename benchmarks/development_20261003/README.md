# Matched development evaluation and PPO continuation — 2026-10-03

Both predeclared continuation arms completed training and all 64 development
cases. Legacy continuation achieved 39/64 terminal-held successes; the
terminal-hold reward candidate achieved 40/64. The recovered policy achieved
37/64 and deterministic spiral search achieved 44/64 on the same cases.
The candidate has a small observed gain, but has not established an advantage
over spiral search or reached the 95% target.

## Results

All methods use 150 scored control intervals (10 seconds), 1/120-second physics,
decimation 8, and valid insertion throughout the final continuous second.
The shared controller's existing pose latch uses simulator success geometry;
this is a simulation development comparison, not a sensor-only deployment.

| Controller / checkpoint | Terminal-held successes | Wilson 95% interval | Misalignment failures | Depth failures | Hold failures |
|---|---:|---:|---:|---:|---:|
| Zero residual | 14/64 (21.88%) | 13.50–33.43% | 33 | 17 | 0 |
| Impedance + spiral search | 44/64 (68.75%) | 56.61–78.77% | 15 | 4 | 1 |
| Recovered PPO, epoch 93 | 37/64 (57.81%) | 45.61–69.13% | 20 | 6 | 1 |
| Legacy continuation, epoch 113 | 39/64 (60.94%) | 48.69–71.94% | 19 | 6 | 0 |
| Terminal-hold reward, epoch 113 | 40/64 (62.50%) | 50.25–73.33% | 19 | 5 | 0 |

For the two continuation checkpoints, any-time, final and terminal-held success
counts agree. All remaining candidate failures never reached success geometry:
19 end misaligned and five have insufficient depth. These categories describe
recorded geometry, not a diagnosis of friction-induced jamming.

## Paired changes

| Candidate relative to reference | Newly successful | Previously successful cases lost | Net successes | Common-success cases | Mean completion-time change on common successes |
|---|---:|---:|---:|---:|---:|
| Legacy continuation vs recovered | 5 | 3 | +2 | 34 | −0.473 s |
| Terminal-hold vs recovered | 4 | 1 | +3 | 36 | −0.663 s |
| Terminal-hold vs legacy continuation | 3 | 2 | +1 | 37 | −0.005 s |
| Terminal-hold vs spiral | 5 | 9 | −4 | 35 | −0.625 s |

The reward treatment adds three successes (`case_000008`, `case_000019`,
`case_000049`) and loses two (`case_000028`, `case_000034`) relative to legacy
continuation. A net gain of one case is 1.5625 percentage points; it is weak
evidence for a repeatable treatment effect. Both arms start from the same
checkpoint and use one training seed, so they are not independent seed trials.

Mean per-case peak **commanded** force is 2.3057 N for legacy and 2.3114 N for
terminal-hold, a paired increase of 0.0058 N. These are impedance command
magnitudes, not sensor-measured contact forces. Common-success completion times
are essentially unchanged between the reward arms; averages over each arm's
different successful subset must not be used to claim a speed advantage.

The frozen evaluator's common task-return means are 686.05 (recovered), 697.29
(legacy), 716.98 (terminal-hold), and 653.64 (spiral). Higher return does not
override the success-rate ranking. Training returns use different reward
definitions across the two arms and must not be ranked directly.

## Training and provenance

The [predeclared manifest](../../configs/held_v2_pilot.json) fixed seed 42,
128 environments, 128-step rollouts, final epoch 113, and exactly **327,680
additional transitions per arm** (20 epochs from epoch 93). Model, normalizer
and optimizer state were restored. Checkpoints were selected by fixed final
epoch, not by reward. The [budget record](training_budget.json) links the final
hashes to their epoch/frame counters; inherited counters are not a complete
historical curriculum cost.

Evaluation remained on frozen commit `c15f16bbd2f9fa5f9480d103cd69e0fc125a9e38`.
Its reward, physical controller, observations and success criterion were
unchanged. Both new training arms use 150 scored intervals. The retained
historical/default configuration implies 149 intervals, but the exact historical
runtime configuration is unavailable. Comparison with the recovered checkpoint
therefore cannot isolate additional training from environment reconstruction
and the declared horizon alignment. The original training environment snapshot
remains unavailable.

The saved TensorBoard `info/epochs` points cover 94–112. Logs and final
checkpoint metadata independently establish completion at 113; the missing
last scalar must not be relabeled. The candidate's logged terminal-held rate
averages 83.59% over its first five available points and 80.31% over its last
five, with 78.91% at epoch 112. It does not show a sustained upward trend and
does not replace the deterministic development result. Available KL, learning
rate, loss and entropy values are finite; one seed cannot establish training
stability.

## Decision on further training

No additional training was launched after this pilot. The small observed gain
does not justify an unrestricted extension or an expectation that more epochs
alone will produce 95–100% success. The dominant unresolved failure is reaching
insertion geometry, so alignment/search behavior deserves priority over adding
more terminal-hold reward.

For a practical larger-data trial at the requested parallelism, the first
diagnostic cap proposed here is 1,024 environments × 128
steps × eight rollouts = **1,048,576 new transitions per arm**, or 2,097,152
across both reward arms. Evaluate the fixed eighth-rollout checkpoint before
deciding whether any second block is justified; 16 total rollouts would be
2,097,152 per arm, but are not committed in advance. This proposal is not an
active training allocation. Define selection and stopping rules before
execution, check actual PPO memory first, and retain
the frozen evaluation protocol. Changing environment count changes rollout
batch size and optimization, so it is a new training regime, not an equivalent
replay of the 128-environment experiment. The existing capacity measurement
does not validate PPO learning quality or training memory. Such a trial cannot
attribute any gain solely to extra samples: isolating that effect would require
keeping the original environment count or adding a batch-size control.

The same 64 cases have been used for development selection. Their intervals do
not account for that selection, and even 64/64 successes would not establish a
true success probability of at least 95%. The final 500-case holdout remains
unused; release it only after controller/checkpoint selection is finished.

## Verification and retained evidence

- All five runs contain the same 64 case IDs, 150 trace records per case and
  9,600 trace records per run: **48,000 records** were reconstructed locally.
- Each PPO passes the unchanged comparison gate against both zero residual and
  spiral. PPO pairs additionally pass the existing realized-initial-state gate.
  Metadata checks cover runtime, source, protocol, manifest and model identity.
- Local reconstruction agrees with the remote candidate report. Only directory
  labels and two Wilson lower-bound last-bit roundoffs differ (below 1e-15).
  Recorded trajectories and outcomes were not changed to achieve agreement.
- A second audit independently recomputed all five runs from the trace fields
  without using the project's metric helpers. It checked every hold transition,
  terminal geometry, failure label, return, force and time. Realized initial
  states were exactly equal across all pairs, within the unchanged gate.
  The saved control rows contain aggregated physics-interval flags; this audit
  cannot reconstruct unrecorded 120 Hz geometry independently of those flags.
- The evaluations were paused at 29/64 and 28/64 for the user-requested capacity
  work, then resumed in the same processes. All four saved trial/trajectory
  byte prefixes remain unchanged. The old deadline coordinator was retired;
  a read-only observer verified complete artifacts and archived them.
- Worker exit codes are unavailable after coordinator replacement, and are
  retained as null. Both workers exited, `run_complete` is true, final summaries
  and full traces reconcile, and the comparison completed successfully. The
  logs retain Warp/Fabric startup errors shared with the reference runs and USD
  shutdown warnings; complete artifact checks do not certify those subsystems.
  Evaluation wall times include the pause
  and must not be interpreted as throughput measurements.
- The full training archive, including both final checkpoints, configurations
  and TensorBoard data, and the completed evaluation archive were downloaded
  and SHA256-verified. Checkpoint/configuration hashes match evaluation metadata.
  Checkpoint binaries remain in the retained backup; saved YAML and event files
  are included here.

[provenance.json](provenance.json) records SHA256 values for the raw evidence.
Each controller directory contains metadata, summary, per-case results and full
traces. [candidate_comparison.json](candidate_comparison.json) is the remote
report; [local_verified_comparison.json](local_verified_comparison.json) records
the local reconstruction and [independent_trace_audit.json](independent_trace_audit.json)
records the separate implementation's audit. Recovery records document the pause and
resume without changing the original evaluation metadata.

To reproduce the offline comparison from the repository root:

```bash
python scripts/compare_ppo_candidates.py \
  --reference benchmarks/development_20261003/zero_residual \
  --runs benchmarks/development_20261003/recovered \
         benchmarks/development_20261003/legacy_continuation \
         benchmarks/development_20261003/held_v2 \
  --labels recovered legacy_continuation held_v2 \
  --output /tmp/new_candidate_comparison.json
```

Use a new output path. This reads the saved evidence and does not launch a
simulator or use the final holdout.
