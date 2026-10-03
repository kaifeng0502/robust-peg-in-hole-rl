# Z policy transfer calibration

The direct transfer into `relative_z_v1` previously produced lifting-away
behavior and 0/64 held successes in each reward arm. This follow-up changes
only the Z mean-output row of the original epoch-113 policy: its weights are
zero and its bias is -0.5. Other model parameters, including the shared trunk,
critic, observation/value normalizers and state-dependent sigma, are retained.
Adam first/second moments for the two changed rows are zeroed; the shared Adam
step is retained. This is partial policy reinitialization, not an equivalent
action mapping or a policy that has learned to lower the peg.

On 600 identical recorded observations, the other five mean outputs and all
sigma outputs are bitwise unchanged. Inverting the old bounded Z target into a
relative command yielded the downward limit at all 473 unlatched first
substeps. A compatible target conversion alone therefore retains saturation.
The new bias produces a -1.5 mm target offset after the existing EMA settles;
it does not prescribe a 1.5 mm physical movement. Feedback into the next
observation means closed-loop non-Z actions need not remain unchanged.

## Four-case behavioral gate — complete

Cases were selected before this calibration and reused from the previous
diagnosis. They are debugging examples, not a representative success-rate
estimate. All four realized initial states match the reference exactly.
Independent reconstruction checks 600 control intervals / 4,800 substeps,
10-second episodes, strict final one-second holds, rewards and target bounds.

| Case | Held before / calibrated | Final calibrated depth (mm) | Final XY error (mm) | Peak commanded force (N) |
|---|---|---:|---:|---:|
| 000000 | yes / yes | 24.926 | 0.516 | 1.748 |
| 000002 | no / no | -0.155 | 5.196 | 2.178 |
| 000010 | no / no | -0.158 | 4.761 | 1.738 |
| 000028 | no / no | 1.303 | 0.648 | 1.525 |

All four lower toward the entrance, and the previously successful case retains
its terminal hold. Commanded-force peaks are lower than each reference case.
Two cases remain misaligned and one remains insufficiently inserted. In
particular, the depth-failure case progresses less than under the old policy.
Passing this gate establishes usable initial Z behavior, not better insertion
performance. The predeclared force regression bound (twice the matching
reference peak) is a screening condition, not a measured hardware safety limit.

## Bounded follow-up — complete and audited

The single continuation completed eight 128-step rollouts on 1,024 environments:
exactly 1,048,576 new transitions, epoch 113 to 121 and frame 1,851,392 to
2,899,968. Seed 42, legacy dense reward and terminal-hold-v2 success reward
were fixed. Independent CPU loading verifies finite model tensors, the exact
sample budget, saved configuration, and the checkpoint used by evaluation.

The same four cases retain **one held success and three failures**, with zero
paired gains or losses. All unlatched first-substep Z commands remain downward;
unlatched raw mean Z ranges from -0.844 to -0.898 across cases. This fixes the
lifting-away failure but does not demonstrate successful contact search.

| Measure | Calibrated initializer | After eight rollouts | Original absolute-interface reference |
|---|---:|---:|---:|
| Case 000000: first complete one-second hold (s) | 8.733 | 4.267 | 2.533 |
| Case 000002: final XY error (mm) | 5.196 | 3.802 | 5.433 |
| Case 000010: final XY error (mm) | 4.761 | 5.945 | 4.694 |
| Case 000028: final insertion depth (mm) | 1.303 | 16.483 | 22.471 |
| Mean common task return | 360.265 | 436.296 | 492.970 |
| Mean peak commanded force (N) | 1.797 | 1.915 | 2.271 |

The depth-failure case advances substantially relative to the initializer but
still falls short of both the 25 mm target and the old policy. One misalignment
error improves and the other worsens. All post-training force peaks are below
the original reference peaks, although three rise relative to the calibrated
initializer. These observations do not justify replacing the original model.
The original-interface column is historical context, not an action-only causal
comparison. No 64-case evaluation or additional training was started.

The five logged training terminal-held-success fractions increase from 4.98%
to 26.46%; they are training samples under exploration, not held-out performance
or convergence evidence. TensorBoard contains epoch scalars through 120; the
checkpoint and all eight log entries independently establish completion at 121.
Training wall time is 515.89 s including startup, and the four-case check takes
222.09 s. Median logged total throughput is 2,234 transitions/s. Device-wide
10-second memory samples peak at 5,380 MiB during training and 4,538 MiB during
the single evaluator; these are not allocator peaks.

The completed archive contains 27 SHA-verified files including checkpoints,
TensorBoard, configurations and all raw diagnostics. `backup_provenance.json`
records its SHA. `post_train_four_audit.json` verifies 600 control intervals and
4,800 diagnostic substeps; all realized initial states match both preceding
stages. Hold reconstruction uses the recorded full-interval geometry flags.
`four_case_comparison.json` retains the complete descriptive comparison.

Physics, observations and success criteria are unchanged. The final 500-case
holdout remains sealed. Force values are commanded impedance wrench, not
sensor-measured contact force. Shared critic/normalizer transfer, stochastic
Z exploration and a single training seed remain limitations.

`files_sha256.json` hashes the published gate and completion artifacts. `four_case_gate.tar.gz`
contains full raw diagnostics and evaluator metadata. The original reference
is in `../contact_revision_20261003/diagnostic_reference_retry1.tar.gz`.
To rerun the stdlib audit, extract this gate archive here and extract the
reference into a directory named `reference`, then run `audit_gate.py`. Extract `post_training_four_cases.tar.gz` and run
`audit_gate.py post_train_four` for the final check; `compare_four.py` reconstructs
the three-stage comparison.
The archived scripts record the exact migration and bounded remote workflow;
remote absolute paths are experiment provenance. Model binaries are preserved
in SHA-verified local and remote backups rather than committed to Git.
