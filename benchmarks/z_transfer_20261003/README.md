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

## Bounded follow-up — running

After the gate passed, one continuation started with 1,024 environments,
128-step rollouts and eight added rollouts: exactly 1,048,576 new transitions,
epoch 113 to 121. Seed 42, legacy dense reward and terminal-hold-v2 success
reward are fixed. The final model will be checked on the same four cases.
There is no 64-case evaluation or automatic training extension in this queue.
Training outcome is pending; no new policy is adopted.

Physics, observations and success criteria are unchanged. The final 500-case
holdout remains sealed. Force values are commanded impedance wrench, not
sensor-measured contact force. Shared critic/normalizer transfer, stochastic
Z exploration and a single training seed remain limitations.

`files_sha256.json` hashes the frozen gate artifacts. `four_case_gate.tar.gz`
contains full raw diagnostics and evaluator metadata. The original reference
is in `../contact_revision_20261003/diagnostic_reference_retry1.tar.gz`.
To rerun the stdlib audit, extract this gate archive here and extract the
reference into a directory named `reference`, then run `audit_gate.py`.
The archived scripts record the exact migration and bounded remote workflow;
remote absolute paths are experiment provenance. Model binaries are preserved
in SHA-verified local and remote backups rather than committed to Git.
