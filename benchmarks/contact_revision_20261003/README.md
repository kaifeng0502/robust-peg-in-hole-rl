# Contact action and reward diagnostic, 2026-10-03

The four selected development cases replay the retained held_v2 epoch113 PPO.
The archive includes complete substep control records, exact reward components,
initial states, outcome traces, metadata, logs, and the failed first diagnostic
attempt. It does not contain new trained policy results.

Both misalignment cases (000002 and000010) mapped every legal Z action to the
same downwards impedance target in all1200 physics substeps. Case000028 did so
in902/1200 substeps. The successful case000000 did so in154/182 unlatched
substeps; success holding overrides later policy targets.

Recorded initial states match the previous evaluation exactly. Success/failure
categories match, but continuous trajectories are not all identical: maximum
XY discrepancy is0.8615mm and case000028 depth discrepancy is8.2841mm. Case000000
is bitwise equal for reported XY,depth,reward,and commanded-force samples. Do not
claim universal deterministic replay, attribute all differences to logging, or
interpret commanded wrench as measured contact force. Selected cases establish
a controller limitation, not its causal contribution to aggregate success.

`contact_contract_1024.json` verifies two150-interval episodes on1024 environments
with the new relative-Z control and entry reward. Randomization is disabled for
this runtime check; scripted actions exercise lifting/lowering then insertion.
Both nominal episodes satisfy held-success in all1024 environments. This is
contract validation and must not be reported as randomized PPO performance.

Candidate action `relative_z_v1` anchors Z once per control interval to measured
current tool height plus filtered action×3mm. Positive lifts and negative lowers;
the target is held for eight physics substeps. XY,rotation,observations,workspace
bounds,impedance bounds,and success latch are unchanged. Default action remains
`absolute_residual_v1` for explicit legacy replay; training count defaults and
training entry-point guard now require1024 environments.

Candidate dense reward `entry_v1` is opt-in. It combines3mm and8mm alignment
kernels, removes repeated approach-state reward, and adds differences of
alignment/entry potentials. Entry potential continuously combines approach and
first2.5mm of entry. Existing depth progress,depth,engagement,keypoint,penalties,
and strict terminal-hold terms remain. Potential-change sums around a closed
geometric path are zero before discounting; no policy-invariance or optimality
claim is made. Reward geometry uses simulator truth, not newly available sensors.

The bounded follow-up uses two arms with identical revised action contract and
warm-start checkpoint113, each1024×128×8=1,048,576 new transitions, stopping at121.
Only the dense reward profile differs. This isolates that reward change under
the revised interface; it does not isolate the action change from training.
The first native training startup aborted before sampling with
`malloc(): invalid size (unsorted)`; a fresh identical-config retry retains the
failure evidence. The [completed 64-case comparison](pilot_results/README.md) records 0/64 for both transferred policies. Both predominantly lift away from the entrance; neither candidate is adopted. Final500
cases remain sealed. No100-rollout extension is authorized by this experiment.
