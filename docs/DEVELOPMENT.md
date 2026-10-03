# Development diagnosis and bounded PPO continuation

The final 500-case holdout remains sealed during development. The 64-case
development manifest is shared by the nominal, spiral, and PPO references and
by subsequent candidate checkpoints. Reusing these cases for selection makes
their success rates development measurements, not unseen-test estimates.

## Initial diagnosis

The eight migration cases motivate two hypotheses. Four PPO failures remained
near the hole entrance throughout the episode, while another reached valid
geometry at 9.8 s and could not complete the required terminal hold. Some
recorded observation errors exceed the search coverage used by the current
policy. These traces suggest insufficient search, but commanded force and
geometry alone do not establish physical jamming or identify its cause.

There is also a confirmed objective mismatch: legacy training rewards the
Factory one-sided success condition, whereas evaluation requires two-sided
depth tolerance and a continuous final one-second hold. The eight cases do
not demonstrate that this mismatch is the main cause of failure.

Use `scripts/analyze_development.py` on complete matched runs. It first checks
protocol, runtime, source and realized initial-state agreement, then reconstructs
trial metrics from all trajectory samples. Fixed bins describe initial root XY
distance, insertion-axis tilt and friction. They are descriptive associations;
they do not isolate the causal effect of each randomization source. Force is
the commanded impedance force, not measured contact force.

```bash
python scripts/analyze_development.py \
  --runs /path/to/zero_residual /path/to/spiral /path/to/ppo \
  --output /path/to/new_diagnosis.json
```

## Predeclared pilot

The [pilot manifest](../configs/held_v2_pilot.json) fixes the budget, seed,
checkpoint selection and decision rule before observing candidate outcomes.
Both continuation arms start from the retained epoch-93 checkpoint, restore
its model/normalizers/optimizer, and add exactly 20 epochs of 128 actors × 128
steps, or 327,680 transitions per arm. Evaluate the final epoch-113 checkpoint
from each arm, rather than selecting by its training reward.

The control arm keeps the legacy reward. The candidate uses strict geometry,
requires 15 consecutive successful control intervals for its first-success
reward, and adds a predeclared reward of 100 for a qualifying terminal hold.
Each interval requires all of its physics-boundary samples to satisfy the
geometry criterion. The physical controller, simulator truth used by the
controller's existing pose latch, observations, actions, gains and randomization
remain shared between arms.

Both arms set the training episode length to `10.066666666666666` seconds.
Factory ends at `max_episode_length - 1`, so this produces 150 scored control
intervals (10 seconds) instead of the historical default's 149. This alignment
is shared by both new arms; their only treatment difference is the reward
profile. A simulator check must verify the actual terminal step before training.

Candidate evaluation uses a separate frozen checkout of `c15f16b`. All old and
new policies therefore face the same evaluator, physical controller, evaluation
reward and success criterion. Keep the working/training checkout separate from
that evaluation checkout until all runs finish; never update source files used
by a live experiment.

`scripts/check_training_contract.py` performs two full nominal episodes with
two simulated environments. It checks finite observations/rewards, the actual
150-step terminal boundary, training held-success counts against the evaluation
accumulator at every step, emitted terminal metrics, and clearing state after
automatic reset. Its result is an integration check, not a randomized success
estimate.

The [2026-10-03 runtime evidence](../benchmarks/held_v2_contract_20261003/README.md)
records this check and all 102 regression tests passing on AutoDL.

`scripts/run_development_pilot.py` waits for this runtime check and the complete
three-method reference comparison, verifies traces, runs the two fixed-budget
continuations, verifies saved epoch/frame counts, evaluates the fixed final
checkpoints with the frozen evaluator, and produces a paired candidate report.
It writes phase/PID information to `status.json`; a failed prerequisite or
subprocess stops dependent work and retains its logs. Each stage has a bounded
deadline. Example from the activated candidate checkout:

```bash
python scripts/run_development_pilot.py \
  --runtime-root /root/autodl-tmp/insertion \
  --reference-results /root/autodl-tmp/insertion/results/development_v1_parallel_20261003 \
  --contract-check /root/autodl-tmp/insertion/results/held_v2_training_contract_20261003.json \
  --output /root/autodl-tmp/insertion/results/held_v2_pilot_20261003
```

The output directory must be new. Use a detached process with a retained log
for long runs. Inspect `status.json`, per-stage logs and the final
`exit_code.txt` rather than assuming a disconnected SSH session completed the
experiment. A failed/partial run is never included in performance comparisons.

Compare each candidate against the same complete non-learned references using
the standard comparison gate. For direct PPO-to-PPO analysis, retain their
distinct checkpoint identities and inspect paired per-case outcomes only after
both have independently passed the same reference gate. Do not relabel a PPO
run as another controller to bypass the duplicate-method protection.

If held_v2 has equal or lower development success than the matched legacy
continuation, report no demonstrated improvement. If it improves, report the
observed counts and limitations before extending the budget or adding training
seeds. Only after a controller and checkpoint are frozen should the final
holdout be used. Search-target changes, recurrence and sensor-observation
changes belong to later separate experiments, not this first reward experiment.

The original training environment snapshot is unavailable, and the recovered
checkpoint's frame count does not establish total curriculum cost. Report
newly added transitions separately from the inherited checkpoint provenance.
