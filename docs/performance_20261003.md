# Environment stepping performance — 2026-10-03

Increasing a single process from 128 to 512 environments raised measured
sampling throughput from **290.5–323.4 to 1,175.6–1,304.0 transitions/s** on an
RTX 4090 D. The two baseline/capacity measurement pairs gave **3.64× and 4.49×**
throughput. These are synthetic stepping measurements, not PPO training-speed
or learning-quality results. The original controller remains in use; the
experimental `target_updates` patch has no demonstrated speed benefit.

## Measurement contract

Each process used 150 warm-up control steps followed by 450 measured steps,
covering three complete episodes per environment. Synchronized wall-clock
boundaries include automatic resets and the same lightweight GPU validity
checks. Startup, warm-up and policy inference/updates are excluded. Throughput
runs have no profiler. Actions are reproducible uniform replay in [-0.2, 0.2],
generated with a private CPU RNG; seed 42 is shared across measurements.

The recorded environment configuration differs only in `scene.num_envs`
between the 128- and 512-environment reference runs. Physics remains at
1/120 second with decimation 8 and 192 position-solver iterations. Each scored
episode has 150 control intervals, or 10 seconds; the hold-duration setting
remains one second. Controller, gains, observations, actions, reward profile,
pose/friction randomization and geometry criteria are unchanged. Capacity
tests use the legacy reward profile.

The project fingerprint is
`970a5a6fcb8daacd08cccf66c85532ebbd07b4b08328b28df66299261ba713e6`.
Runtime, Factory/DirectEnv/controller hashes and Torch execution settings match
across reference runs: Isaac Sim 5.0.0.0, Torch 2.7.0+cu128 and 64 intraop/64
interop threads. Other experiment processes were paused; their CUDA contexts
continued to reserve approximately 9 GB of VRAM.

## Uninstrumented results

| Measurement | Environments | Transitions | Wall time (s) | Transitions/s |
| --- | ---: | ---: | ---: | ---: |
| [Reference 1](../benchmarks/performance_20261003/reference_single_1/result.json) | 128 | 57,600 | 178.13 | 323.35 |
| [Capacity 1](../benchmarks/performance_20261003/reference_capacity_512/result.json) | 512 | 230,400 | 195.99 | 1,175.60 |
| [Reference 2](../benchmarks/performance_20261003/reference_single_2/result.json) | 128 | 57,600 | 198.25 | 290.55 |
| [Capacity 2](../benchmarks/performance_20261003/reference_capacity_512_repeat/result.json) | 512 | 230,400 | 176.68 | 1,304.05 |
| [Target updates](../benchmarks/performance_20261003/target_single_1/result.json) | 128 | 57,600 | 195.18 | 295.11 |

All runs completed the expected 384 or 1,536 environment episodes with finite
observations/rewards. Their 3,600 policy physics substeps exclude additional
reset substeps, whose time is included in the wall-clock measurement.

The first capacity/reference ratio is 3.6356×; the second is 4.4883×. At these
rates, a fixed sample count would take approximately 22.3–27.5% of the original
sampling time, a 72.5–77.7% reduction. This is a throughput conversion, not a
direct measurement of end-to-end training under an equal sample budget.
Only two reference/capacity pairs were collected, sequentially and without
randomized ordering; the ranges are observed variation, not confidence bounds.

The first reference used measurement-script hash `56e3b1c6…`; later runs used
`b59e41a7…`, which adds variant selection and provenance recording. Both repeat
runs use the same latter script, removing the script-version mismatch from the
second pair. The archived v1 and current v2 sources retain the same reference
workload and timed loop; their hashes and the exact result/configuration files
are listed in the [provenance manifest](../benchmarks/performance_20261003/provenance.json).
The action hash is repeatable within each environment count but
differs across counts because the action tensor has a different shape. Initial
randomization trajectories are not matched across batch sizes.

`target_updates` took 9.57% longer than the first reference, but its 195.18-second
result lies inside the original controller's observed 178.13–198.25-second
range. A causal regression is therefore not established. There is no measured
benefit supporting adoption, so the patch is left out of the active controller.

Two concurrent 128-environment reference processes produced 115,200 transitions
over a common 225.663-second measurement window: **510.50 transitions/s** in
aggregate. This is 1.58× the first single-process reference, while individual
workers slowed to 255.25 and 261.50 transitions/s. The aggregate uses the earliest
start and latest end, not the sum of separately timed rates; start skew was
0.57 ms. See the [worker 0](../benchmarks/performance_20261003/reference_dual_1_worker_0/result.json)
and [worker 1](../benchmarks/performance_20261003/reference_dual_1_worker_1/result.json)
records.

## Diagnostic observations and limitations

The separate [instrumented run](../benchmarks/performance_20261003/reference_diagnostic_1/result.json)
took 195.39 seconds. Host-observed region totals included 96.94 seconds in
`sim_step`, 63.55 seconds in `apply_action` (23.47 exclusive), 16.33 seconds in
control-torque computation and 15.61 seconds in kinematics. Reset took 9.45
seconds inclusive, including 4.02 seconds in IK; friction updates took 0.014
seconds. These measurements include CPU work, enqueueing and waiting. Nested
inclusive regions overlap and must not be added. They do not isolate pure GPU
kernel time or prove that a particular CUDA synchronization caused the delay.

Reviewed capacity logs contain no PhysX/contact-buffer overflow, CUDA
out-of-memory, assertion or traceback. All compared runs nevertheless emit the
same Warp `cuDeviceGetUuid`/API-36 startup error, followed by completed CUDA
stepping. Its impact was not separately established. CPU powersave, PCIe link
width, shared Kit-cache and shutdown-stage warnings are also retained in the
evidence; the logs are not described as error-free. Finite states and correct
episode counts alone do not establish contact-physics or trajectory equivalence.

The 512-environment setting is a supported capacity option for subsequent
controlled work. It does not accelerate one case by 3.64–4.49×, nor does it prove
equal sample efficiency, policy quality or PPO learning dynamics: retaining the
same rollout length would quadruple samples per update. No formal training
budget, checkpoint, reward, controller or final evaluation protocol was changed
by these measurements. The final 500-case holdout remains sealed. Any later
training configuration must state its sample budget explicitly and measure
actual training throughput and matched evaluation outcomes separately.
