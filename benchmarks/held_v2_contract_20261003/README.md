# Terminal-hold training runtime check

On 2026-10-03 the RTX 4090 D AutoDL runtime passed all 102 unit/regression tests
and a two-environment, two-episode nominal simulator check. Each episode ended
after exactly 150 control intervals. The training hold counters matched the
independent evaluation accumulator at every control interval, and counters
cleared after automatic reset. All four nominal episodes exercised a positive
terminal hold. The terminal bonus was zero before the endpoint and 100 at each
successful endpoint. Observations and rewards remained finite.

Evidence: [contract.json](contract.json) and [unit_tests.txt](unit_tests.txt).
The tested candidate code is commit `d9086c0`, with source fingerprint
`970a5a6fcb8daacd08cccf66c85532ebbd07b4b08328b28df66299261ba713e6`.
The fingerprint covers Python/YAML files under `local_insertion/` and `scripts/`.
The artifact also records runtime and upstream source identities.

The actual Hydra training command passed a configuration-only preflight, with
the explicit experiment-name override and fixed reward/horizon settings. This
does not establish that a continuation has completed.

These nominal checks validate training mechanics. They are not randomized
policy-performance measurements. The 64-case reference evaluation and the
equal-budget continuation pilot are separate experiments; their outcomes are
pending. The 500-case final holdout remains unused.
