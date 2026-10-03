# Isolated stepping measurement tools

These tools benchmark a supplied project checkout without changing its source
or the installed Isaac Lab files. Results and limits are described in
[the performance report](../../docs/performance_20261003.md). The measured source
fingerprint and runtime hashes are recorded with every run. Use the configured
Isaac Lab environment; ordinary Python supports only the offline tests.

Run one process at a time when comparing capacity. Use fresh output directories:

```bash
python tools/performance/profile_environment.py \
  --project /absolute/path/to/measured-checkout \
  --output /absolute/path/to/new-result \
  --num-envs 128 --steps 450 --warmup-steps 150
```

Repeat with `--num-envs 512`. Initialization and warm-up are excluded; automatic
resets and identical validity checks are included. No PPO updates are performed.
The default workload uses reproducible synthetic actions, not a learned policy.
Keep training sample budgets explicit if changing batch size in later work.

For diagnostic host timings and an optional eight-step trace spanning a reset,
add `--mode diagnostic --trace-last-steps 8`. Do not use instrumented wall time
as evidence of speedup. Inclusive nested region totals overlap; host timing is
not pure PhysX GPU kernel time.

For two concurrent workers, pass a fresh common `--barrier` directory (do not
create it), the same fresh UUID as `--barrier-token`, `--participants 2`, and
`--worker-id 0` / `1` with separate output directories. Aggregate throughput is
the sum of transitions divided by the latest end minus earliest start in the
recorded monotonic timestamps.

`--variant target_updates` is an experimental instance-local patch retained to
reproduce the unsuccessful optimization screen. Its speed benefit and full
simulation equivalence are unproven; it is not enabled in the project runtime.
Restore functions remove all instrumentation and instance patches on exit.

```bash
python -m unittest discover -s tools/performance -v
```

The parser, barrier, and timing tests need only Python. Seven tensor test groups
need PyTorch and skip explicitly if unavailable; they passed without skips on
the AutoDL runtime. They do not substitute for GPU trajectory validation.

The exact original v1 measurement script is archived under
`benchmarks/performance_20261003/tool_versions/`; the current v2 script, probe
and target patch hashes are preserved in the evidence provenance manifest.
