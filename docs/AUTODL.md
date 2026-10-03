# AutoDL runtime and migration checks

## Verified status — 2026-10-03

The migrated environment passes dependency consistency checks, all 79 unit/regression tests, and three CUDA allocation/computation checks in fresh Python processes. The headless RL simulator smoke check also passed with four environments and 12 control steps.

The eight-case evaluation is complete: zero residual succeeded in 2/8 cases, spiral search in 4/8, and PPO in 3/8. The paired comparison passed on AutoDL and locally. See the [complete migration evidence](../benchmarks/gpu_integration_20261003_autodl/README.md). These replicate the historical per-case success decisions; PPO trajectories have small numerical differences. This is not a new training run or evidence of a 95% success rate.

| Component | Migrated runtime |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D, approximately 24 GB VRAM |
| Driver | `595.71.05` |
| Operating system | Ubuntu `22.04.5` |
| Python | `3.11.13` |
| PyTorch / CUDA runtime | `2.7.0+cu128` / `12.8` |
| Isaac Sim | `5.0.0.0`, including extension-cache packages |
| Isaac Lab | release `2.2.1`, commit `0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20` |
| RL-Games | `1.6.1`, source commit `6b3534f29568158e9e29ec8bf83cc88fce5f0cae` |

This GPU/runtime is distinct from the historical L4 environment. Re-evaluate all three methods on AutoDL and compare them using the same new runtime and case manifest; retain the [RunPod integration evidence](../benchmarks/gpu_integration_20261002/README.md) as a separate benchmark.

## Runtime paths and activation

The migration root is `/root/autodl-tmp/insertion`:

| Path below the migration root | Purpose |
|---|---|
| `env/` | Python environment and installed simulator packages |
| `IsaacLab/` | Pinned upstream source |
| `baseline_rl/` | This project |
| `activate_insertion.sh` | Environment, graphics, cache, and temporary-directory settings |
| `nvidia_egl_icd.json` | Project-specific Vulkan ICD selection |
| `logs/` | Installation, dependency, CUDA, and migration-check logs |
| `results/autodl_integration_20261003/` | Completed integration run outputs |

Activate the provisioned environment before running commands:

```bash
source /root/autodl-tmp/insertion/activate_insertion.sh
cd /root/autodl-tmp/insertion/baseline_rl
python -m pip check
```

The helper activates the environment, sets `ISAACLAB_PATH`, and keeps cache/temp directories under the migration root. Its settings apply to the current shell and child processes.

## Headless Vulkan configuration

The container's default NVIDIA GLX ICD failed, while NVIDIA EGL exposed a working Vulkan device. CUDA access was verified independently. The project activation helper therefore selects the verified EGL ICD:

```bash
export VK_ICD_FILENAMES=/root/autodl-tmp/insertion/nvidia_egl_icd.json
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
```

The ICD points to `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`. This selection is scoped to project processes; it does not replace the host driver or change the machine's global Vulkan configuration. If the container or host changes, verify that the library and ICD still exist and that both CUDA allocation and Vulkan device enumeration work before starting Isaac Sim.

## Dependency constraints and source provenance

Use [runtime constraints](../configs/autodl_runtime_constraints.txt) for package installations and [build constraints](../configs/isaaclab_build_constraints.txt) for isolated builds. The runtime constraints preserve Isaac Sim 5.0's dependency requirements alongside the compatible task/learner packages. They are not a replacement for the complete installed-package snapshot in `/root/autodl-tmp/insertion/logs/pip_freeze.txt`.

The verified environment uses pip `25.3`, which supports isolated build constraints. For subsequent installation or repair commands, preserve both layers:

```bash
export PIP_CONSTRAINT=/root/autodl-tmp/insertion/baseline_rl/configs/autodl_runtime_constraints.txt
export PIP_BUILD_CONSTRAINT=/root/autodl-tmp/insertion/baseline_rl/configs/isaaclab_build_constraints.txt
```

Build dependencies retain `setuptools<81` and `wheel==0.45.1`. This preserves compatibility with packages requiring `pkg_resources` and the simulator's `packaging==23.0` requirement. Verified supporting versions include `transformers==4.56.2`, `huggingface-hub==0.36.2`, `tokenizers==0.22.2`, `onnx==1.18.0`, and `typing_extensions==4.12.2`. Always run `python -m pip check` after a package change and retain a fresh `python -m pip freeze --all` snapshot.

The RL-Games source and portable wheel were prepared from the pinned commit above. Local migration artifacts are stored under `output/autodl_migration_20261003`, outside this project repository, together with `rl_games_provenance.json`. The remote migration root holds the wheel and provenance needed for installation without a GitHub fetch:

| Artifact | SHA256 |
|---|---|
| `rl_games-1.6.1-py3-none-any.whl` | `f2f4684ee78128a32b542843f3d7b9074a3b8a195203aa857ba8ab65c1313986` |
| `rl_games_source.tar.gz` | `6cf0150bee5ea014c77101487b3bbadf9dce327b982f10d3974eca2dd737d9ce` |

The wheel uses the Poetry metadata in `pyproject.toml`, rather than the divergent legacy `setup.py`. It is pure Python and contains no platform-specific binary extensions. The fetched commit is pinned for AutoDL; the exact historical RunPod RL-Games revision remains unknown despite its matching `1.6.1` package version.

## Retained checkpoint

The model and its original learning configuration are ignored by Git and must be transferred separately when reconstructing the environment. Their hashes were verified after transfer:

| Path relative to this project | SHA256 |
|---|---|
| `results/recovered_continue_20260917/nn/LocalInsertion.pth` | `9811f2bfa13dab366531324fbab5bb758a7e031753a6378b5d318ad99d843721` |
| `results/recovered_continue_20260917/params/agent.yaml` | `10cfe0b4e3974952b02b565054a47ebcc2cf68a5ff21ca5fa316f448f7dfbe7e` |

The retained checkpoint is from epoch 93 of the recovered continuation run. The original training environment snapshot, `params/env.yaml` / `params/env.pkl`, is still missing locally. The agent YAML preserves network/learner settings and normalization flags; it cannot substitute for the missing training environment configuration when investigating distribution or controller differences.

## Run and recovery procedure

The completed migration launcher wrote progress to `/root/autodl-tmp/insertion/logs/migration_checks.log`. Individual test and simulator logs are under `/root/autodl-tmp/insertion/results/autodl_integration_20261003`. Inspect these before starting another process. Do not start a duplicate evaluator while the current sequence is running.

For a deliberate rerun after addressing a failure, keep the failed output and use new directories. From the activated project directory, the following creates a separate three-method integration attempt; `autodl_recheck_01` must not already contain these method outputs:

```bash
python scripts/evaluate_insertion.py \
  --method zero_residual --cases configs/evaluation_smoke.json --headless \
  --output /root/autodl-tmp/insertion/results/autodl_recheck_01/zero_residual

python scripts/evaluate_insertion.py \
  --method spiral --cases configs/evaluation_smoke.json --headless \
  --output /root/autodl-tmp/insertion/results/autodl_recheck_01/spiral

python scripts/evaluate_insertion.py \
  --method ppo --cases configs/evaluation_smoke.json --headless \
  --checkpoint results/recovered_continue_20260917/nn/LocalInsertion.pth \
  --agent-config results/recovered_continue_20260917/params/agent.yaml \
  --output /root/autodl-tmp/insertion/results/autodl_recheck_01/ppo

python scripts/compare_evaluations.py \
  --runs /root/autodl-tmp/insertion/results/autodl_recheck_01/zero_residual \
         /root/autodl-tmp/insertion/results/autodl_recheck_01/spiral \
         /root/autodl-tmp/insertion/results/autodl_recheck_01/ppo \
  --output /root/autodl-tmp/insertion/results/autodl_recheck_01/comparison_smoke.json
```

Each evaluator checks CUDA availability and allocation before simulator startup. Treat a startup or dependency failure as an invalid run, not an insertion failure. All three method directories must report complete coverage and pass the runtime, protocol, source, and initial-state comparison checks before a new comparison is accepted. Keep the established state tolerance; investigate a mismatch rather than relaxing the gate.

The [evaluation protocol](EVALUATION.md) defines the terminal one-second hold, artifact fields, and failure interpretation. AutoDL integration is accepted. Use the separate 64-case `configs/evaluation_development.json` for diagnosis and tuning. The 500-case `configs/evaluation_holdout.json` remains unused until the controller/checkpoint is frozen. Algorithm expansion to SAC/TD3 and video recording are outside this migration.
