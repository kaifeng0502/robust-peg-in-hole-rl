"""Frozen reset inputs and strict provenance checks for insertion comparisons.

Reset seeds identify requested cases; they do not establish equal physical states.
Comparisons additionally require the recorded initial state to match within an
absolute tolerance. This module depends only on the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

if __package__:
    from .evaluation_metrics import EvaluationCriteria, summarize_trials
else:
    from evaluation_metrics import EvaluationCriteria, summarize_trials


SCHEMA_VERSION = 1
DISTRIBUTION = "challenge_v2"
METHODS = frozenset({"spiral", "zero_residual", "ppo"})
INITIAL_STATE_KEYS = frozenset(
    {
        "robot_joint_pos",
        "robot_joint_vel",
        "robot_root_state",
        "held_root_state",
        "fixed_root_state",
        "episode_friction",
        "fixed_pos_obs_noise",
    }
)
TRIAL_METRIC_KEYS = frozenset(
    {
        "ever_success", "final_success", "held_success", "had_hold_anytime",
        "success_time_s", "failure_category", "peak_commanded_force_n",
        "max_depth_m", "episode_reward", "task_episode_reward", "sample_count", "elapsed_time_s",
    }
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def case_set_sha256(data: dict) -> str:
    """Hash the case-set body, excluding its self-referential case_set_id field."""
    if not isinstance(data, dict):
        raise ValueError("case set must be an object")
    body = {key: value for key, value in data.items() if key != "case_set_id"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def validate_case_set(data: dict) -> dict:
    """Reject invalid schemas, duplicate inputs and altered case-set contents."""
    required = {"schema_version", "distribution", "seed", "cases", "case_set_id"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError(f"case set must contain exactly {sorted(required)}")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported case-set schema_version")
    if data["distribution"] != DISTRIBUTION:
        raise ValueError(f"distribution must be {DISTRIBUTION!r}")
    _integer(data["seed"], "seed")
    cases = data["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    case_ids, seeds = set(), set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"case_id", "seed"}:
            raise ValueError(f"case {index} must contain exactly case_id and seed")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case {index} has an invalid case_id")
        seed = _integer(case["seed"], f"case {case_id} seed")
        if seed >= 2**31:
            raise ValueError(f"case {case_id} seed must be below 2**31")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        if seed in seeds:
            raise ValueError(f"duplicate reset seed: {seed}")
        case_ids.add(case_id)
        seeds.add(seed)
    if data["case_set_id"] != case_set_sha256(data):
        raise ValueError("case_set_id checksum does not match the case-set contents")
    return data


def make_case_set(trials: int = 128, seed: int = 20261002) -> dict:
    """Create reproducible, unique per-trial reset seeds for the frozen task."""
    _integer(trials, "trials", 1)
    _integer(seed, "seed")
    if trials > 2**31:
        raise ValueError("trials exceeds the available unique reset seeds")
    seeds = random.Random(seed).sample(range(2**31), trials)
    data = {
        "schema_version": SCHEMA_VERSION,
        "distribution": DISTRIBUTION,
        "seed": seed,
        "cases": [
            {"case_id": f"case_{index:06d}", "seed": case_seed}
            for index, case_seed in enumerate(seeds)
        ],
    }
    data["case_set_id"] = case_set_sha256(data)
    return validate_case_set(data)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate JSON key: {key}")
        data[key] = value
    return data


def read_json(path: str | Path) -> Any:
    """Read JSON without accepting NaN, Infinity or duplicate object fields."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def write_json_new(path: str | Path, data: Any) -> None:
    """Write a new JSON artifact, refusing to replace an existing file."""
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def read_case_set(path: str | Path) -> dict:
    return validate_case_set(read_json(path))


def write_case_set(path: str | Path, data: dict) -> None:
    write_json_new(path, validate_case_set(data))


def _finite_json(value: Any, path: str) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} has non-string keys")
        for key, child in value.items():
            _finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}[{index}]")
    elif type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite number")
    elif value is not None and not isinstance(value, (str, bool)):
        raise ValueError(f"{path} contains an unsupported value")


def _numeric_state(value: Any, path: str) -> None:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{path} is empty")
        for key, child in value.items():
            _numeric_state(child, f"{path}.{key}")
    elif isinstance(value, list):
        if not value:
            raise ValueError(f"{path} is empty")
        for index, child in enumerate(value):
            _numeric_state(child, f"{path}[{index}]")
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{path} must contain finite numeric state values")


def _compare_state(reference: Any, candidate: Any, path: str, atol: float) -> None:
    if isinstance(reference, dict):
        if not isinstance(candidate, dict) or set(reference) != set(candidate):
            raise ValueError(f"initial-state keys differ at {path}")
        for key in reference:
            _compare_state(reference[key], candidate[key], f"{path}.{key}", atol)
    elif isinstance(reference, list):
        if not isinstance(candidate, list) or len(reference) != len(candidate):
            raise ValueError(f"initial-state shape differs at {path}")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _compare_state(left, right, f"{path}[{index}]", atol)
    elif type(candidate) not in (int, float) or not math.isclose(
        reference, candidate, rel_tol=0.0, abs_tol=atol
    ):
        raise ValueError(f"initial-state mismatch at {path}: {reference!r} vs {candidate!r}")


def _validate_protocol(metadata: dict, run_dir: Path) -> tuple[EvaluationCriteria, int]:
    protocol = metadata.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError(f"{run_dir}: missing embedded protocol")
    digest = hashlib.sha256(_canonical_bytes(protocol)).hexdigest()
    if metadata["protocol_id"] != digest:
        raise ValueError(f"{run_dir}: protocol_id checksum does not match embedded protocol")
    if metadata["criteria"] != protocol.get("criteria"):
        raise ValueError(f"{run_dir}: metadata criteria differ from embedded protocol criteria")
    expected_keys = {"dt_s", "hold_duration_s", "xy_tolerance_m", "depth_tolerance_m"}
    if set(metadata["criteria"]) != expected_keys:
        raise ValueError(f"{run_dir}: protocol criteria must contain exactly {sorted(expected_keys)}")
    if any(type(value) not in (int, float) for value in metadata["criteria"].values()):
        raise ValueError(f"{run_dir}: protocol criteria must contain numeric values")
    criteria = EvaluationCriteria(**metadata["criteria"])
    horizon_steps = _integer(protocol.get("horizon_steps"), f"{run_dir} horizon_steps", 1)
    if horizon_steps < criteria.required_hold_samples:
        raise ValueError(f"{run_dir}: protocol horizon is shorter than the hold requirement")
    if not math.isfinite(horizon_steps * criteria.dt_s):
        raise ValueError(f"{run_dir}: protocol horizon duration must be finite")
    if not isinstance(protocol.get("shared_environment_config"), dict) or not protocol["shared_environment_config"]:
        raise ValueError(f"{run_dir}: protocol must record the shared environment configuration")
    return criteria, horizon_steps


def _validate_run(run_dir: Path) -> tuple[dict, dict[str, dict]]:
    metadata = read_json(run_dir / "metadata.json")
    trials = read_json(run_dir / "trials.json")
    if not isinstance(metadata, dict):
        raise ValueError(f"{run_dir}: metadata must be an object")
    _finite_json(metadata, f"{run_dir}.metadata")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError(f"{run_dir}: unsupported run schema_version")
    if metadata.get("method") not in METHODS:
        raise ValueError(f"{run_dir}: invalid method")
    for key in ("case_set_id", "protocol_id", "code_revision"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"{run_dir}: missing or invalid {key}")
    for key in ("runtime_versions", "criteria"):
        if not isinstance(metadata.get(key), dict) or not metadata[key]:
            raise ValueError(f"{run_dir}: missing or empty {key}")
    criteria, horizon_steps = _validate_protocol(metadata, run_dir)
    case_set = validate_case_set(metadata.get("case_set"))
    if metadata["case_set_id"] != case_set["case_set_id"]:
        raise ValueError(f"{run_dir}: embedded case_set does not match case_set_id")
    trial_count = _integer(metadata.get("trial_count"), f"{run_dir} trial_count", 1)
    if trial_count != len(case_set["cases"]):
        raise ValueError(f"{run_dir}: trial_count differs from the frozen case set")
    if metadata.get("run_complete") is not True:
        raise ValueError(f"{run_dir}: run_complete must be true")
    checkpoint_hash = metadata.get("checkpoint_sha256")
    if metadata["method"] == "ppo" and (
        not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_hash)
    ):
        raise ValueError(f"{run_dir}: PPO requires a lowercase SHA256 checkpoint digest")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{run_dir}: trials must be a nonempty list")
    expected = {case["case_id"]: case["seed"] for case in case_set["cases"]}
    indexed = {}
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError(f"{run_dir}: each trial must be an object")
        _finite_json(trial, f"{run_dir}.trial")
        case_id = trial.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected:
            raise ValueError(f"{run_dir}: unknown or missing case_id {case_id!r}")
        if case_id in indexed:
            raise ValueError(f"{run_dir}: duplicate trial case_id {case_id}")
        reset_seed = _integer(trial.get("reset_seed"), f"{run_dir}/{case_id} reset_seed")
        if reset_seed != expected[case_id]:
            raise ValueError(f"{run_dir}/{case_id}: reset_seed differs from the frozen case set")
        state = trial.get("initial_state")
        if not isinstance(state, dict) or not INITIAL_STATE_KEYS.issubset(state):
            raise ValueError(f"{run_dir}/{case_id}: initial_state lacks required physical state")
        _numeric_state(state, f"{run_dir}/{case_id}.initial_state")
        if type(trial.get("held_success")) is not bool:
            raise ValueError(f"{run_dir}/{case_id}: held_success must be a boolean")
        if not TRIAL_METRIC_KEYS.issubset(trial):
            raise ValueError(f"{run_dir}/{case_id}: missing trial metrics")
        if type(trial["sample_count"]) is not int or trial["sample_count"] != horizon_steps:
            raise ValueError(f"{run_dir}/{case_id}: sample_count differs from the protocol horizon")
        elapsed = trial["elapsed_time_s"]
        if type(elapsed) not in (int, float) or not math.isclose(
            elapsed, horizon_steps * criteria.dt_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"{run_dir}/{case_id}: elapsed_time_s differs from the protocol horizon")
        if type(trial["task_episode_reward"]) not in (int, float):
            raise ValueError(f"{run_dir}/{case_id}: task_episode_reward must be numeric")
        indexed[case_id] = trial
    if set(indexed) != set(expected):
        missing = sorted(set(expected) - set(indexed))
        raise ValueError(f"{run_dir}: incomplete run; missing cases: {missing[:5]}")
    return metadata, indexed


def compare_runs(run_dirs: list[str | Path], atol: float = 1e-6) -> dict:
    """Summarize only complete runs with matched protocol and physical resets.

    ``atol`` applies to every recorded numeric state component, with no relative
    tolerance. Passing this gate establishes recorded initial-state agreement;
    it does not establish identical future GPU physics trajectories.
    """
    if type(atol) not in (int, float) or not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite nonnegative number")
    if len(run_dirs) < 2:
        raise ValueError("comparison requires at least two method runs")
    runs = [(Path(path), *_validate_run(Path(path))) for path in run_dirs]
    methods = [metadata["method"] for _, metadata, _ in runs]
    if len(set(methods)) != len(methods):
        raise ValueError("comparison contains duplicate method runs")
    _, reference_meta, reference_rows = runs[0]
    shared = ("case_set_id", "protocol_id", "code_revision", "runtime_versions", "criteria")
    for run_dir, metadata, rows in runs[1:]:
        for key in shared:
            if metadata[key] != reference_meta[key]:
                raise ValueError(f"{run_dir}: {key} differs between runs")
        if set(rows) != set(reference_rows):
            raise ValueError(f"{run_dir}: trial case IDs differ between runs")
        for case_id, reference in reference_rows.items():
            if reference["reset_seed"] != rows[case_id]["reset_seed"]:
                raise ValueError(f"{run_dir}/{case_id}: reset seeds differ between runs")
            _compare_state(
                reference["initial_state"], rows[case_id]["initial_state"],
                f"{run_dir}/{case_id}", atol,
            )
    summary = {}
    for _, metadata, rows in runs:
        count = len(rows)
        successes = sum(row["held_success"] for row in rows.values())
        summary[metadata["method"]] = {
            **summarize_trials(rows.values()),
            "held_successes": successes,
            "held_success_rate": successes / count,
            "mean_task_episode_reward": sum(row["task_episode_reward"] / count for row in rows.values()),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_valid": True,
        "case_set_id": reference_meta["case_set_id"],
        "protocol_id": reference_meta["protocol_id"],
        "code_revision": reference_meta["code_revision"],
        "runtime_versions": reference_meta["runtime_versions"],
        "criteria": reference_meta["criteria"],
        "initial_state_atol": atol,
        "initial_state_rtol": 0.0,
        "methods": summary,
    }
