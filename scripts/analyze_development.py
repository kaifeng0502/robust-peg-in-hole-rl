"""Diagnose matched insertion runs using fixed strata and verified control traces.

This offline report describes observed geometry and hold failures. It cannot
identify their physical cause or establish generalization after development-set
tuning. Aggregates are reconstructed from control-rate rows and recorded interval
flags, not from raw 120 Hz physics samples. Only the standard library is required;
no simulator is started.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_insertion"))

from evaluation_cases import compare_runs, read_json, write_json_new  # noqa: E402
from evaluation_metrics import EpisodeMetrics, EvaluationCriteria, wilson95  # noqa: E402


# Fixed before inspecting the 64-case development outcomes. Never infer edges
# from observed successes, quantiles, or the evaluated sample size.
STRATA = {
    "initial_root_xy_mm": (
        (0.0, 2.5, False), (2.5, 5.0, False), (5.0, 10.0, False), (10.0, None, False),
    ),
    "initial_axis_tilt_deg": (
        (0.0, 1.0, False), (1.0, 3.0, False), (3.0, 5.0, False), (5.0, None, False),
    ),
    "friction": (
        (0.0, 0.6, False), (0.6, 0.9, False), (0.9, 1.2, True), (1.2, None, False),
    ),
}


def _number(value, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _reject_trace_constant(value: str):
    raise ValueError(f"Trace JSON contains a nonfinite constant: {value}")


def _unique_trace_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Trace JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def _parse_trace(line: str) -> dict:
    return json.loads(
        line,
        parse_constant=_reject_trace_constant,
        object_pairs_hook=_unique_trace_object,
        # JSON exponents can overflow without using a NaN/Infinity literal.
        parse_float=lambda value: _number(float(value), "Trace JSON number"),
    )


def _z_axis(quaternion: list[float]) -> tuple[float, float, float]:
    """Rotate local +Z by an Isaac wxyz quaternion, normalizing roundoff."""
    if not isinstance(quaternion, list) or len(quaternion) != 4:
        raise ValueError("Expected a four-component wxyz quaternion")
    values = [_number(value, "quaternion") for value in quaternion]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion norm must be finite and nonzero")
    w, x, y, z = (value / norm for value in values)
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def initial_features(state: dict) -> dict:
    held, fixed = state["held_root_state"], state["fixed_root_state"]
    if not all(isinstance(root, list) and len(root) >= 7 for root in (held, fixed)):
        raise ValueError("Initial root states must contain position and wxyz quaternion")
    held_axis, fixed_axis = _z_axis(held[3:7]), _z_axis(fixed[3:7])
    dot = sum(a * b for a, b in zip(held_axis, fixed_axis))
    friction = _number(state["episode_friction"], "episode_friction")
    if friction < 0:
        raise ValueError("episode_friction must be nonnegative")
    return {
        "initial_root_xy_mm": 1000 * math.hypot(
            _number(held[0], "held x") - _number(fixed[0], "fixed x"),
            _number(held[1], "held y") - _number(fixed[1], "fixed y"),
        ),
        "initial_axis_tilt_deg": math.degrees(math.acos(min(1.0, max(-1.0, dot)))),
        "friction": friction,
    }


def bin_index(feature: str, value: float) -> int:
    value = _number(value, feature)
    if value < 0:
        raise ValueError(f"{feature} must be nonnegative")
    for index, (lower, upper, inclusive) in enumerate(STRATA[feature]):
        if value >= lower and (upper is None or value < upper or (inclusive and value == upper)):
            return index
    raise ValueError(f"No fixed bin for {feature}: {value}")


def _same_metric(actual, expected, label: str) -> None:
    if type(expected) in (int, float):
        if not math.isclose(_number(actual, label), expected, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"{label}: trial/trace metric mismatch: {actual!r} != {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label}: trial/trace metric mismatch: {actual!r} != {expected!r}")


def verify_traces(run: Path, metadata: dict, trials: list[dict]) -> int:
    """Validate control traces and reconstruct aggregates using recorded flags.

    The interval flags summarize physics substeps recorded by the simulator.
    Raw 120 Hz geometry is absent here, so this checks their consistency rather
    than independently re-evaluating success at each physics substep.
    """
    criteria = EvaluationCriteria(**metadata["criteria"])
    indexed = {trial["case_id"]: trial for trial in trials}
    metrics = {
        case: EpisodeMetrics(criteria, row["target_depth_m"]) for case, row in indexed.items()
    }
    task_returns = dict.fromkeys(indexed, 0.0)
    with (run / "trajectories.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            sample = _parse_trace(line)
            if (
                not isinstance(sample, dict)
                or not isinstance(sample.get("case_id"), str)
                or sample["case_id"] not in indexed
            ):
                raise ValueError(f"{run}: trace line {line_number} has an unknown case")
            case = sample["case_id"]
            accumulator = metrics[case]
            step = accumulator.sample_count + 1
            if type(sample.get("step")) is not int or sample["step"] != step:
                raise ValueError(f"{run}/{case}: trace steps must be complete and ordered")
            if step > metadata["protocol"]["horizon_steps"]:
                raise ValueError(f"{run}/{case}: trace exceeds the scored horizon")
            _same_metric(sample["time_s"], step * criteria.dt_s, f"{case}.time_s")
            _same_metric(sample["target_depth_m"], accumulator.target_depth_m, f"{case}.target_depth_m")
            for field in ("xy_error_m", "depth_m", "reward", "task_reward", "commanded_force_n"):
                _number(sample.get(field), f"{run}/{case}.{field}")
            action = sample.get("action")
            # All controllers use the frozen six-dimensional clipped action API.
            if not isinstance(action, list) or len(action) != 6:
                raise ValueError(f"{run}/{case}: action must contain six numbers")
            for component in action:
                if not -1.0 <= _number(component, f"{run}/{case}.action") <= 1.0:
                    raise ValueError(f"{run}/{case}: action must lie within [-1, 1]")
            for flag in ("interval_success", "interval_ever_success"):
                if type(sample.get(flag)) is not bool:
                    raise ValueError(f"{run}/{case}: {flag} must be boolean")
            if sample["interval_success"] and not sample["interval_ever_success"]:
                raise ValueError(f"{run}/{case}: interval_success requires interval_ever_success")
            accumulator.update(
                sample["xy_error_m"], sample["depth_m"], sample["reward"], sample["commanded_force_n"],
                interval_success=sample["interval_success"],
                interval_ever_success=sample["interval_ever_success"],
            )
            current = accumulator.result()
            for flag in ("final_success", "held_success"):
                _same_metric(sample[flag], current[flag], f"{case}.{flag}")
            task_returns[case] += _number(sample["task_reward"], "task_reward")
    for case, accumulator in metrics.items():
        if accumulator.sample_count != indexed[case]["sample_count"]:
            raise ValueError(f"{run}/{case}: incomplete trace")
        reconstructed = {**accumulator.result(), "task_episode_reward": task_returns[case]}
        for key, expected in reconstructed.items():
            if key not in indexed[case]:
                raise ValueError(f"{run}/{case}: missing trial metric {key}")
            _same_metric(indexed[case][key], expected, f"{run}/{case}.{key}")
    return sum(accumulator.sample_count for accumulator in metrics.values())


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _outcomes(rows: list[dict]) -> dict:
    count = len(rows)
    successes = sum(row["held_success"] for row in rows)
    return {
        "cases": count,
        "held_successes": successes,
        "held_success_rate": successes / count if count else None,
        "wilson95": list(wilson95(successes, count)) if count else None,
        "failure_categories": dict(sorted(Counter(
            row["failure_category"] for row in rows if not row["held_success"]
        ).items())),
        "terminal_xy_error_mm": _stats([1000 * row["final_xy_error_m"] for row in rows]),
        "terminal_depth_mm": _stats([1000 * row["final_depth_m"] for row in rows]),
        "terminal_absolute_depth_error_mm": _stats([
            1000 * abs(row["final_depth_m"] - row["target_depth_m"]) for row in rows
        ]),
        "peak_commanded_force_n": _stats([row["peak_commanded_force_n"] for row in rows]),
        "task_episode_reward": _stats([row["task_episode_reward"] for row in rows]),
        "successful_completion_time_s": _stats([
            row["success_time_s"] for row in rows if row["held_success"]
        ]),
        "case_ids": [row["case_id"] for row in rows],
    }


def _paired(left: dict[str, dict], right: dict[str, dict]) -> dict:
    groups = {"both_success": [], "left_only_success": [], "right_only_success": [], "both_failure": []}
    for case in left:
        l_success, r_success = left[case]["held_success"], right[case]["held_success"]
        key = ("both_success" if r_success else "left_only_success") if l_success else (
            "right_only_success" if r_success else "both_failure"
        )
        groups[key].append(case)
    return {
        "case_ids": groups,
        "counts": {key: len(value) for key, value in groups.items()},
        "left_minus_right_success_rate": (
            len(groups["left_only_success"]) - len(groups["right_only_success"])
        ) / len(left),
        "left_minus_right_success_time_s_on_both_success": _stats([
            left[case]["success_time_s"] - right[case]["success_time_s"] for case in groups["both_success"]
        ]),
        "left_minus_right_terminal_xy_mm_all_cases": _stats([
            1000 * (left[case]["final_xy_error_m"] - right[case]["final_xy_error_m"]) for case in left
        ]),
        "left_minus_right_peak_commanded_force_n_all_cases": _stats([
            left[case]["peak_commanded_force_n"] - right[case]["peak_commanded_force_n"] for case in left
        ]),
    }


def analyze_runs(run_dirs: list[Path]) -> dict:
    comparison = compare_runs(run_dirs)  # Keep the fixed 1e-6 / rtol=0 gate.
    runs, trace_counts = {}, {}
    for run in run_dirs:
        metadata = read_json(run / "metadata.json")
        trials = read_json(run / "trials.json")
        method = metadata["method"]
        trace_counts[method] = verify_traces(run, metadata, trials)
        runs[method] = {trial["case_id"]: trial for trial in trials}
    reference = next(iter(runs.values()))
    features = {case: initial_features(row["initial_state"]) for case, row in reference.items()}
    methods = {}
    for method, indexed in runs.items():
        rows = list(indexed.values())
        strata = {}
        for feature, bins in STRATA.items():
            strata[feature] = [
                {
                    "lower": lower,
                    "upper": upper,
                    "upper_inclusive": inclusive,
                    "lower_inclusive": not (feature == "friction" and index == 3),
                    **_outcomes([
                        row for row in rows if bin_index(feature, features[row["case_id"]][feature]) == index
                    ]),
                }
                for index, (lower, upper, inclusive) in enumerate(bins)
            ]
        methods[method] = {
            "overall": _outcomes(rows),
            "by_failure_category": {
                category: _outcomes([row for row in rows if row["failure_category"] == category])
                for category in sorted({row["failure_category"] for row in rows if not row["held_success"]})
            },
            "strata": strata,
        }
    return {
        "schema_version": 1,
        "strata_version": "fixed_physical_intervals_v1",
        "purpose": "Exploratory development diagnosis; not a final holdout estimate.",
        "comparison": comparison,
        "verified_trace_rows": trace_counts,
        "feature_definitions": {
            "initial_root_xy_mm": "World XY distance between recorded held and fixed asset root origins at reset; not an observation error.",
            "initial_axis_tilt_deg": "Angle between each asset's local +Z axis from normalized wxyz root quaternions; excludes axial twist/yaw and is not full orientation error.",
            "friction": "Recorded episode coefficient applied to both static and dynamic material friction.",
        },
        "metric_definitions": {
            "terminal_depth_mm": "Signed insertion depth at the final control sample; negative means above the socket entrance.",
            "terminal_absolute_depth_error_mm": "Absolute difference between final depth and the recorded geometric target depth, in mm.",
            "terminal_xy_error_mm": "Final peg-base to target-base world XY distance from simulator geometry.",
            "peak_commanded_force_n": "Episode maximum of physics-substep commanded impedance force magnitudes; not measured contact force.",
        },
        "interpretation_limits": [
            "Trace verification recomputes aggregates from control-rate rows and recorded interval success flags; it does not re-evaluate raw 120 Hz physics substeps.",
            "Strata edges are fixed independently of the 64-case development outcomes; no quantile or success-dependent bins.",
            "Strata describe associations in small, overlapping case subsets, not causal effects of pose or friction.",
            "Failure categories describe terminal geometry and hold behavior; they do not diagnose jamming or sensor failure.",
            "Commanded force is the maximum generated impedance force over physics substeps, not sensor-measured contact force.",
            "Completion time is conditional on success; paired time differences use only cases successful for both methods.",
            "Wilson intervals are descriptive case proportions, not training-seed variation or protection against development-set selection.",
        ],
        "case_features": features,
        "case_outcomes": {
            case: {
                method: {
                    "held_success": indexed[case]["held_success"],
                    "failure_category": indexed[case]["failure_category"],
                    "terminal_xy_error_mm": 1000 * indexed[case]["final_xy_error_m"],
                    "terminal_depth_mm": 1000 * indexed[case]["final_depth_m"],
                    "terminal_absolute_depth_error_mm": 1000 * abs(
                        indexed[case]["final_depth_m"] - indexed[case]["target_depth_m"]
                    ),
                    "peak_commanded_force_n": indexed[case]["peak_commanded_force_n"],
                    "trailing_hold_duration_s": indexed[case]["trailing_hold_duration_s"],
                    "max_hold_duration_s": indexed[case]["max_hold_duration_s"],
                    "success_time_s": indexed[case]["success_time_s"],
                }
                for method, indexed in runs.items()
            }
            for case in reference
        },
        "methods": methods,
        "paired": {
            f"{left}_vs_{right}": {"left": left, "right": right, **_paired(runs[left], runs[right])}
            for left, right in itertools.combinations(sorted(runs), 2)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; never overwrites an existing report.")
    args = parser.parse_args()
    try:
        report = analyze_runs(args.runs)
        write_json_new(args.output, report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(args.output),
        "verified_trace_rows": report["verified_trace_rows"],
        "successes": {method: row["overall"]["held_successes"] for method, row in report["methods"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
