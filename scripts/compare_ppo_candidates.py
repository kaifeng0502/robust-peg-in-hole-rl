"""Compare PPO candidates through an unchanged, common baseline reference.

Each candidate independently passes the existing matched-run comparison gate.
The original metadata and the evaluator's unique-method restriction remain
unchanged. These are development comparisons, not independent training trials.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from analyze_development import _outcomes, _paired, _stats, verify_traces
from evaluation_cases import _compare_state, compare_runs, read_json, write_json_new


def _digest(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _paired_rewards(left: dict, right: dict) -> dict:
    return {
        **_paired(left, right),
        "left_minus_right_task_episode_reward_all_cases": _stats([
            left[case]["task_episode_reward"] - right[case]["task_episode_reward"] for case in left
        ]),
        "left_minus_right_regularized_episode_reward_all_cases": _stats([
            left[case]["episode_reward"] - right[case]["episode_reward"] for case in left
        ]),
    }


def compare_candidates(reference: Path, runs: list[Path], labels: list[str]) -> dict:
    """Verify full runs before comparing candidates without relabeling methods."""
    if len(runs) < 2:
        raise ValueError("At least two PPO candidate runs are required")
    if len(labels) != len(runs):
        raise ValueError("Exactly one label is required per candidate run")
    if any(not isinstance(label, str) or not label.strip() or label != label.strip() for label in labels):
        raise ValueError("Labels must be nonempty strings without surrounding whitespace")
    if len(set(labels)) != len(labels):
        raise ValueError("Candidate labels must be unique")
    reference_meta = read_json(reference / "metadata.json")
    if reference_meta.get("method") not in ("zero_residual", "spiral"):
        raise ValueError("Reference method must be zero_residual or spiral")
    reference_trials = read_json(reference / "trials.json")
    reference_rows = {row["case_id"]: row for row in reference_trials}
    candidates, trial_maps, checkpoint_groups, config_groups = {}, {}, {}, {}
    shared_gate = None
    for label, run in zip(labels, runs):
        metadata = read_json(run / "metadata.json")
        if metadata.get("method") != "ppo":
            raise ValueError(f"{label}: candidate method must be ppo")
        checkpoint = _digest(metadata.get("checkpoint_sha256"), f"{label}.checkpoint_sha256")
        agent_config = _digest(metadata.get("agent_config_sha256"), f"{label}.agent_config_sha256")
        if metadata.get("deterministic_policy") is not True:
            raise ValueError(f"{label}: deterministic_policy must be true")
        # Do not alter metadata.method to make compare_runs accept duplicate PPOs.
        gate = compare_runs([reference, run])
        if shared_gate is None:
            shared_gate = {key: value for key, value in gate.items() if key != "methods"}
        elif shared_gate != {key: value for key, value in gate.items() if key != "methods"}:
            raise ValueError(f"{label}: comparison gate differs across candidates")
        trials = read_json(run / "trials.json")
        trace_rows = verify_traces(run, metadata, trials)
        indexed = {row["case_id"]: row for row in trials}
        trial_maps[label] = indexed
        candidates[label] = {
            "method": "ppo",
            "run_directory": str(run),
            "checkpoint_sha256": checkpoint,
            "agent_config_sha256": agent_config,
            "deterministic_policy": True,
            "comparison_to_reference_valid": True,
            "verified_trace_rows": trace_rows,
            "outcomes": {
                **_outcomes(trials),
                "regularized_episode_reward": _stats([row["episode_reward"] for row in trials]),
            },
            "candidate_minus_reference": _paired_rewards(indexed, reference_rows),
        }
        checkpoint_groups.setdefault(checkpoint, []).append(label)
        config_groups.setdefault(agent_config, []).append(label)
    reference_trace_rows = verify_traces(reference, reference_meta, reference_trials)
    paired = []
    for left, right in itertools.combinations(labels, 2):
        # Closeness to one reference does not imply pairwise closeness: enforce
        # the original tolerance again, rather than silently allowing 2 * atol.
        for case in reference_rows:
            _compare_state(
                trial_maps[left][case]["initial_state"], trial_maps[right][case]["initial_state"],
                f"{left}_vs_{right}/{case}", shared_gate["initial_state_atol"],
            )
        paired.append({
            "left": left,
            "right": right,
            "same_checkpoint_sha256": (
                candidates[left]["checkpoint_sha256"] == candidates[right]["checkpoint_sha256"]
            ),
            "same_agent_config_sha256": (
                candidates[left]["agent_config_sha256"] == candidates[right]["agent_config_sha256"]
            ),
            "pairwise_initial_state_match": True,
            **_paired_rewards(trial_maps[left], trial_maps[right]),
        })
    return {
        "schema_version": 1,
        "purpose": "Matched development comparison of retained PPO candidates; not final holdout performance.",
        "shared_comparison_gate": shared_gate,
        "reference": {
            "method": reference_meta["method"],
            "run_directory": str(reference),
            "verified_trace_rows": reference_trace_rows,
            "outcomes": _outcomes(reference_trials),
        },
        "candidates": candidates,
        "paired_candidates": paired,
        "repeated_checkpoint_groups": [group for group in checkpoint_groups.values() if len(group) > 1],
        "shared_agent_config_groups": [group for group in config_groups.values() if len(group) > 1],
        "interpretation_limits": [
            "Every PPO passed compare_runs against the same unchanged baseline reference, and full traces were reconstructed.",
            "Candidate pairs additionally satisfy the same absolute initial-state tolerance; relative tolerance remains zero.",
            "Repeated checkpoint hashes identify the same model artifact, not independently trained policies. Different hashes alone also do not establish independent training seeds.",
            "The same evaluation cases are reused across candidates; do not pool them as additional independent cases.",
            "Success-time differences use only cases successful for both candidates; force and return differences use all matched cases.",
            "Force is generated impedance command magnitude, not measured contact force.",
            "Task return removes residual/action-rate penalties. Regularized return alone must not rank controller quality.",
            "Development-set selection can overfit these cases. An untouched final evaluation is required after selection.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON output path; never overwrites.")
    args = parser.parse_args()
    try:
        report = compare_candidates(args.reference, args.runs, args.labels)
        write_json_new(args.output, report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(args.output),
        "successes": {label: row["outcomes"]["held_successes"] for label, row in report["candidates"].items()},
        "repeated_checkpoint_groups": report["repeated_checkpoint_groups"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
