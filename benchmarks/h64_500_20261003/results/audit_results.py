"""Audit the epoch-621 four-case replay against the matched epoch-300 replay."""

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
BEFORE = HERE / "eval_ep300_run3"
AFTER = HERE / "final_four_cases"


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    metadata = load(AFTER / "metadata.json")
    summary = load(AFTER / "summary.json")
    trials = load(AFTER / "trials.json")
    previous = {trial["case_id"]: trial for trial in load(BEFORE / "trials.json")}
    trajectories = lines(AFTER / "trajectories.jsonl")
    diagnostics = lines(AFTER / "control_diagnostics.jsonl")
    budget = load(HERE / "training_budget.json")
    checkpoint = next((HERE / "training_backup/nn").glob("*ep_621*.pth"))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == budget["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert metadata["run_complete"] and metadata["trial_count"] == 4
    assert metadata["protocol"]["horizon_steps"] == 150
    assert metadata["criteria"]["hold_duration_s"] == 1.0
    cfg = metadata["protocol"]["shared_environment_config"]
    assert cfg["sim"]["dt"] == 1 / 120 and cfg["decimation"] == 8
    assert cfg["action_contract"] == "relative_z_v1"
    assert cfg["sim"]["physx"]["max_position_iteration_count"] == 192
    assert len(trials) == 4 and len(trajectories) == len(diagnostics) == 600
    assert finite([trials, trajectories, diagnostics])
    cases = []
    for trial in trials:
        case = trial["case_id"]
        before = previous[case]
        rows = [row for row in trajectories if row["case_id"] == case]
        diag = [row for row in diagnostics if row["seed"] == trial["reset_seed"]]
        assert [row["step"] for row in rows] == list(range(1, 151))
        assert [row["step"] for row in diag] == list(range(1, 151))
        assert all(len(row["substeps"]) == 8 for row in diag)
        assert all(abs(row["time_s"] - row["step"] / 15) < 1e-9 for row in rows)
        assert trial["sample_count"] == 150 and trial["elapsed_time_s"] == 10.0
        assert trial["initial_state"] == before["initial_state"]
        assert all(abs(sum(d["reward_terms"].values()) - r["reward"]) < 1e-4 for r, d in zip(rows, diag))
        held = all(row["interval_success"] for row in rows[-15:])
        assert held == trial["held_success"] == trial["success"]
        assert abs(rows[-1]["depth_m"] - trial["final_depth_m"]) < 1e-9
        assert abs(max(row["commanded_force_n"] for row in rows) - trial["peak_commanded_force_n"]) < 1e-9
        cases.append({
            "case_id": case,
            "before_held": before["held_success"],
            "after_held": held,
            "before_failure": before["failure_category"],
            "after_failure": trial["failure_category"],
            "before_final_xy_mm": before["final_xy_error_m"] * 1000,
            "after_final_xy_mm": trial["final_xy_error_m"] * 1000,
            "before_final_depth_mm": before["final_depth_m"] * 1000,
            "after_final_depth_mm": trial["final_depth_m"] * 1000,
            "before_first_hold_s": before["first_hold_time_s"],
            "after_first_hold_s": trial["first_hold_time_s"],
            "before_peak_commanded_force_n": before["peak_commanded_force_n"],
            "after_peak_commanded_force_n": trial["peak_commanded_force_n"],
            "before_task_return": before["task_episode_reward"],
            "after_task_return": trial["task_episode_reward"],
        })
    assert summary["held_success_count"] == sum(case["after_held"] for case in cases) == 2
    assert summary["failure_counts"] == {"misalignment": 2}
    report = {
        "passed": True,
        "scope": "Four reused development cases; not an independent success-rate estimate",
        "case_count": 4,
        "control_rows": 600,
        "physics_substeps": 4800,
        "initial_states_exactly_matched": True,
        "held_before": sum(case["before_held"] for case in cases),
        "held_after": sum(case["after_held"] for case in cases),
        "paired_wins": sum(not case["before_held"] and case["after_held"] for case in cases),
        "paired_losses": sum(case["before_held"] and not case["after_held"] for case in cases),
        "cases": cases,
        "artifact_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in AFTER.iterdir() if path.is_file()},
    }
    (HERE / "evaluation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["passed", "held_before", "held_after", "paired_wins", "paired_losses", "cases"]}, indent=2))


if __name__ == "__main__":
    main()
