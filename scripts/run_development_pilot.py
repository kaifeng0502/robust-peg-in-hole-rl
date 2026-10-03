"""Run the predeclared development pilot after reference and runtime checks.

Launch from an activated AutoDL runtime in an isolated candidate source tree.
All policy evaluation uses the separate frozen reference checkout. Long stages
have bounded deadlines and write status and logs to a fresh output directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from evaluate_insertion import source_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wait_for_exit(path, seconds):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(10)
    if path.read_text().strip() != "0":
        raise RuntimeError(f"Prerequisite failed: {path}: {path.read_text().strip()}")


def verify_source(frozen, revision):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=frozen, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=frozen, text=True,
    )
    if not head.startswith(revision) or dirty.strip():
        raise ValueError("Reference checkout must be clean at the frozen revision")


def run_parallel(commands, seconds, output, status):
    processes, streams = [], []
    try:
        for label, command, directory in commands:
            stream = (output / f"{label}.log").open("x")
            streams.append(stream)
            process = subprocess.Popen(
                command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((label, process))
        status["processes"] = {label: process.pid for label, process in processes}
        write_json(output / "status.json", status)
        deadline = time.monotonic() + seconds
        while True:
            codes = {label: process.poll() for label, process in processes}
            failures = {label: code for label, code in codes.items() if code not in (None, 0)}
            if failures:
                raise RuntimeError(f"Stage subprocess failed: {failures}")
            if all(code == 0 for code in codes.values()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Stage exceeded {seconds} s")
            time.sleep(10)
    finally:
        for _, process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--contract-check", type=Path, required=True, help="Simulator check JSON; .exit sibling required")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"started_utc": datetime.now(timezone.utc).isoformat(), "complete": False}
    plan = json.loads((PROJECT_ROOT / "configs/held_v2_pilot.json").read_text())
    write_json(args.output / "plan.json", plan)
    candidate_revision = source_fingerprint()
    status["candidate_code_revision"] = candidate_revision

    def phase(name):
        status.update(phase=name, phase_started_utc=datetime.now(timezone.utc).isoformat(), processes={})
        write_json(args.output / "status.json", status)
        print(name, flush=True)

    try:
        frozen = args.runtime_root / "baseline_rl"
        if frozen.resolve() == PROJECT_ROOT:
            raise ValueError("Candidate and frozen evaluator must use distinct source directories")
        verify_source(frozen, plan["evaluation_source_commit"])
        checkpoint = frozen / "results/recovered_continue_20260917/nn/LocalInsertion.pth"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan["checkpoint_sha256"]:
            raise ValueError("Starting checkpoint hash mismatch")
        if json.loads((frozen / plan["development_manifest"]).read_text())["case_set_id"] != plan["development_case_set_id"]:
            raise ValueError("Development manifest mismatch")
        phase("waiting_for_runtime_contract")
        wait_for_exit(args.contract_check.with_suffix(".exit"), 1800)
        contract = json.loads(args.contract_check.read_text())
        if not contract.get("passed"):
            raise RuntimeError("Runtime contract did not pass")
        if contract.get("code_revision") != candidate_revision:
            raise RuntimeError("Runtime contract is stale for this candidate source")
        if not any(row["terminal_metrics"]["terminal_held_success"] > 0 for row in contract["episodes"]):
            raise RuntimeError("Runtime check did not exercise a positive held-success/reset path")
        shutil.copy2(args.contract_check, args.output / "training_contract.json")
        phase("waiting_for_reference_evaluation")
        wait_for_exit(args.reference_results / "exit_code.txt", 14400)
        reference_runs = [args.reference_results / name for name in ("zero_residual", "spiral", "ppo")]
        reference_metadata = json.loads((reference_runs[0] / "metadata.json").read_text())
        if contract["runtime_versions"] != reference_metadata["runtime_versions"]:
            raise RuntimeError("Runtime contract and reference evaluation used different runtimes")
        phase("reference_diagnosis")
        subprocess.run([
            sys.executable, "scripts/analyze_development.py", "--runs", *map(str, reference_runs),
            "--output", str(args.output / "reference_diagnosis.json"),
        ], cwd=PROJECT_ROOT, check=True)
        phase("training_continuation_arms")
        if source_fingerprint() != candidate_revision:
            raise RuntimeError("Candidate source changed while waiting")
        commands, train_dirs = [], {}
        for arm in plan["arms"]:
            name = arm["name"]
            run_name = f"{plan['experiment_id']}_{name}"
            train_dir = PROJECT_ROOT / "logs/rl_games/LocalInsertion" / run_name
            if train_dir.exists():
                raise FileExistsError(train_dir)
            train_dirs[name] = train_dir
            commands.append((f"train_{name}", [
                sys.executable, "scripts/train_rl.py", "--task", "Isaac-LocalInsertion-RL-Direct-v0",
                "--num_envs", str(plan["num_envs"]), "--seed", str(plan["training_seed"]), "--headless",
                "--checkpoint", str(checkpoint), "--max_iterations", str(plan["stop_epoch"]),
                f"env.episode_length_s={plan['training_episode_length_s']}",
                f"env.reward.success_contract={arm['success_contract']}",
                f"env.reward.hold_duration_s={arm.get('hold_duration_s', 1.0)}",
                f"env.reward.terminal_hold_bonus={arm.get('terminal_hold_bonus', 100.0)}",
                f"agent.params.config.horizon_length={plan['rollout_steps']}",
                f"+agent.params.config.full_experiment_name={run_name}",
            ], PROJECT_ROOT))
        run_parallel(commands, 7200, args.output, status)
        phase("verify_training_budget")
        if source_fingerprint() != candidate_revision:
            raise RuntimeError("Candidate source changed during training")
        import torch
        import yaml

        original = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if original["epoch"] != plan["starting_checkpoint_epoch"]:
            raise ValueError("Starting epoch mismatch")
        models, training_records, saved_envs, saved_agents = {}, {}, [], []
        for name, train_dir in train_dirs.items():
            arm = next(value for value in plan["arms"] if value["name"] == name)
            # BaseLoader reads the environment's tagged Python symbols as data,
            # never imports or executes the serialized callables.
            saved_env = yaml.load((train_dir / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
            saved_agent = yaml.safe_load((train_dir / "params/agent.yaml").read_text())
            learning = saved_agent["params"]["config"]
            if (int(saved_env["scene"]["num_envs"]) != plan["num_envs"]
                    or int(saved_env["seed"]) != plan["training_seed"]
                    or float(saved_env["episode_length_s"]) != plan["training_episode_length_s"]
                    or saved_env["reward"]["success_contract"] != arm["success_contract"]
                    or float(saved_env["reward"]["hold_duration_s"]) != arm.get("hold_duration_s", 1.0)
                    or float(saved_env["reward"]["terminal_hold_bonus"]) != arm.get("terminal_hold_bonus", 100.0)
                    or learning["horizon_length"] != plan["rollout_steps"]
                    or learning["max_epochs"] != plan["stop_epoch"]
                    or saved_agent["params"]["seed"] != plan["training_seed"]):
                raise RuntimeError(f"Saved training configuration differs from plan: {name}")
            saved_env["reward"]["success_contract"] = "shared_for_comparison"
            learning.pop("full_experiment_name")
            saved_envs.append(saved_env)
            saved_agents.append(saved_agent)
            found = list((train_dir / "nn").glob(f"last_LocalInsertion_ep_{plan['stop_epoch']}_rew_*.pth"))
            if len(found) != 1:
                raise RuntimeError(f"Expected one fixed final checkpoint for {name}, got {found}")
            model = torch.load(found[0], map_location="cpu", weights_only=False)
            transitions = int(model["frame"]) - int(original["frame"])
            if int(model["epoch"]) != plan["stop_epoch"] or transitions != plan["additional_transitions_per_arm"]:
                raise RuntimeError(f"Training budget mismatch: {name}, {model['epoch']}, {transitions}")
            models[name] = found[0]
            training_records[name] = {
                "checkpoint": str(found[0]), "sha256": hashlib.sha256(found[0].read_bytes()).hexdigest(),
                "epoch": int(model["epoch"]), "added_transitions": transitions,
                "inherited_frame_counter": int(original["frame"]), "final_frame_counter": int(model["frame"]),
            }
            shutil.copytree(train_dir, args.output / "training" / name)
        if any(value != saved_envs[0] for value in saved_envs[1:]) or any(
            value != saved_agents[0] for value in saved_agents[1:]
        ):
            raise RuntimeError("Continuation arms differ beyond the declared reward profile and run name")
        write_json(args.output / "training_budget.json", training_records)
        phase("evaluating_candidates_on_frozen_source")
        verify_source(frozen, plan["evaluation_source_commit"])
        commands = []
        for name, model in models.items():
            commands.append((f"evaluate_{name}", [
                sys.executable, "scripts/evaluate_insertion.py", "--method", "ppo", "--headless",
                "--cases", str(frozen / plan["development_manifest"]),
                "--checkpoint", str(model), "--agent-config", str(train_dirs[name] / "params/agent.yaml"),
                "--output", str(args.output / name),
            ], frozen))
        run_parallel(commands, 10800, args.output, status)
        phase("paired_candidate_diagnosis")
        subprocess.run([
            sys.executable, "scripts/compare_ppo_candidates.py", "--reference", str(reference_runs[0]),
            "--runs", str(reference_runs[2]), *[str(args.output / name) for name in models],
            "--labels", "recovered", *models,
            "--output", str(args.output / "candidate_comparison.json"),
        ], cwd=PROJECT_ROOT, check=True)
        status.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
        phase("complete")
        (args.output / "exit_code.txt").write_text("0\n")
    except BaseException as exc:
        status.update(error=f"{type(exc).__name__}: {exc}", complete=False)
        write_json(args.output / "status.json", status)
        (args.output / "exit_code.txt").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
