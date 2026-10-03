"""Run one original-controller capacity screen with sampled device memory.

No PPO updates, training configuration edits, or evaluation resumption.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root
    project = root / "candidate_held_v2"
    output = root / "results/performance_20261003"
    sys.path.insert(0, str(project / "scripts"))
    from run_development_pilot import run_parallel, write_json

    previous = json.loads((output / "status.json").read_text())
    if previous["phase"] != "target_constants_full_step_screen":
        raise RuntimeError("Unexpected previous stage; refusing overlapping work")
    for identity in previous["processes"].values():
        pid = identity["pid"] if isinstance(identity, dict) else identity
        state_file = Path(f"/proc/{pid}/status")
        if state_file.exists() and not any(
            line.startswith("State:") and "Z" in line for line in state_file.read_text().splitlines()
        ):
            raise RuntimeError(f"Previous worker still exists: {pid}")
    if not json.loads((output / "target_constants_full_1/result.json").read_text())["passed"]:
        raise RuntimeError("Previous measurement did not finish successfully")
    for pid in (11545, 15330, 15331):
        if not any(line.startswith("State:") and "T" in line
                   for line in Path(f"/proc/{pid}/status").read_text().splitlines()):
            raise RuntimeError(f"Expected original experiment to remain stopped: {pid}")

    name = "reference_capacity_1024"
    plan = output / "capacity_1024_plan.json"
    baseline = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    total_mib, baseline_mib = [int(value.strip()) for value in baseline.split(",")]
    with plan.open("x") as stream:
        json.dump({"previous": previous, "num_envs": 1024, "variant": "reference",
                   "training": False, "memory_sampling_interval_s": 5,
                   "started_unix_s": time.time()}, stream, indent=2)
    status = {"phase": name, "complete": False, "sequence_pid": os.getpid(),
              "processes": {}, "phase_started_unix_s": time.time(),
              "performance_work_complete": False}
    memory_path = output / "capacity_1024_gpu_memory.csv"
    monitor = None

    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt(f"Capacity supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        with memory_path.open("x") as stream:
            monitor = subprocess.Popen([
                "nvidia-smi", "-i", "0", "--query-gpu=timestamp,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits", "-l", "5",
            ], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status["memory_monitor_pid"] = monitor.pid
            write_json(output / "status.json", status)
            command = [sys.executable, "-u", str(Path(__file__).with_name("profile_environment.py")),
                       "--project", str(project), "--output", str(output / name),
                       "--variant", "reference", "--num-envs", "1024", "--steps", "450",
                       "--warmup-steps", "150"]
            run_parallel([(name, command, root)], 1200, output, status)
            if not json.loads((output / name / "result.json").read_text())["passed"]:
                raise RuntimeError("1024-environment measurement did not pass")
        status.update(phase="capacity_1024_complete", complete=True, processes={})
    except BaseException as error:
        status.update(phase="capacity_1024_failed", complete=False, error=repr(error))
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=10)
        samples = []
        if memory_path.exists():
            for row in csv.reader(memory_path.read_text().splitlines()):
                if len(row) == 3:
                    try:
                        samples.append(int(row[1].strip()))
                    except ValueError:
                        pass
        write_json(output / "capacity_1024_memory_summary.json", {
            "gpu_total_mib": total_mib, "baseline_device_used_mib": baseline_mib,
            "maximum_sampled_device_used_mib": max(samples) if samples else None,
            "sample_count": len(samples), "sampling_interval_s": 5,
            "interpretation": "Device-wide sampled memory includes paused evaluation contexts. "
                              "Short peaks may be missed; this does not measure PPO training memory.",
        })
        status["memory_monitor_stopped"] = monitor is None or monitor.poll() is not None
        write_json(output / "status.json", status)


if __name__ == "__main__":
    main()
