"""Read-only CPU audit of the 500-rollout PPO continuation and event series."""

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


torch.set_num_threads(1)
root = Path(sys.argv[1])
backup = root / "training_backup"
plan = json.loads((root / "plan.json").read_text())
budget = json.loads((root / "training_budget.json").read_text())
initial = torch.load(root / "initial_checkpoint.pth", map_location="cpu", weights_only=False)
checkpoints = {}
for path in (backup / "nn").glob("*.pth"):
    epoch = int(re.search(r"_ep_(\d+)_", path.name).group(1))
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert int(value["epoch"]) == epoch
    assert int(value["frame"]) == 2899968 + (epoch - 121) * 65536
    assert set(value["model"]) == set(initial["model"])
    assert all(value["model"][key].shape == initial["model"][key].shape for key in initial["model"])
    assert all(torch.isfinite(tensor).all() for tensor in value["model"].values() if tensor.is_floating_point())
    checkpoints[epoch] = {"frame": int(value["frame"]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "filename": path.name}
assert sorted(checkpoints) == [200, 300, 400, 500, 600, 621]
assert (int(initial["epoch"]), int(initial["frame"])) == (133, 3686400)
assert budget["initial_epoch"] == 133 and budget["final_epoch"] == 621
assert budget["added_transitions"] == 31981568
assert budget["final_frame"] == checkpoints[621]["frame"] == 35667968
assert budget["checkpoint_sha256"] == checkpoints[621]["sha256"]
assert plan["original_logical_rollouts"] == 500 and plan["original_logical_transitions"] == 32768000
assert plan["discarded_completed_rollouts"] == 1
env = yaml.load((backup / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
agent = yaml.safe_load((backup / "params/agent.yaml").read_text())
assert int(env["scene"]["num_envs"]) == agent["params"]["config"]["num_actors"] == 1024
ac = agent["params"]["config"]
assert ac["horizon_length"] == 64 and ac["max_epochs"] == 621
assert ac["save_frequency"] == 100 and ac["save_best_after"] == 1000000000
assert ac["minibatch_size"] == 2048 and ac["mini_epochs"] == 4
assert ac["gamma"] == 0.99 and ac["tau"] == 0.95
assert agent["params"]["seed"] == 42
assert float(env["sim"]["dt"]) == 1 / 120 and int(env["decimation"]) == 8
assert env["action_contract"] == "relative_z_v1"
assert env["reward"]["dense_profile"] == "legacy" and env["reward"]["success_contract"] == "terminal_hold_v2"
assert float(env["reward"]["hold_duration_s"]) == 1
assert int(env["sim"]["physx"]["max_position_iteration_count"]) == 192
assert float(env["episode_length_s"]) == 151 / 15
assert hashlib.sha256((root / "initial_checkpoint.pth").read_bytes()).hexdigest() == plan["initial_checkpoint_sha256"]

pattern = re.compile(r"fps total: (\d+) epoch: (\d+)/621 frames: (\d+)")
initial_log = root / "initial_training.log"
if not initial_log.exists():
    initial_log = root.parent / "run/training.log"
old = [(int(epoch), int(frame), int(fps)) for fps, epoch, frame in pattern.findall(initial_log.read_text())]
new = [(int(epoch), int(frame), int(fps)) for fps, epoch, frame in pattern.findall((root / "training.log").read_text())]
old = [item for item in old if 122 <= item[0] <= 134]
assert [item[0] for item in old] == list(range(122, 135))
assert [item[0] for item in new] == list(range(134, 622))
assert len(old) + len(new) == 501
assert sorted(set(item[0] for item in old + new)) == list(range(122, 622))
assert all(frame == 2899968 + (epoch - 122) * 65536 for epoch, frame, _ in old)
assert all(frame == 3686400 + (epoch - 134) * 65536 for epoch, frame, _ in new)

events = EventAccumulator(str(backup / "summaries"), size_guidance={"scalars": 0})
events.Reload()
scalars = {}
for tag in events.Tags()["scalars"]:
    series = events.Scalars(tag)
    if not series:
        continue
    item = series[-1]
    scalars[tag] = {"count": len(series), "last_step": item.step, "last_value": item.value,
                    "last_10_mean": statistics.mean(event.value for event in series[-10:])}
assert not torch.cuda.is_initialized()
report = {
    "passed": True,
    "cpu_only": True,
    "initial_checkpoint_epoch": 133,
    "final_checkpoint_epoch": 621,
    "original_logical_rollouts": 500,
    "retained_new_transitions": 32768000,
    "resumed_retained_transitions": 31981568,
    "discarded_completed_rollout_transitions": 65536,
    "possible_additional_partial_discard_upper_bound": 65536,
    "log_epoch_134_repeated": True,
    "checkpoint_epochs": checkpoints,
    "post_resume_median_fps_total": statistics.median(fps for _, _, fps in new if fps >= 1000),
    "saved_config_verified": True,
    "all_model_weights_finite": True,
    "tensorboard_scalar_summaries": scalars,
}
print(json.dumps(report, indent=2, allow_nan=False))
