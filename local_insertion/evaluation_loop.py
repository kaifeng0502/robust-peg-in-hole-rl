"""Finite-horizon insertion evaluation with an injectable simulator adapter.

This module uses only the Python standard library. A backend owns observations,
policy inference, and simulation; this loop rejects reset-contaminated samples
before computing metrics and never resets or steps beyond the requested case.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

if __package__:
    from .evaluation_metrics import EpisodeMetrics, EvaluationCriteria
else:
    from evaluation_metrics import EpisodeMetrics, EvaluationCriteria


@dataclass(frozen=True)
class ControlSample:
    """One completed control interval, including its physics-boundary checks."""

    observation: Any
    xy_error_m: float
    depth_m: float
    reward: float
    task_reward: float
    commanded_force_n: float
    interval_success: bool
    interval_ever_success: bool
    terminated: bool
    truncated: bool
    episode_step: int
    action: list[float]


def _finite_task_reward(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("task reward must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("task reward must be finite")
    return value


def run_case(
    case: Mapping[str, Any],
    criteria: EvaluationCriteria,
    target_depth_m: float,
    horizon_steps: int,
    reset: Callable[[int], tuple[Any, Any]],
    act: Callable[[Any], Any],
    step: Callable[[Any], ControlSample],
    write_trace: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run exactly one seeded case and return its terminal metrics.

    ``reset(seed)`` returns the initial observation and a serializable snapshot
    of realized initial state. ``step(action)`` must report the episode counter
    after that step, starting at one. Any timeout, termination, or unexpected
    counter makes the case invalid; it is never scored as success or failure.
    """
    if type(horizon_steps) is not int or horizon_steps < criteria.required_hold_samples:
        raise ValueError("horizon_steps must be an integer covering the required hold duration")
    case_id = case["case_id"]
    reset_seed = case["seed"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a nonempty string")
    if type(reset_seed) is not int or not 0 <= reset_seed < 2**31:
        raise ValueError("reset seed must be an integer in [0, 2**31)")

    metrics = EpisodeMetrics(criteria, target_depth_m)
    observation, initial_state = reset(reset_seed)
    initial_state = deepcopy(initial_state)
    task_episode_reward = 0.0

    for expected_step in range(1, horizon_steps + 1):
        sample = step(act(observation))
        # DirectRLEnv can replace terminal geometry with the next reset state.
        # Check the lifecycle before inspecting or scoring that geometry.
        if sample.terminated or sample.truncated:
            raise RuntimeError(
                f"Case {case_id} reset or terminated at step {expected_step}; "
                "the trajectory cannot be scored"
            )
        if type(sample.episode_step) is not int or sample.episode_step != expected_step:
            raise RuntimeError(
                f"Case {case_id} expected episode step {expected_step}, "
                f"received {sample.episode_step!r}; possible reset contamination"
            )
        task_reward = _finite_task_reward(sample.task_reward)
        next_task_episode_reward = _finite_task_reward(task_episode_reward + task_reward)
        metrics.update(
            sample.xy_error_m,
            sample.depth_m,
            sample.reward,
            sample.commanded_force_n,
            interval_success=sample.interval_success,
            interval_ever_success=sample.interval_ever_success,
        )
        task_episode_reward = next_task_episode_reward
        observation = sample.observation
        if write_trace is not None:
            current = metrics.result()
            write_trace(
                {
                    "case_id": case_id,
                    "step": expected_step,
                    "time_s": expected_step * criteria.dt_s,
                    "xy_error_m": sample.xy_error_m,
                    "depth_m": sample.depth_m,
                    "target_depth_m": metrics.target_depth_m,
                    "reward": sample.reward,
                    "task_reward": task_reward,
                    "commanded_force_n": sample.commanded_force_n,
                    "interval_success": sample.interval_success,
                    "interval_ever_success": sample.interval_ever_success,
                    "final_success": current["final_success"],
                    "held_success": current["held_success"],
                    "action": list(sample.action),
                }
            )

    return {
        **metrics.result(),
        "case_id": case_id,
        "reset_seed": reset_seed,
        "initial_state": initial_state,
        "task_episode_reward": task_episode_reward,
    }
