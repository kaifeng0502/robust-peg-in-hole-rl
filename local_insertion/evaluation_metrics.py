"""Dependency-free, sampled terminal metrics for local insertion evaluation.

Each update is one post-control sample representing the preceding ``dt_s``
interval. Callers can supply ``interval_success`` as the conjunction of geometry
checks at the interval start and every physics substep, and
``interval_ever_success`` as their disjunction. A one-second hold at 15 Hz then
requires 15 consecutive fully valid intervals. Without these optional flags,
each final control sample is assumed to represent its preceding interval. This
is a discrete simulation definition, not continuous contact sensing.
Only the successful run still active at the evaluation horizon determines
    ``held_success`` and ``success_time_s``; an earlier hold is reported separately.
Forces in this module are commanded force magnitudes, not contact measurements.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import NormalDist, mean, median
from typing import Any


def _finite_number(value: float, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _within_tolerance(error: float, tolerance: float) -> bool:
    # Permit roundoff at the inclusive boundary, without adding a spatial margin.
    return error <= tolerance or math.isclose(error, tolerance, rel_tol=1e-12, abs_tol=0.0)


@dataclass(frozen=True)
class EvaluationCriteria:
    """Common terminal geometry and sampled hold duration for all controllers."""

    dt_s: float
    hold_duration_s: float = 1.0
    xy_tolerance_m: float = 0.0025
    depth_tolerance_m: float = 0.001

    def __post_init__(self) -> None:
        for name in ("dt_s", "hold_duration_s", "xy_tolerance_m", "depth_tolerance_m"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name, positive=True))
        if not math.isfinite(self.hold_duration_s / self.dt_s):
            raise ValueError("hold_duration_s / dt_s must be finite")

    @property
    def required_hold_samples(self) -> int:
        ratio = self.hold_duration_s / self.dt_s
        nearest = round(ratio)
        if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12):
            ratio = float(nearest)
        return max(1, math.ceil(ratio))


class EpisodeMetrics:
    """Accumulate one episode; call result before the simulator resets it.

    Success geometry requires XY error at or below its tolerance AND absolute
    insertion-depth error at or below its tolerance. Excessive insertion depth
    is not automatically successful. ``success_time_s`` is the time at which
    the final trailing successful run first satisfied the hold duration. It is
    None unless that run still meets the hold requirement at the horizon.
    ``first_success_time_s`` is the end of the first interval containing success,
    so its timing resolution is dt_s even when a substep OR flag is provided.
    """

    def __init__(self, criteria: EvaluationCriteria, target_depth_m: float):
        self.criteria = criteria
        self.target_depth_m = _finite_number(target_depth_m, "target_depth_m", positive=True)
        self.sample_count = 0
        self._trailing_samples = 0
        self._max_hold_samples = 0
        self._ever_success = False
        self._final_success = False
        self._first_success_time_s: float | None = None
        self._first_hold_time_s: float | None = None
        self._final_xy_error_m: float | None = None
        self._final_depth_m: float | None = None
        self._max_depth_m = -math.inf
        self._peak_commanded_force_n = 0.0
        self._episode_reward = 0.0

    def update(
        self,
        xy_error_m: float,
        depth_m: float,
        reward: float,
        commanded_force_n: float,
        *,
        interval_success: bool | None = None,
        interval_ever_success: bool | None = None,
    ) -> None:
        """Add one control interval atomically; reject invalid samples in full.

        Pass the maximum commanded force over the physics substeps when that
        trace is available; otherwise this records the supplied sample value.
        Interval flags can strengthen hold detection and capture brief success
        between control samples; final_success always uses the final geometry.
        """
        for name, flag in (("interval_success", interval_success), ("interval_ever_success", interval_ever_success)):
            if flag is not None and not isinstance(flag, bool):
                raise ValueError(f"{name} must be bool or None")
        xy_error_m = _finite_number(xy_error_m, "xy_error_m", nonnegative=True)
        depth_m = _finite_number(depth_m, "depth_m")
        reward = _finite_number(reward, "reward")
        commanded_force_n = _finite_number(commanded_force_n, "commanded_force_n", nonnegative=True)
        accumulated_reward = _finite_number(self._episode_reward + reward, "accumulated episode reward")
        elapsed_time = _finite_number((self.sample_count + 1) * self.criteria.dt_s, "elapsed time")

        success = _within_tolerance(xy_error_m, self.criteria.xy_tolerance_m) and _within_tolerance(
            abs(depth_m - self.target_depth_m), self.criteria.depth_tolerance_m
        )
        valid_interval = success and interval_success is not False
        ever_in_interval = success or interval_ever_success is True
        self.sample_count += 1
        self._trailing_samples = self._trailing_samples + 1 if valid_interval else 0
        self._max_hold_samples = max(self._max_hold_samples, self._trailing_samples)
        self._ever_success |= ever_in_interval
        self._final_success = success
        if ever_in_interval and self._first_success_time_s is None:
            self._first_success_time_s = elapsed_time
        if self._trailing_samples >= self.criteria.required_hold_samples and self._first_hold_time_s is None:
            self._first_hold_time_s = elapsed_time
        self._final_xy_error_m = xy_error_m
        self._final_depth_m = depth_m
        self._max_depth_m = max(self._max_depth_m, depth_m)
        self._peak_commanded_force_n = max(self._peak_commanded_force_n, commanded_force_n)
        self._episode_reward = accumulated_reward

    def result(self) -> dict[str, Any]:
        """Return metrics at the current horizon; an empty episode is invalid."""
        if self.sample_count == 0:
            raise ValueError("Cannot evaluate an episode without control samples")
        held_success = self._trailing_samples >= self.criteria.required_hold_samples
        completion_sample = self.sample_count - self._trailing_samples + self.criteria.required_hold_samples
        success_time_s = completion_sample * self.criteria.dt_s if held_success else None
        if held_success:
            failure_category = None
        elif self._final_success:
            failure_category = "insufficient_hold"
        elif self._ever_success:
            failure_category = "lost_insertion"
        elif not _within_tolerance(self._final_xy_error_m, self.criteria.xy_tolerance_m):
            failure_category = "misalignment"
        elif self._final_depth_m < self.target_depth_m:
            failure_category = "insufficient_depth"
        else:
            failure_category = "excessive_depth"
        return {
            "success": held_success,
            "ever_success": self._ever_success,
            "final_success": self._final_success,
            "held_success": held_success,
            "had_hold_anytime": self._first_hold_time_s is not None,
            "first_success_time_s": self._first_success_time_s,
            "first_hold_time_s": self._first_hold_time_s,
            "success_time_s": success_time_s,
            "trailing_hold_duration_s": self._trailing_samples * self.criteria.dt_s,
            "max_hold_duration_s": self._max_hold_samples * self.criteria.dt_s,
            "final_xy_error_m": self._final_xy_error_m,
            "final_depth_m": self._final_depth_m,
            "target_depth_m": self.target_depth_m,
            "max_depth_m": self._max_depth_m,
            "peak_commanded_force_n": self._peak_commanded_force_n,
            "episode_reward": self._episode_reward,
            "failure_category": failure_category,
            "sample_count": self.sample_count,
            "elapsed_time_s": self.sample_count * self.criteria.dt_s,
        }


def wilson95(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided Wilson 95% interval for a binomial success proportion."""
    if isinstance(successes, bool) or isinstance(trials, bool) or not isinstance(successes, int) or not isinstance(trials, int):
        raise ValueError("successes and trials must be integers")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Require trials > 0 and 0 <= successes <= trials")
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    # Preserve exact binomial edge bounds in the presence of floating roundoff.
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return lower, upper


def summarize_trials(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate EpisodeMetrics results; success_rate always means held success.

    Extra per-trial fields (seed, trial ID, controller) are accepted. Intervals
    describe the trial success proportion; they are not variation across
    independently trained policy seeds. Failure categories describe terminal
    geometry and hold behavior, not inferred physical causes such as jamming.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("At least one evaluated trial is required")
    flags = ("ever_success", "final_success", "held_success", "had_hold_anytime")
    counts = Counter({key: 0 for key in flags})
    times = []
    forces = []
    depths = []
    rewards = []
    failure_counts: Counter[str] = Counter()
    for row in rows:
        for key in flags:
            if row[key] not in (False, True):
                raise ValueError(f"{key} must be a boolean or 0/1")
            counts[key] += int(row[key])
        if row["held_success"] and not (row["final_success"] and row["had_hold_anytime"]):
            raise ValueError("held_success requires final_success and had_hold_anytime")
        if (row["final_success"] or row["had_hold_anytime"]) and not row["ever_success"]:
            raise ValueError("Successful final geometry or hold requires ever_success")
        if row["held_success"]:
            times.append(_finite_number(row["success_time_s"], "success_time_s", nonnegative=True))
            if row["failure_category"] is not None:
                raise ValueError("A successful trial must not have a failure category")
        else:
            if row["success_time_s"] is not None:
                raise ValueError("An unsuccessful trial must not have a completion time")
            category = row["failure_category"]
            if not isinstance(category, str) or not category:
                raise ValueError("An unsuccessful trial must have a failure category")
            failure_counts[category] += 1
        forces.append(_finite_number(row["peak_commanded_force_n"], "peak_commanded_force_n", nonnegative=True))
        depths.append(_finite_number(row["max_depth_m"], "max_depth_m"))
        rewards.append(_finite_number(row["episode_reward"], "episode_reward"))
    trials = len(rows)
    return {
        "trials": trials,
        "success_rate": counts["held_success"] / trials,
        "success_rate_ci95": list(wilson95(counts["held_success"], trials)),
        **{f"{key}_count": counts[key] for key in flags},
        "ever_success_rate": counts["ever_success"] / trials,
        "final_success_rate": counts["final_success"] / trials,
        "had_hold_anytime_rate": counts["had_hold_anytime"] / trials,
        "mean_success_time_s": mean(times) if times else None,
        "median_success_time_s": median(times) if times else None,
        "mean_peak_commanded_force_n": mean(forces),
        "max_peak_commanded_force_n": max(forces),
        "mean_max_depth_m": mean(depths),
        "mean_episode_reward": mean(rewards),
        "failure_counts": dict(sorted(failure_counts.items())),
        "force_note": "Commanded force magnitude, not sensor-measured contact force.",
    }
