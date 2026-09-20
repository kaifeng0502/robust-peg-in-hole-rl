"""Pure-Python schedules shared by the scripted insertion baseline and its tests."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpiralSchedule:
    """Archimedean spiral used after the peg first touches the hole plate."""

    radial_speed_m_s: float = 0.0010
    turns_per_second: float = 1.25
    max_radius_m: float = 0.0045

    def validate(self) -> None:
        if self.radial_speed_m_s <= 0.0:
            raise ValueError("radial_speed_m_s must be positive")
        if self.turns_per_second <= 0.0:
            raise ValueError("turns_per_second must be positive")
        if self.max_radius_m <= 0.0:
            raise ValueError("max_radius_m must be positive")

    def offset(self, elapsed_s: float) -> tuple[float, float]:
        """Return the deterministic XY search offset at ``elapsed_s``."""
        self.validate()
        t = max(0.0, elapsed_s)
        radius = min(self.radial_speed_m_s * t, self.max_radius_m)
        angle = 2.0 * math.pi * self.turns_per_second * t
        return radius * math.cos(angle), radius * math.sin(angle)


def bounded_target(current: float, desired: float, max_error: float) -> float:
    """Limit a Cartesian target so impedance error cannot grow without bound."""
    if max_error <= 0.0:
        raise ValueError("max_error must be positive")
    return min(max(desired, current - max_error), current + max_error)
