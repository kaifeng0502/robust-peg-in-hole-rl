"""Terminal-hold training state without any Isaac Sim dependency."""

from typing import Any, NamedTuple


class HoldTransition(NamedTuple):
    consecutive_samples: Any
    ever_geometry_success: Any
    ever_held_success: Any
    first_held_success: Any
    held_success: Any


def advance_terminal_hold(
    consecutive_samples,
    ever_geometry_success,
    ever_held_success,
    interval_success,
    interval_ever_success,
    final_success,
    required_samples: int,
) -> HoldTransition:
    """Advance scalar or batched-tensor state by one complete control interval.

    ``interval_success`` is the conjunction over every physics boundary, not
    just the final geometry. Reaching a hold twice earns the first-hold bonus
    only once. A previously completed hold does not imply terminal success.
    The operations also accept Python integers/bools for offline boundary tests.
    """
    if type(required_samples) is not int or required_samples < 1:
        raise ValueError("required_samples must be a positive integer")
    qualifying_interval = interval_success & final_success
    consecutive_samples = (consecutive_samples + 1) * qualifying_interval
    held_success = consecutive_samples >= required_samples
    return HoldTransition(
        consecutive_samples,
        ever_geometry_success | interval_ever_success | final_success,
        ever_held_success | held_success,
        held_success & (ever_held_success ^ True),
        held_success,
    )
