"""Small, torch-independent CPU wall-time probe for synchronous Python methods.

The probe never synchronizes CUDA. These measurements describe host-observed
method latency, including any waiting already performed by the original method;
they do not measure asynchronous GPU kernel execution time. Nested regions have
both inclusive time and exclusive time (inclusive less timed child calls).

Wrap instances or modules, rather than their classes. Async/generator callables
are deliberately unsupported because timing their creation would be misleading.
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _Frame:
    label: str
    started: float
    children_s: float = 0.0


@dataclass
class _Patch:
    obj: Any
    method_name: str
    had_own_attribute: bool
    original_attribute: Any


class Probe:
    """Temporarily time bound methods, without altering arguments or results.

    ``region_factory`` may be ``torch.profiler.record_function``. Its context
    manager surrounds each measured call, while its entry/exit overhead is
    outside that call's measured duration. Nested overhead can still appear in
    the parent's exclusive time. Thread-local stacks prevent unrelated threads
    from being treated as parent and child. Totals across threads may overlap.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.perf_counter,
        region_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._clock = clock
        self._region_factory = region_factory
        self._local = threading.local()
        self._lock = threading.RLock()
        self._patches: list[_Patch] = []
        self._regions: dict[str, dict[str, int | float]] = {}
        self._active_calls = 0
        self._root_calls = 0
        self._root_wall_s = 0.0

    def wrap(self, obj: Any, method_name: str, label: str) -> Callable[..., Any]:
        """Replace one callable on an instance/module and return its wrapper.

        A duplicate wrap of the same object attribute is rejected. An inherited
        method is restored by deleting the temporary instance attribute. An
        existing instance attribute is restored to the exact original object.
        """
        if isinstance(obj, type):
            raise TypeError("Wrap an instance or module, not a class")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a nonempty string")
        with self._lock:
            if any(p.obj is obj and p.method_name == method_name for p in self._patches):
                raise ValueError(f"Already wrapped attribute: {method_name}")
            original = getattr(obj, method_name)
            if not callable(original):
                raise TypeError(f"Attribute is not callable: {method_name}")
            if (
                inspect.iscoroutinefunction(original)
                or inspect.isgeneratorfunction(original)
                or inspect.isasyncgenfunction(original)
            ):
                raise TypeError("Only synchronous, non-generator callables are supported")
            # Requiring a namespace lets restoration preserve descriptor binding
            # exactly, instead of guessing how an object's custom slots behave.
            try:
                namespace = vars(obj)
            except TypeError as exc:
                raise TypeError("Wrapped object must expose an attribute namespace") from exc
            had_own_attribute = method_name in namespace
            original_attribute = namespace.get(method_name)

            @functools.wraps(original)
            def measured(*args: Any, **kwargs: Any) -> Any:
                context = (
                    self._region_factory(label)
                    if self._region_factory is not None
                    else nullcontext()
                )
                with context:
                    stack = getattr(self._local, "stack", None)
                    if stack is None:
                        stack = []
                        self._local.stack = stack
                    frame = _Frame(label=label, started=self._clock())
                    stack.append(frame)
                    with self._lock:
                        self._active_calls += 1
                    failed = False
                    try:
                        return original(*args, **kwargs)
                    except BaseException:
                        failed = True
                        raise
                    finally:
                        # Pop even if a caller-supplied clock fails. Clock errors
                        # propagate, but cannot leave a stale nesting frame.
                        try:
                            elapsed = max(0.0, self._clock() - frame.started)
                        except BaseException:
                            stack.pop()
                            with self._lock:
                                self._active_calls -= 1
                            raise
                        stack.pop()
                        exclusive = max(0.0, elapsed - frame.children_s)
                        if stack:
                            stack[-1].children_s += elapsed
                        with self._lock:
                            region = self._regions.setdefault(
                                label,
                                {
                                    "count": 0,
                                    "errors": 0,
                                    "inclusive_s": 0.0,
                                    "exclusive_s": 0.0,
                                },
                            )
                            region["count"] += 1
                            region["errors"] += int(failed)
                            region["inclusive_s"] += elapsed
                            region["exclusive_s"] += exclusive
                            if not stack:
                                self._root_calls += 1
                                self._root_wall_s += elapsed
                            self._active_calls -= 1

            setattr(obj, method_name, measured)
            self._patches.append(
                _Patch(obj, method_name, had_own_attribute, original_attribute)
            )
            return measured

    def restore(self) -> None:
        """Restore all original attributes; attempt every restore on failure.

        Restoration is idempotent. A failed restoration remains registered so a
        later call can retry it. Already-running original methods can complete.
        """
        failures: list[tuple[_Patch, Exception]] = []
        with self._lock:
            for patch in reversed(self._patches):
                try:
                    if patch.had_own_attribute:
                        setattr(patch.obj, patch.method_name, patch.original_attribute)
                    else:
                        delattr(patch.obj, patch.method_name)
                except Exception as exc:
                    failures.append((patch, exc))
            self._patches = [patch for patch, _ in reversed(failures)]
        if failures:
            names = ", ".join(patch.method_name for patch, _ in failures)
            raise RuntimeError(f"Could not restore attributes: {names}") from failures[0][1]

    def clear(self) -> None:
        """Clear measurements without unwrapping methods (between calls only)."""
        with self._lock:
            if self._active_calls:
                raise RuntimeError("Cannot clear while measured calls are active")
            self._regions.clear()
            self._root_calls = 0
            self._root_wall_s = 0.0

    def summary(self) -> dict[str, Any]:
        """Return an independent, JSON-serializable snapshot of completed calls."""
        with self._lock:
            return {
                "measurement": "cpu_wall_time",
                "unit": "seconds",
                "probe_inserts_cuda_synchronization": False,
                "note": (
                    "Host-observed method latency, not GPU kernel time. Inclusive "
                    "times overlap for nested calls. Root wall time sums root "
                    "calls and may overlap across threads."
                ),
                "completed_calls": sum(int(r["count"]) for r in self._regions.values()),
                "active_calls": self._active_calls,
                "root_calls": self._root_calls,
                "root_wall_s": self._root_wall_s,
                "regions": {name: dict(values) for name, values in self._regions.items()},
            }
