"""Single-factor candidate: reuse one mass-matrix inverse within one call.

This is not a measured speedup. Install before instrumentation, on an isolated
environment instance. It changes neither the shared module nor its source file.
"""

from __future__ import annotations

from functools import update_wrapper
import hashlib
import json
from pathlib import Path
from types import CodeType, FunctionType, MethodType


EXPECTED_SOURCE_SHA256 = "7fbbdea0b25f63864e5ee2684e5cd2f85c0baee36a9960192508919cd51c33f3"
ORIGINAL_EXPRESSION = "jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T"
REPLACEMENT_EXPRESSION = "jacobian @ arm_mass_matrix_inv @ jacobian_T"


def _replace_expression(source):
    if source.count(ORIGINAL_EXPRESSION) != 1:
        raise ValueError("Expected exactly one repeated mass-matrix inverse")
    return source.replace(ORIGINAL_EXPRESSION, REPLACEMENT_EXPRESSION, 1)


def _code_differences(runtime, trusted):
    """Compact evidence for loader/compiler differences; never bypass the guard."""
    fields = (
        "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
        "co_stacksize", "co_flags", "co_code", "co_consts", "co_names",
        "co_varnames", "co_freevars", "co_cellvars", "co_name", "co_qualname",
        "co_firstlineno", "co_linetable", "co_exceptiontable",
    )

    def describe(value):
        if isinstance(value, bytes):
            return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        if isinstance(value, (int, str)):
            return value
        representation = repr(value)
        if len(representation) <= 240:
            return representation
        return {"repr_length": len(representation), "prefix": representation[:120],
                "sha256": hashlib.sha256(representation.encode()).hexdigest()}

    return {field: {"runtime": describe(getattr(runtime, field)),
                    "trusted": describe(getattr(trusted, field))}
            for field in fields if getattr(runtime, field, None) != getattr(trusted, field, None)}


def _controller_code(source, filename):
    # Whole-module compilation preserves the imported-module symbol table.
    # It does not execute imports or any other top-level module statements.
    module_code = compile(source, filename, "exec", dont_inherit=True)
    matches = [constant for constant in module_code.co_consts
               if isinstance(constant, CodeType) and constant.co_name == "compute_dof_torque"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one compute_dof_torque function")
    return matches[0]


def build_candidate(factory_control):
    """Compile the pinned module without executing it; change one expression.

    Dependencies are the original function's globals. Dynamic mass matrices
    are still inverted on every call; no tensors or inverses are cached.
    """
    path = Path(factory_control.__file__)
    source_bytes = path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("Unsupported factory_control.py source SHA256")
    original = factory_control.compute_dof_torque
    if not isinstance(original, FunctionType) or original.__closure__:
        raise TypeError("Expected the original unwrapped compute_dof_torque function")
    source = source_bytes.decode("utf-8")
    # Reject an already wrapped or modified runtime function, even if its disk
    # source still has the expected hash. Ignore only the loader's file path.
    namespace = dict(original.__globals__)
    trusted_code = _controller_code(source, str(path))
    runtime_code = original.__code__.replace(co_filename=str(path))
    if runtime_code != trusted_code:
        differences = _code_differences(runtime_code, trusted_code)
        raise TypeError("Runtime compute_dof_torque differs from the pinned source: "
                        + json.dumps(differences, sort_keys=True))
    candidate_source = _replace_expression(source)
    candidate_code = _controller_code(candidate_source, str(path) + " [inverse_reuse]")
    candidate = FunctionType(candidate_code, namespace, original.__name__, original.__defaults__)
    candidate.__kwdefaults__ = original.__kwdefaults__
    return update_wrapper(candidate, original)


class _FactoryControlProxy:
    """Forward every module attribute except the one experimental function."""

    def __init__(self, module, candidate):
        self._module = module
        self.compute_dof_torque = candidate

    def __getattr__(self, name):
        return getattr(self._module, name)


def install(base):
    """Override this instance's control method; return an idempotent restore."""
    original = base.generate_ctrl_signals
    if not isinstance(original, MethodType) or original.__self__ is not base:
        raise TypeError("Expected the original bound generate_ctrl_signals method")
    function = original.__func__
    module = function.__globals__.get("factory_control")
    if module is None or isinstance(module, _FactoryControlProxy):
        raise TypeError("Install once, before control-method instrumentation")
    candidate = build_candidate(module)
    namespace = dict(function.__globals__)
    namespace["factory_control"] = _FactoryControlProxy(module, candidate)
    replacement = FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    replacement.__kwdefaults__ = function.__kwdefaults__
    update_wrapper(replacement, function)
    absent = object()
    previous = vars(base).get("generate_ctrl_signals", absent)
    base.generate_ctrl_signals = MethodType(replacement, base)
    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        if previous is absent:
            delattr(base, "generate_ctrl_signals")
        else:
            base.generate_ctrl_signals = previous
        restored = True

    return restore
