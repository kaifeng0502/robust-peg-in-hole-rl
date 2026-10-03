"""Single-factor, instance-local cache of two target-bound tensors.

Install after initialization and before instrumentation. Configuration values,
device and Torch's default floating dtype must stay fixed until restore. Call
validate_cache outside timing windows if needed; after changing these settings,
restore and reinstall. There are no extra per-step checks or synchronizations.
This experimental candidate has no established speed or simulator-parity claim.
"""

from __future__ import annotations

from functools import update_wrapper
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from types import CodeType, FunctionType, MethodType


EXPECTED_SOURCE_SHA256 = "71fe498d071c25a163990a2b6debac10f43cb1ff68fa3c8b9253080fba9e01df"
BOUNDS_EXPRESSION = "torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)"
ERROR_EXPRESSION = (
    "torch.tensor(\n"
    "            [hold.max_xy_error_m, hold.max_xy_error_m, hold.max_z_error_m], device=self.device\n"
    "        )"
)
# A single underscore avoids class-scope name mangling during whole-module
# compilation; the method loads these exact private names from its own globals.
BOUNDS_NAME = "_insertion_target_bounds_cache"
ERROR_NAME = "_insertion_target_error_cache"
SIGNATURE_NAME = "_insertion_target_cache_signature"


def _replace_expressions(source):
    """Only replace the two pinned constructor expressions, retaining line count."""
    for expression in (BOUNDS_EXPRESSION, ERROR_EXPRESSION):
        if source.count(expression) != 1:
            raise ValueError("Expected exactly one occurrence of each target-bound constructor")
    return source.replace(BOUNDS_EXPRESSION, BOUNDS_NAME, 1).replace(
        ERROR_EXPRESSION, "(\n            " + ERROR_NAME + "\n        )", 1
    )


def _method_code(source, filename):
    """Compile the whole module's symbol context without executing imports/classes."""
    module = compile(source, filename, "exec", dont_inherit=True)
    classes = [item for item in module.co_consts
               if isinstance(item, CodeType) and item.co_name == "LocalInsertionRLEnv"]
    if len(classes) != 1:
        raise ValueError("Expected exactly one LocalInsertionRLEnv class")
    methods = [item for item in classes[0].co_consts
               if isinstance(item, CodeType) and item.co_name == "_apply_action"]
    if len(methods) != 1:
        raise ValueError("Expected exactly one LocalInsertionRLEnv._apply_action method")
    return methods[0]


def _code_differences(runtime, trusted):
    fields = (
        "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
        "co_stacksize", "co_flags", "co_code", "co_consts", "co_names",
        "co_varnames", "co_freevars", "co_cellvars", "co_name", "co_qualname",
        "co_firstlineno", "co_linetable", "co_exceptiontable",
    )

    def describe(value):
        if isinstance(value, bytes):
            return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        value_repr = repr(value)
        return value_repr if len(value_repr) <= 240 else {
            "repr_length": len(value_repr), "sha256": hashlib.sha256(value_repr.encode()).hexdigest()
        }

    return {field: {"runtime": describe(getattr(runtime, field)),
                    "trusted": describe(getattr(trusted, field))}
            for field in fields if getattr(runtime, field, None) != getattr(trusted, field, None)}


def _configuration_signature(base, torch_module):
    bounds = base.cfg.ctrl.pos_action_bounds
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 3:
        raise ValueError("pos_action_bounds must contain three fixed numeric values")
    hold = base.cfg.hold
    values = (*bounds, hold.max_xy_error_m, hold.max_z_error_m)
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Cached bounds must contain positive finite numeric values")
    # Include scalar types: [1, 1, 1] and [1.0, 1.0, 1.0] infer different dtypes.
    return (tuple((type(value).__name__, value) for value in values),
            str(torch_module.device(base.device)), torch_module.get_default_dtype())


def build_candidate(base):
    """Validate the original method, then compile only the two substitutions."""
    bound = base._apply_action
    if not isinstance(bound, MethodType) or bound.__self__ is not base:
        raise TypeError("Expected the original bound _apply_action method")
    original = bound.__func__
    if not isinstance(original, FunctionType) or original.__closure__:
        raise TypeError("Expected the original unwrapped _apply_action function")
    if any(name in original.__globals__ for name in (BOUNDS_NAME, ERROR_NAME, SIGNATURE_NAME)):
        raise TypeError("Install only once, before _apply_action instrumentation")
    path = Path(original.__code__.co_filename)
    source_bytes = path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("Unsupported envs.py source SHA256")
    source = source_bytes.decode("utf-8")
    trusted_code = _method_code(source, str(path))
    runtime_code = original.__code__.replace(co_filename=str(path))
    if runtime_code != trusted_code:
        raise TypeError("Runtime _apply_action differs from the pinned source: "
                        + json.dumps(_code_differences(runtime_code, trusted_code), sort_keys=True))
    torch_module = original.__globals__.get("torch")
    if torch_module is None:
        raise TypeError("Original _apply_action has no Torch dependency")
    signature = _configuration_signature(base, torch_module)
    # Use the exact original constructors: no dtype derived from state tensors,
    # no expand/repeat, and no conversion of scalar input types.
    hold = base.cfg.hold
    bounds = torch_module.tensor(base.cfg.ctrl.pos_action_bounds, device=base.device)
    max_error = torch_module.tensor(
        [hold.max_xy_error_m, hold.max_xy_error_m, hold.max_z_error_m], device=base.device
    )
    if _configuration_signature(base, torch_module) != signature:
        raise RuntimeError("Configuration changed during cache construction")
    namespace = dict(original.__globals__)
    namespace["__builtins__"] = original.__builtins__
    namespace.update({BOUNDS_NAME: bounds, ERROR_NAME: max_error, SIGNATURE_NAME: signature})
    code = _method_code(_replace_expressions(source), str(path) + " [target_constants]")
    candidate = FunctionType(code, namespace, original.__name__, original.__defaults__)
    candidate.__kwdefaults__ = original.__kwdefaults__
    return update_wrapper(candidate, original)


def validate_cache(base):
    """Optional phase-boundary check; never called by the replacement hot path."""
    bound = base._apply_action
    if not isinstance(bound, MethodType) or bound.__self__ is not base:
        raise TypeError("No target-constant cache is installed on this instance")
    namespace = bound.__func__.__globals__
    if SIGNATURE_NAME not in namespace:
        raise TypeError("No target-constant cache is installed on this instance")
    if _configuration_signature(base, namespace["torch"]) != namespace[SIGNATURE_NAME]:
        raise RuntimeError("Cached configuration/device/default dtype changed; restore and reinstall")


def install(base):
    """Override one initialized instance's method; return an idempotent restore."""
    candidate = build_candidate(base)
    absent = object()
    previous = vars(base).get("_apply_action", absent)
    base._apply_action = MethodType(candidate, base)
    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        if previous is absent:
            delattr(base, "_apply_action")
        else:
            base._apply_action = previous
        restored = True

    return restore
