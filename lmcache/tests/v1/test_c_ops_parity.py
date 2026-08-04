# SPDX-License-Identifier: Apache-2.0
"""
Verify that every public function/enum in torch_ops that also
exists in c_ops has a matching signature.

Does NOT require torch_ops to implement everything in c_ops —
only checks the intersection. If you implement a function in the fallback,
its signature must match c_ops exactly.

Default value rules (relaxed):
  - If c_ops has a default, fallback MUST also have one with the same value —
    otherwise callers relying on the default will break.
  - If c_ops has NO default, fallback MAY add one (superset, backward-compatible).

Requires CUDA (c_ops must be importable). Automatically skipped on CPU-only CI.
"""

# Standard
import enum
import inspect
import re

# Third Party
import pytest

# First Party
from lmcache.v1.platform import torch_ops as fallback

try:
    # First Party
    import lmcache.c_ops as c_ops

    HAS_C_OPS = c_ops is not fallback
except ImportError:
    HAS_C_OPS = False


# ── Helpers ──


def _public_callables(module):
    """Return {name: obj} for all public, non-dunder, non-enum callables."""
    return {
        name: obj
        for name, obj in inspect.getmembers(module)
        if not name.startswith("_")
        and callable(obj)
        and not inspect.isclass(obj)  # classes tested by descriptor/enum tests
        and not hasattr(obj, "__members__")  # exclude pybind11 enums
        and getattr(obj, "__module__", None) == getattr(module, "__name__", None)
    }


def _public_enums(module):
    """Return {name: obj} for all public enum-like classes.

    Detects both Python enum.Enum subclasses and pybind11 enums.
    pybind11 enums have a ``__members__`` dict attribute.
    """
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if not name.startswith("_")
        and (issubclass(obj, enum.Enum) or hasattr(obj, "__members__"))
    }


def _get_enum_members(enum_cls):
    """Extract {name: value} from a Python enum or pybind11 enum."""
    if issubclass(enum_cls, enum.Enum):
        return {m.name: m.value for m in enum_cls}
    elif hasattr(enum_cls, "__members__"):
        return {name: int(val) for name, val in enum_cls.__members__.items()}
    return {}


def _public_descriptor_classes(module: object) -> dict[str, type]:
    """Return {name: cls} for public classes that are neither enums
    nor plain functions — e.g. pybind11 ``py::class_<>`` types like
    ``PageBufferShapeDesc``.
    """
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if not name.startswith("_")
        and not (issubclass(obj, enum.Enum) or hasattr(obj, "__members__"))
        and callable(obj)
    }


def _normalize_default(value_str):
    """Normalize a default value string for comparison.

    Handles differences between pybind11 docstring representation
    and Python inspect representation:
      - 'false'/'False' → False
      - 'true'/'True'   → True
      - numeric strings  → float/int
      - quoted strings   → unquoted
    """
    s = value_str.strip()

    if s.lower() == "false":
        return False
    if s.lower() == "true":
        return True
    if s == "None" or s == "none":
        return None

    # Strip quotes: 'foo' or "foo" → foo
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        return s[1:-1]

    # Try numeric
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass

    # Return as-is for anything else (e.g. enum values)
    return s


def _parse_docstring_params(func):
    """Parse pybind11 docstring to extract [(name, has_default, default_value)].

    pybind11 generates docstrings like:
        func_name(arg1: type1, arg2: type2 = default) -> ret

    Returns None if parsing fails.
    default_value is the normalized value if present, or None if no default.
    """
    doc = getattr(func, "__doc__", None)
    if not doc:
        return None

    first_line = doc.strip().split("\n")[0]
    match = re.match(r"\w+\((.*)\)\s*->", first_line)
    if not match:
        return None

    params_str = match.group(1)
    if not params_str.strip():
        return []

    # Split on commas that are NOT inside brackets/parens
    params = []
    depth = 0
    current = []
    for char in params_str:
        if char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        params.append("".join(current).strip())

    result = []
    for param in params:
        colon_idx = param.find(":")
        if colon_idx == -1:
            name = param.split("=")[0].strip()
        else:
            name = param[:colon_idx].strip()

        has_default = "=" in param
        default_value = None
        if has_default:
            default_value = _normalize_default(param.split("=", 1)[1])

        result.append((name, has_default, default_value))

    return result


def _get_python_params(func):
    """Extract [(name, has_default, default_value)] via inspect.signature."""
    try:
        sig = inspect.signature(func)
    except ValueError as err:
        raise ValueError("no signature found for {!r}".format(func)) from err
    result = []
    for p in sig.parameters.values():
        has_default = p.default is not inspect.Parameter.empty
        default_value = None
        if has_default:
            default_value = _normalize_default(repr(p.default))
        result.append((p.name, has_default, default_value))
    return result


def _has_real_names(params):
    """Check if parameter names are real (not arg0, arg1, ...)."""
    return params and not any(re.match(r"^arg\d+$", name) for name, _, _ in params)


# ── Discover the intersection automatically ──

# Functions intentionally excluded from parity checks.
# These are native-only with no torch fallback; call sites guard with hasattr.
# NOTE: execute_object_group_transfer previously had a signature-only stub
# (raised NotImplementedError unconditionally) — not a real fallback.
_EXCLUDED_FUNCS: set[str] = {
    "execute_object_group_transfer",
    "execute_cb_retrieve_plan_flat",
}

# Plan types (StagingCopy, LaunchVar, BatchStep, KernelGroupSpec)
# are native-only with no torch fallback — auto-discovered by bind_native.
# NOTE: these previously had signature-only stubs (raised NotImplementedError
# unconditionally) — not real fallbacks.
_EXCLUDED_DESCS: set[str] = {
    "StagingCopy",
    "LaunchVar",
    "BatchStep",
    "KernelGroupSpec",
    "CBGroupSpec",
}

_fallback_callables = _public_callables(fallback)
_c_ops_callables = _public_callables(c_ops) if HAS_C_OPS else {}
_shared_func_names = sorted(
    (set(_fallback_callables) & set(_c_ops_callables)) - _EXCLUDED_FUNCS
)

_fallback_enums = _public_enums(fallback)
_c_ops_enums = _public_enums(c_ops) if HAS_C_OPS else {}
_shared_enum_names = sorted(set(_fallback_enums) & set(_c_ops_enums))

_fallback_descs = _public_descriptor_classes(fallback)
_c_ops_descs = _public_descriptor_classes(c_ops) if HAS_C_OPS else {}
_shared_desc_names = sorted(set(_fallback_descs) & set(_c_ops_descs))


# ── Tests ──


@pytest.mark.skipif(not HAS_C_OPS, reason="c_ops not available (no CUDA)")
@pytest.mark.parametrize(
    "func_name",
    _shared_func_names if _shared_func_names else ["__placeholder__"],
)
def test_function_signature_parity(func_name):
    """For every function that torch_ops chose to implement,
    its signature must match c_ops exactly.

    When c_ops has real py::arg() names  → check names, count, defaults.
    When c_ops only has arg0/arg1/...    → check count and defaults only.

    Default rules (relaxed):
      c_ops has default, fallback does NOT      → FAIL (callers will break)
      c_ops has default, fallback has different  → FAIL (silent behavior change)
      c_ops has NO default, fallback does        → OK   (backward-compatible)
    """
    if func_name == "__placeholder__":
        pytest.skip("No shared functions found between c_ops and fallback")

    c_func = _c_ops_callables[func_name]
    py_func = _fallback_callables[func_name]

    # Get c_ops params: try docstring first (more reliable for pybind11),
    # then inspect.signature
    c_params = _parse_docstring_params(c_func)
    if c_params is None:
        try:
            c_params = _get_python_params(c_func)
        except (ValueError, TypeError):
            pass

    if c_params is None:
        pytest.skip(f"Cannot inspect c_ops.{func_name} signature")

    # Get fallback params: always via inspect.signature
    try:
        py_params = _get_python_params(py_func)
    except (ValueError, TypeError):
        pytest.skip(f"Cannot inspect fallback.{func_name} signature")

    c_has_names = _has_real_names(c_params)

    # 1. Always check parameter count
    assert len(c_params) == len(py_params), (
        f"{func_name}: parameter count mismatch\n"
        f"  c_ops ({len(c_params)}):    {[p[0] for p in c_params]}\n"
        f"  fallback ({len(py_params)}): {[p[0] for p in py_params]}"
    )

    # 2. Check each parameter
    for i, (
        (c_name, c_has_def, c_default),
        (py_name, py_has_def, py_default),
    ) in enumerate(zip(c_params, py_params, strict=False)):
        # Check names only when c_ops has real py::arg() names
        if c_has_names:
            assert c_name == py_name, (
                f"{func_name}: param #{i} name mismatch — "
                f"c_ops: '{c_name}', fallback: '{py_name}'"
            )

        # Relaxed default check:
        # Only fail if c_ops has a default but fallback does NOT,
        # or if both have defaults but values differ.
        # Fallback having an extra default is fine (backward-compatible).
        if c_has_def:
            assert py_has_def, (
                f"{func_name}: param #{i} ('{py_name}') — "
                f"c_ops has default={c_default!r} but fallback has NO default. "
                f"Callers relying on the default will break."
            )
            assert c_default == py_default, (
                f"{func_name}: param #{i} ('{py_name}') — "
                f"default value mismatch: "
                f"c_ops={c_default!r}, fallback={py_default!r}"
            )


@pytest.mark.skipif(not HAS_C_OPS, reason="c_ops not available (no CUDA)")
@pytest.mark.parametrize(
    "enum_name",
    _shared_enum_names if _shared_enum_names else ["__placeholder__"],
)
def test_enum_parity(enum_name):
    """For every enum that torch_ops defines,
    its members and values must match c_ops."""
    if enum_name == "__placeholder__":
        pytest.skip("No shared enums found between c_ops and fallback")

    c_members = _get_enum_members(_c_ops_enums[enum_name])
    py_members = _get_enum_members(_fallback_enums[enum_name])

    assert c_members == py_members, (
        f"{enum_name} mismatch:\n  c_ops:    {c_members}\n  fallback: {py_members}"
    )


# ── Coverage tests: every c_ops API must have a fallback ──


@pytest.mark.skipif(not HAS_C_OPS, reason="c_ops not available (no CUDA)")
def test_all_c_ops_callables_have_fallback() -> None:
    """Every public callable in c_ops must exist in
    torch_ops."""
    missing = sorted(set(_c_ops_callables) - set(_fallback_callables) - _EXCLUDED_FUNCS)
    assert not missing, f"c_ops callables missing from torch_ops: {missing}"


@pytest.mark.skipif(not HAS_C_OPS, reason="c_ops not available (no CUDA)")
def test_all_c_ops_enums_have_fallback() -> None:
    """Every public enum in c_ops must exist in
    torch_ops."""
    missing = sorted(set(_c_ops_enums) - set(_fallback_enums))
    assert not missing, f"c_ops enums missing from torch_ops: {missing}"


@pytest.mark.skipif(not HAS_C_OPS, reason="c_ops not available (no CUDA)")
def test_all_c_ops_descriptors_have_fallback() -> None:
    """Every public descriptor class in c_ops must exist in
    torch_ops."""
    missing = sorted(set(_c_ops_descs) - set(_fallback_descs) - _EXCLUDED_DESCS)
    assert not missing, f"c_ops descriptor classes missing from torch_ops: {missing}"


@pytest.mark.skipif(not HAS_C_OPS, reason="c_ops not available (no CUDA)")
@pytest.mark.parametrize(
    "cls_name",
    _shared_desc_names if _shared_desc_names else ["__placeholder__"],
)
def test_descriptor_class_parity(cls_name: str) -> None:
    """For every descriptor class shared between c_ops and the fallback,
    every public, non-callable attribute on the c_ops instance must also
    be present on the fallback instance.
    """
    if cls_name == "__placeholder__":
        pytest.skip("No shared descriptor classes found")

    # Attribute parity requires a no-arg constructor on both sides. Some plan
    # value structs (e.g. StagingCopy, KernelGroupSpec) only expose an
    # args-required constructor and no readable fields, so there is nothing to
    # compare — skip them rather than fail on construction.
    try:
        c_inst = _c_ops_descs[cls_name]()
    except TypeError:
        pytest.skip(f"{cls_name} has no no-arg constructor in c_ops")

    py_inst = _fallback_descs[cls_name]()

    def _public_data_attrs(inst: object) -> set[str]:
        return {
            a
            for a in dir(inst)
            if not a.startswith("_") and not callable(getattr(inst, a))
        }

    missing = sorted(_public_data_attrs(c_inst) - _public_data_attrs(py_inst))
    assert not missing, (
        f"{cls_name}: fallback is missing attributes present on c_ops: {missing}"
    )
