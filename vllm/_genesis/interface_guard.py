# SPDX-License-Identifier: Apache-2.0
"""P49 — Interface contract validation for Genesis runtime-rebind patches.

Problem
-------
Our existing wiring layer (P22, P28, P38, P39a, P40, P7b) relies on
`_import_tq_impl()`-style helpers that return a class IF the upstream
module + class name resolve. They DON'T verify that the class has the
EXPECTED interface (attrs, methods, shapes).

This is unsafe under two drift scenarios:

1. **Upstream refactor.** Future vLLM release keeps
   `TurboQuantAttentionImpl` class name but renames
   `_mse_bytes` → `_mse_bytes_packed` OR drops `tq_config` attribute.
   Our `_genesis_continuation_prefill` body reads these names directly,
   blowing up at first long-context request with `AttributeError`.

2. **Different model with overlapping class name.** Hypothetical
   upstream addition of TurboQuant support to a different model family
   that uses `TurboQuantAttentionImpl` class but with DIFFERENT shape
   convention (e.g., 3-D dequant buffer instead of 4-D). Our P22
   attaches 4-D buffer; the model's engine expects 3-D; slicing fails
   at forward.

Defense layers
--------------
**Layer 1 (pre-flight, in `apply()`):** This module's
`validate_impl(impl_cls, required_attrs, required_methods,
optional_attrs, role)` raises `GenesisInterfaceMismatch` if any
required attr is missing. Wiring catches the exception and returns
`("skipped", "interface drift: <details>")`. Genesis continues
without the patch — upstream lazy path takes over.

**Layer 2 (runtime, in patch body):** `assert_shape_compat(tensor,
expected_shape, role)` raises at forward-path critical points; catch
sites in our patch body call `_genesis_p38_original` (or equivalent
fallback) on mismatch instead of propagating the exception.

Usage pattern in a wiring patch
-------------------------------
```python
from vllm._genesis.interface_guard import (
    validate_impl, GenesisInterfaceMismatch,
)

def apply() -> tuple[str, str]:
    impl_cls = _import_tq_impl()
    if impl_cls is None:
        return "skipped", "TurboQuant backend not available"
    try:
        validate_impl(
            impl_cls,
            role="TurboQuantAttentionImpl for P38",
            required_attrs={
                "num_heads": int,
                "num_kv_heads": int,
                "head_size": int,
                "tq_config": "TurboQuantConfig",
                "_mse_bytes": int,
                "_val_data_bytes": int,
            },
            required_methods=[
                "_continuation_prefill",
                "_flash_attn_varlen",
            ],
        )
    except GenesisInterfaceMismatch as e:
        return "skipped", f"interface drift: {e}"
    ...  # proceed with rebind
```

Scope
-----
- Pure-Python helpers; no torch dependency at module import time.
- Never mutates inspected objects.
- Lightweight — single validate call per apply(), cost < 1ms.
- Cannot detect SEMANTIC drift (e.g., same signature but different
  math) — only NAME / TYPE / SHAPE drift. Semantic drift requires
  numerical equivalence testing, out of scope.

Platform notes
--------------
- CPU-only: all checks are ast-level introspection → works everywhere.
- GPU-required checks (e.g., `buffer.device.type == "cuda"`) are
  expressed via `required_attrs={..., "_cuda_marker": <guard>}` if
  needed; not relied upon by core logic.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.8 — foundation for every future runtime-rebind patch
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Iterable, Mapping, Optional

log = logging.getLogger("genesis.interface_guard")


class GenesisInterfaceMismatch(Exception):
    """Raised when a target class/module doesn't satisfy Genesis's
    expected interface.

    Wiring patches catch this at `apply()` time and return
    `("skipped", str(e))` — the engine continues without the patch.
    """

    def __init__(
        self,
        role: str,
        missing_attrs: Iterable[str] = (),
        wrong_type_attrs: Iterable[tuple[str, str, str]] = (),
        missing_methods: Iterable[str] = (),
        message_suffix: str = "",
    ):
        missing_attrs = list(missing_attrs)
        wrong_type_attrs = list(wrong_type_attrs)
        missing_methods = list(missing_methods)

        parts = [f"{role!r} interface mismatch"]
        if missing_attrs:
            parts.append(f"missing attrs {missing_attrs}")
        if wrong_type_attrs:
            parts.append(
                f"wrong-type attrs "
                + ", ".join(
                    f"{n}(got {got}, expected {exp})"
                    for n, got, exp in wrong_type_attrs
                )
            )
        if missing_methods:
            parts.append(f"missing methods {missing_methods}")
        if message_suffix:
            parts.append(message_suffix)
        super().__init__("; ".join(parts))

        self.role = role
        self.missing_attrs = missing_attrs
        self.wrong_type_attrs = wrong_type_attrs
        self.missing_methods = missing_methods


# Sentinel for "any non-None value is OK" — useful when we only check
# presence, not type. E.g. `required_attrs={"tq_config": ANY}`.
ANY = object()


def _type_matches(value: Any, expected: Any) -> bool:
    """True iff `value` is an instance of `expected` (or ANY sentinel,
    or a STRING describing the expected type-name).

    String form `expected: str` is useful when the expected type is a
    class defined in a module we don't want to import eagerly (e.g.,
    `TurboQuantConfig` lives in `vllm.model_executor.layers.quantization.
    turboquant.config` — a path that may not resolve on CPU unit-test
    envs). Matching by class name avoids the import.
    """
    if expected is ANY:
        return value is not None
    if isinstance(expected, str):
        # Accept by class-name match, or by any base class name.
        cls = type(value)
        return (
            cls.__name__ == expected
            or any(b.__name__ == expected for b in cls.__mro__)
        )
    if isinstance(expected, type):
        return isinstance(value, expected)
    if isinstance(expected, tuple):
        return any(_type_matches(value, e) for e in expected)
    # Unknown expected form → treat as pure presence check.
    return value is not None


def validate_impl(
    impl: Any,
    *,
    role: str,
    required_attrs: Optional[Mapping[str, Any]] = None,
    optional_attrs: Optional[Mapping[str, Any]] = None,
    required_methods: Optional[Iterable[str]] = None,
) -> None:
    """Raise `GenesisInterfaceMismatch` if `impl` doesn't have the
    required interface. Silent on success.

    Args:
        impl: The class OR instance to inspect.
        role: Human-readable label used in error messages.
        required_attrs: Mapping `name -> expected_type`. Type can be
            a class, tuple of classes, ANY sentinel, or a STRING
            (matched against class name in MRO — avoids eager imports).
        optional_attrs: Same shape; values reported as
            `note: optional_attr X missing` via `log.info` but don't
            raise.
        required_methods: Iterable of method names. Each must be
            `callable(getattr(impl, name))`.
    """
    required_attrs = dict(required_attrs or {})
    optional_attrs = dict(optional_attrs or {})
    required_methods = list(required_methods or [])

    missing_attrs: list[str] = []
    wrong_type_attrs: list[tuple[str, str, str]] = []
    missing_methods: list[str] = []

    for name, expected in required_attrs.items():
        if not hasattr(impl, name):
            missing_attrs.append(name)
            continue
        value = getattr(impl, name)
        if not _type_matches(value, expected):
            got_type = type(value).__name__
            exp_type = (
                expected.__name__ if isinstance(expected, type)
                else str(expected)
            )
            wrong_type_attrs.append((name, got_type, exp_type))

    for name in required_methods:
        member = getattr(impl, name, None)
        if not callable(member):
            missing_methods.append(name)

    # Optional attrs — log but don't fail
    for name, expected in optional_attrs.items():
        if not hasattr(impl, name):
            log.info(
                "[Genesis iface] %s: optional attr %r missing "
                "(not fatal)", role, name,
            )

    if missing_attrs or wrong_type_attrs or missing_methods:
        raise GenesisInterfaceMismatch(
            role=role,
            missing_attrs=missing_attrs,
            wrong_type_attrs=wrong_type_attrs,
            missing_methods=missing_methods,
        )


def validate_method_signature(
    impl: Any,
    method_name: str,
    *,
    role: str,
    expected_min_params: int,
    expected_param_names: Optional[Iterable[str]] = None,
) -> None:
    """Raise `GenesisInterfaceMismatch` if `impl.method_name`'s signature
    doesn't match our expectations.

    Catches cases where upstream rename'd a method param from
    `cached_len` to `num_computed_tokens` etc. — our text-patch /
    replacement would pass wrong kwargs otherwise.

    Args:
        expected_min_params: Minimum NON-self parameter count. Variadic
            (*args/**kwargs) count as one each.
        expected_param_names: Optional subset of param names that MUST
            be present (positional or keyword).
    """
    method = getattr(impl, method_name, None)
    if method is None or not callable(method):
        raise GenesisInterfaceMismatch(
            role=role,
            missing_methods=[method_name],
            message_suffix=f"(signature check on absent method)",
        )
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError) as e:
        # Some C-extension methods don't expose signature — best-effort.
        log.info(
            "[Genesis iface] %s.%s: signature introspection failed (%s); "
            "skipping signature check", role, method_name, e,
        )
        return

    params = list(sig.parameters.values())
    # Filter out 'self' / 'cls' from bound-method signatures
    if params and params[0].name in ("self", "cls"):
        params = params[1:]

    if len(params) < expected_min_params:
        raise GenesisInterfaceMismatch(
            role=role,
            message_suffix=(
                f"method {method_name!r} has {len(params)} params, "
                f"expected at least {expected_min_params}"
            ),
        )

    if expected_param_names:
        present_names = {p.name for p in params}
        missing = [n for n in expected_param_names if n not in present_names]
        if missing:
            raise GenesisInterfaceMismatch(
                role=role,
                message_suffix=(
                    f"method {method_name!r} missing params {missing}"
                ),
            )


def assert_shape_compat(
    tensor: Any,
    *,
    role: str,
    expected_ndim: Optional[int] = None,
    min_shape: Optional[tuple] = None,
    expected_dtype: Optional[Any] = None,
) -> None:
    """Runtime assertion for shape/dtype compatibility inside a patch
    body. Raise `GenesisInterfaceMismatch` on mismatch.

    Callers should `except GenesisInterfaceMismatch` and fall back to
    upstream original method via saved `_genesis_*_original` attr.

    Example (inside P38 body):
        assert_shape_compat(
            k_buf, role="P38 k_dequant_buf",
            expected_ndim=4, min_shape=(1, 1, alloc_len, 1),
        )
    """
    # Lazy-import torch so this module stays CPU-import-safe.
    try:
        import torch
    except ImportError:
        return  # nothing we can check

    if not isinstance(tensor, torch.Tensor):
        raise GenesisInterfaceMismatch(
            role=role,
            message_suffix=f"expected torch.Tensor, got {type(tensor).__name__}",
        )

    if expected_ndim is not None and tensor.dim() != expected_ndim:
        raise GenesisInterfaceMismatch(
            role=role,
            message_suffix=(
                f"shape ndim {tensor.dim()} != expected {expected_ndim} "
                f"(shape={tuple(tensor.shape)})"
            ),
        )

    if min_shape is not None:
        actual = tuple(tensor.shape)
        if len(actual) != len(min_shape):
            raise GenesisInterfaceMismatch(
                role=role,
                message_suffix=(
                    f"shape {actual} rank != min_shape {min_shape}"
                ),
            )
        for i, (a, m) in enumerate(zip(actual, min_shape)):
            if m is not None and a < m:
                raise GenesisInterfaceMismatch(
                    role=role,
                    message_suffix=(
                        f"shape[{i}]={a} < min {m} (full shape {actual}, "
                        f"min {min_shape})"
                    ),
                )

    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            raise GenesisInterfaceMismatch(
                role=role,
                message_suffix=(
                    f"dtype {tensor.dtype} != expected {expected_dtype}"
                ),
            )


def describe_impl(impl: Any, *, role: str = "target") -> dict:
    """Diagnostic snapshot — dump public-attr names + method names
    + selected type annotations. Used for logging on mismatch OR for
    fingerprint hashing in fast-path drift detection.

    Returns a dict suitable for JSON dump.
    """
    attrs: dict[str, str] = {}
    methods: list[str] = []
    for name in sorted(dir(impl)):
        if name.startswith("__"):
            continue
        try:
            value = getattr(impl, name)
        except Exception:
            continue
        if callable(value):
            methods.append(name)
        else:
            attrs[name] = type(value).__name__
    return {
        "role": role,
        "class": getattr(type(impl), "__name__", "?"),
        "attrs": attrs,
        "methods": methods,
    }
