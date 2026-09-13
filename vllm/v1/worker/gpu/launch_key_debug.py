# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in receipts for comparing Uno warmup and serving Triton launches.

Triton's cache key is deliberately not reconstructed here. When enabled, this
module invokes the same binder and ``compute_cache_key`` call that
``JITFunction.run`` uses immediately afterwards, then records the resulting
specialisation key, options, bound argument layout, and launch grid. The
receipt can therefore prove whether startup warmed the exact specialisation a
first real request needs.

This is a diagnosis tool, not a monitor: it is enabled only with
``VLLM_UNO_LAUNCH_KEY_DEBUG=1`` and has no production-path wrapper when the
flag is absent.
"""

from __future__ import annotations

import functools
import json
import os
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from typing import Any, ParamSpec, TypeVar

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _read_enabled() -> bool:
    """Validate the opt-in diagnostic flag once at module import."""
    value = os.environ.get("VLLM_UNO_LAUNCH_KEY_DEBUG")
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError("VLLM_UNO_LAUNCH_KEY_DEBUG must be 0 or 1")


_ENABLED = _read_enabled()
_phase: ContextVar[str | None] = ContextVar("vllm_uno_launch_key_phase", default=None)
_serving_ready = False

P = ParamSpec("P")
T = TypeVar("T")


def _json_value(value: Any) -> Any:
    """Return deterministic, non-data-bearing runtime metadata for a receipt."""
    if isinstance(value, torch.Tensor):
        pointer = value.data_ptr() if value.numel() else 0
        return {
            "alignment": pointer % 256 if pointer else 0,
            "device": str(value.device),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "storage_offset": value.storage_offset(),
            "stride": list(value.stride()),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return repr(value)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def _object_fields(value: Any) -> Any:
    """Serialize both dict-backed and slots-backed Triton option objects."""
    if is_dataclass(value):
        return _json_value(asdict(value))
    try:
        return _json_value(vars(value))
    except TypeError:
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        fields = {
            slot: _json_value(getattr(value, slot))
            for slot in slots
            if isinstance(slot, str) and hasattr(value, slot)
        }
        return fields or repr(value)


def _launch_record(
    kernel_name: str,
    kernel: Any,
    grid: tuple[int, ...],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the actual Triton cache key without compiling or launching it.

    This intentionally mirrors Triton 3.8's ``JITFunction.run`` through
    ``compute_cache_key``. It runs immediately before the actual call, which
    repeats the calculation and then launches the same key.
    """
    from triton import knobs  # type: ignore[import-untyped]
    from triton.runtime import driver  # type: ignore[import-untyped]
    from triton.runtime.jit import compute_cache_key  # type: ignore[import-untyped]

    launch_kwargs = dict(kwargs)
    launch_kwargs["debug"] = (
        launch_kwargs.get("debug", kernel.debug) or knobs.runtime.debug
    )
    launch_kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

    device = driver.active.get_current_device()
    kernel_cache, kernel_key_cache, target, _backend, binder = kernel.device_caches[
        device
    ]
    bound_args, specialization, options = binder(*args, **launch_kwargs)
    specialization = list(specialization)
    if knobs.runtime.add_stages_inspection_hook is not None:
        _inspection_key, inspection_hash = knobs.runtime.add_stages_inspection_hook()
        specialization.append(f'("custom_pipeline", {inspection_hash})')
    key = compute_cache_key(kernel_key_cache, specialization, options)

    return {
        "args": {str(name): _json_value(value) for name, value in bound_args.items()},
        "cache_hit_before_launch": key in kernel_cache,
        "grid": _json_value(grid),
        "kernel": kernel_name,
        "options": _object_fields(options),
        "specialization": _json_value(specialization),
        "target": repr(target),
        "triton_key": str(key),
    }


class _LaunchKeyReceipts:
    """Keep unique warmup keys and emit one comparison per served kernel."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._warmup: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._served_kernels: set[str] = set()

    def add(self, phase: str, record: dict[str, Any]) -> dict[str, Any] | None:
        kernel = str(record["kernel"])
        key = str(record["triton_key"])
        with self._lock:
            if phase == "warmup":
                if key in self._warmup[kernel]:
                    return None
                self._warmup[kernel][key] = record
                return {"record": record}
            if phase != "served" or kernel in self._served_kernels:
                return None
            self._served_kernels.add(kernel)
            warmup = list(self._warmup[kernel].values())
            return {
                "compare": {
                    "kernel": kernel,
                    "served": record,
                    "served_key_was_warmed": key in self._warmup[kernel],
                    "warmup": warmup,
                },
                "record": record,
            }


_receipts = _LaunchKeyReceipts()


def launch_key_phase(phase: str) -> AbstractContextManager[None]:
    """Label launches performed by a known startup operation."""
    if not _ENABLED:
        return nullcontext()

    @contextmanager
    def _set_phase():
        token = _phase.set(phase)
        try:
            yield
        finally:
            _phase.reset(token)

    return _set_phase()


def mark_launch_key_serving_ready() -> None:
    """Start labeling model-runner sampling calls as actual serving."""
    if not _ENABLED:
        return
    global _serving_ready
    _serving_ready = True
    logger.info("Uno launch-key receipt: startup complete; first serving keys armed.")


def serving_launches(fn: Callable[P, T]) -> Callable[P, T]:
    """Label real model-runner sampling without changing normal execution."""
    if not _ENABLED:
        return fn

    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        if not _serving_ready:
            return fn(*args, **kwargs)
        with launch_key_phase("served"):
            return fn(*args, **kwargs)

    return wrapped


def record_triton_launch(
    kernel_name: str,
    kernel: Any,
    grid: tuple[int, ...],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Log an exact Triton key for the active diagnosis phase, if any."""
    phase = _phase.get()
    if not _ENABLED or phase is None:
        return
    try:
        record = _launch_record(kernel_name, kernel, grid, args, kwargs)
    except Exception as exc:  # Debug must never change a serving result.
        logger.warning(
            "UNO_LAUNCH_KEY_ERROR kernel=%s phase=%s error=%r",
            kernel_name,
            phase,
            exc,
        )
        return
    emitted = _receipts.add(phase, record)
    if emitted is None:
        return
    if "record" in emitted:
        logger.info(
            "UNO_LAUNCH_KEY phase=%s %s", phase, _canonical_json(emitted["record"])
        )
    if "compare" in emitted:
        logger.info("UNO_LAUNCH_KEY_COMPARE %s", _canonical_json(emitted["compare"]))


__all__ = [
    "launch_key_phase",
    "mark_launch_key_serving_ready",
    "record_triton_launch",
    "serving_launches",
]
