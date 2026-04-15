# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-mode KV cache quantization backends.

The core attention kernel
(:mod:`vllm.v1.attention.ops.triton_unified_attention`) handles modes
``NONE``, ``FP8_PER_TENSOR``, ``INT8_PER_TOKEN_HEAD`` and
``FP8_PER_TOKEN_HEAD`` directly via constexpr branches.  Backends
registered here own:

  * the **write side** for any mode that needs more than a plain copy
    (per-token-head absmax, asymmetric INT4 with zero-point packing,
    INT2 Lloyd-Max + Hadamard, …); and
  * the **attention read side** for sub-byte packed modes (INT4 / INT2)
    whose inner loop is structurally different from the core kernel
    (split-dot, centroid lookup, etc.).

Adding a new quantization mode
------------------------------
1. Add a new value to :class:`KVQuantMode` in
   ``vllm/v1/kv_cache_interface.py``.
2. Add a new entry to ``_MODULES`` below mapping the mode to a module path.
3. Create a new file under ``quant_kv/`` that defines a subclass of
   :class:`QuantKVBackend` and calls :func:`register` at module level.
   If the mode can use the core attention kernel, override only
   ``reshape_and_cache`` / ``allocate_scale_caches``; otherwise also
   override ``unified_attention``.
"""

from __future__ import annotations

import importlib

from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVBackend
from vllm.v1.kv_cache_interface import KVQuantMode

__all__ = ["QuantKVBackend", "register", "get_backend", "has_backend"]

_REGISTRY: dict[KVQuantMode, QuantKVBackend] = {}

# Mode -> dotted module path.  Adding a new mode = one line here + one new file.
# INT8 and FP8 per-token-head share a single module because they only differ
# in the (QUANT_MAX, QUANT_MIN) constants and the cache storage dtype, which
# Triton infers from the cache pointer.
_MODULES: dict[KVQuantMode, str] = {
    KVQuantMode.INT8_PER_TOKEN_HEAD: (
        "vllm.v1.attention.ops.triton_quant_kv.int8_fp8_per_token_head"
    ),
    KVQuantMode.FP8_PER_TOKEN_HEAD: (
        "vllm.v1.attention.ops.triton_quant_kv.int8_fp8_per_token_head"
    ),
    KVQuantMode.INT4_PER_TOKEN_HEAD: (
        "vllm.v1.attention.ops.triton_quant_kv.packed_per_token_head"
    ),
    KVQuantMode.INT2_PER_TOKEN_HEAD: (
        "vllm.v1.attention.ops.triton_quant_kv.packed_per_token_head"
    ),
}


def register(backend: QuantKVBackend) -> None:
    if backend.mode in _REGISTRY:
        existing = _REGISTRY[backend.mode].__class__.__name__
        new = backend.__class__.__name__
        if existing != new:
            raise RuntimeError(
                f"Duplicate QuantKVBackend registration for "
                f"{backend.mode.name}: {existing} vs {new}"
            )
        return
    _REGISTRY[backend.mode] = backend


def get_backend(mode: KVQuantMode) -> QuantKVBackend:
    """Lazy-import and return the backend for *mode*.

    Raises ``ValueError`` if no backend module is configured for *mode*.
    """
    if mode == KVQuantMode.NONE:
        raise ValueError("KVQuantMode.NONE is the unquantized path and has no backend")
    if mode not in _REGISTRY:
        module_path = _MODULES.get(mode)
        if module_path is None:
            raise ValueError(
                f"No QuantKVBackend module configured for {mode.name}.  "
                f"Add an entry to _MODULES in "
                f"vllm/v1/attention/ops/quant_kv/__init__.py."
            )
        importlib.import_module(module_path)
        if mode not in _REGISTRY:
            raise RuntimeError(
                f"Module {module_path} did not register a backend for "
                f"{mode.name}.  Each backend module must call "
                f"`quant_kv.register(MyBackend())` at the bottom."
            )
    return _REGISTRY[mode]


def has_backend(mode: KVQuantMode) -> bool:
    """Return True if *mode* has a backend (loaded or lazily available)."""
    return mode in _REGISTRY or mode in _MODULES
