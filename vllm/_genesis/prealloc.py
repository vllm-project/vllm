# SPDX-License-Identifier: Apache-2.0
"""Genesis safe pre-allocation framework — graph-safe, profiler-visible.

Motivation
----------
CUDA graph capture requires IDENTICAL memory pointers across replays. Lazy
allocation inside captured forward paths (e.g. `torch.empty()` gated by a
runtime condition) causes one of two failure modes:

  1. Forced graph re-capture on every shape change (Genesis Patch 19 revert
     symptom: −30% throughput, 188× stdev).
  2. Silent stale-pointer reads causing data corruption (hardest class).

Additionally, vLLM's memory profiler (`profile_run` → `max_memory_allocated`)
only SEES allocations that occur during its warmup dummy-batch forward. Any
allocation gated on a condition unreachable during warmup (e.g. long-context
continuation prefill) is profiler-invisible, leading to KV cache sizing that
undersizes VRAM headroom → engine OOM at production scale (Genesis #40420).

This module provides a canonical framework for SAFE pre-allocation:

  1. Rule 1 (Profiler visibility): allocate during __init__ or
     _ensure_on_device (both run during profile_run warmup).
  2. Rule 2 (Graph safety): return the SAME tensor pointer on every call —
     downstream code must slice (`buf[:n]`), never reallocate.
  3. Rule 3 (Eager/captured awareness): works for both eager prefill and
     captured decode paths without modification.

Usage
-----
From any Genesis kernel module:

    from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

    # In Attention.__init__ or _ensure_on_device:
    self._dequant_buf = GPB.get_or_create(
        namespace="tq_k_dequant",
        shape=(num_kv_heads, head_size, max_alloc_len),
        dtype=torch.bfloat16,
        device=device,
    )

    # In forward (captured or eager):
    working = GPB.slice_to(self._dequant_buf, actual_len, dim=-1)
    working.zero_()  # in-place, pointer-stable

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("genesis.prealloc")


class GenesisPreallocBuffer:
    """Class-level registry of pre-allocated tensors, keyed by namespace+shape+dtype+device.

    Thread-safety: each worker process has its own _REGISTRY — not shared
    across TP ranks. Init order guaranteed by Attention.__init__ invocation
    during model loading (before any forward).

    Lifecycle:
      1. First call with a unique key → allocate fresh tensor in registry.
      2. Subsequent calls with same key → return cached tensor (same pointer).
      3. Process shutdown → tensors freed with Python GC (no explicit cleanup).

    Anti-patterns (WILL break CUDA graphs):
      - Do NOT call get_or_create() from inside forward() for NEW namespaces.
        The allocation must happen during profile_run warmup.
      - Do NOT modify existing tensor dimensions. If you need larger, create
        a new namespace with bigger shape — never resize an existing buffer.
    """

    _REGISTRY: dict[tuple, torch.Tensor] = {}

    @classmethod
    def get_or_create(
        cls,
        namespace: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
        zero_init: bool = False,
    ) -> torch.Tensor:
        """Get existing buffer or allocate fresh one.

        Args:
            namespace: Logical name for the buffer (e.g. "tq_k_dequant",
                "gdn_core_attn_out"). Used as part of cache key.
            shape: Maximum shape buffer will ever need. Downstream code
                slices to actual size per-call.
            dtype: Element type (bf16, fp16, fp32, int32, etc.).
            device: Target device. Converted to string for key hashing.
            zero_init: If True, use torch.zeros (slower but deterministic).
                Default False uses torch.empty (faster, uninitialized memory).

        Returns:
            Pre-allocated tensor, pointer-stable across calls.
        """
        key = (namespace, tuple(shape), dtype, str(device))
        buf = cls._REGISTRY.get(key)
        if buf is None:
            alloc_fn = torch.zeros if zero_init else torch.empty
            buf = alloc_fn(shape, dtype=dtype, device=device)
            cls._REGISTRY[key] = buf
            # NO LOGGING HERE — this function is reachable from torch.compile-
            # traced forward paths (P26/P28 call chains). `torch.dynamo`
            # rejects `logging.Logger` method calls during tracing
            # ("logging.Logger method not supported for non-export cases").
            # Observability is provided via `get_registry_info()` called
            # from outside the traced region.

        return buf

    @classmethod
    def slice_to(
        cls,
        buf: torch.Tensor,
        n: int,
        dim: int = 0,
    ) -> torch.Tensor:
        """Slice pre-allocated buffer to actual size for this call.

        Returns a view (no copy). The view shares storage with the
        parent buffer, so the pointer is stable across calls (critical
        for CUDA graph capture).

        Args:
            buf: Tensor from get_or_create().
            n: Actual size needed for this call.
            dim: Dimension along which to slice.

        Returns:
            View of buf[:n] along specified dim.

        Raises:
            AssertionError with helpful message if n exceeds buf shape.
        """
        if n < 0:
            raise ValueError(f"[Genesis prealloc] slice_to got negative n={n}")
        if n > buf.shape[dim]:
            raise AssertionError(
                f"[Genesis prealloc] requested slice n={n} exceeds buffer "
                f"shape={buf.shape} on dim={dim}. Increase max shape at "
                f"__init__ time. This indicates a config mismatch: the "
                f"pre-allocated max size does not cover the actual workload."
            )
        return buf.narrow(dim, 0, n)

    @classmethod
    def get_registry_info(cls) -> dict[str, Any]:
        """Return diagnostic info about all pre-allocated buffers.

        Useful for monitoring and debug:
            import json
            print(json.dumps(GenesisPreallocBuffer.get_registry_info(),
                             indent=2, default=str))
        """
        total_bytes = 0
        entries = []
        for key, buf in cls._REGISTRY.items():
            namespace, shape, dtype, device = key
            try:
                nbytes = buf.element_size() * buf.numel()
            except Exception:
                nbytes = 0
            total_bytes += nbytes
            entries.append({
                "namespace": namespace,
                "shape": list(shape),
                "dtype": str(dtype),
                "device": device,
                "bytes": nbytes,
                "size_human": _humanize_bytes(nbytes),
            })
        return {
            "total_buffers": len(entries),
            "total_bytes": total_bytes,
            "total_human": _humanize_bytes(total_bytes),
            "entries": entries,
        }

    @classmethod
    def clear_for_tests(cls):
        """Clear the registry. TESTS ONLY — calling at runtime breaks
        CUDA graph capture (buffer pointers become invalid).
        """
        import warnings
        if any("pytest" in s for s in _get_stack_module_names()):
            cls._REGISTRY.clear()
        else:
            warnings.warn(
                "[Genesis prealloc] clear_for_tests() called outside pytest; "
                "this is almost certainly a bug. Buffers are being invalidated "
                "at runtime which will break CUDA graph replay.",
                RuntimeWarning,
                stacklevel=2,
            )
            cls._REGISTRY.clear()

    @classmethod
    def release(cls, namespace: str) -> bool:
        """Drop a single namespace entry from the registry.

        Intended for pool-growth scenarios (e.g.
        `FlaKktBufferManager.acquire`) where a stale tensor should be
        GC-eligible once the manager has switched to a larger pool.
        Returns True if the namespace was present and removed.

        NOT a CUDA-graph-safe operation at runtime if a captured graph
        still references the tensor pointer — callers are responsible
        for ensuring no live graph depends on the released buffer.
        """
        return cls._REGISTRY.pop(namespace, None) is not None


def _humanize_bytes(nbytes: int) -> str:
    """Convert byte count to human-readable string (B / KiB / MiB / GiB)."""
    if nbytes < 1024:
        return f"{nbytes} B"
    for unit, power in [("KiB", 10), ("MiB", 20), ("GiB", 30), ("TiB", 40)]:
        limit = 1 << (power + 10)
        if nbytes < limit:
            return f"{nbytes / (1 << power):.2f} {unit}"
    return f"{nbytes / (1 << 40):.2f} TiB"


def _get_stack_module_names() -> list[str]:
    """Return module names from current call stack (for test detection)."""
    try:
        import inspect
        return [frame.frame.f_globals.get("__name__", "") or ""
                for frame in inspect.stack()]
    except Exception:
        return []
