# SPDX-License-Identifier: Apache-2.0
"""P39a — Persistent `A` buffer for FLA chunk_scaled_dot_kkt_fwd.

Problem
-------
`vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:144` does

    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

per call. Called ONCE per GDN layer per chunk during chunked prefill.
For Qwen3.6-35B-A3B: ~32 GDN-bearing layers × prefill chunks of
(B=1, T=4096, H=16, BT=64, fp32) = **16 MiB per call × 32 layers = 512 MiB
of per-step allocator churn** during long-context prefill.

Each alloc is small (16 MiB) but the churn is:
  * fragmentation-inducing (different chunk lengths → different slab sizes)
  * profiler-invisible (lazy alloc inside forward)
  * saturating at the edge (our yaml=0.93 boundary fails on 12 MiB alloc)

Fix
---
Single persistent `(1, max_batched_tokens, H, BT, fp32)` buffer per
(H, BT, device, dtype) key, shared across ALL GDN layers (sequential
forward invariant). Per call returns `_POOL[:B, :T, :H, :BT]` — a view
into the contiguous pool. Kernel writes into the view in-place (no
alloc, no churn).

Budget
------
Single buffer at max size:
    1 × 4096 × 16 × 64 × 4 = 16 MiB
shared across 32 layers vs the 16 MiB × 32 = 512 MiB churn. Profiler-
visible because we allocate at first call (warmup) and keep the tensor.

Platform guard
--------------
NVIDIA CUDA + SM 8.0+ (same as the rest of P2x). CPU / ROCm fallback
hits `torch.empty` — original behaviour preserved.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.3 implementation
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger("genesis.fla_kkt_buffer")


def _humanize(n: int) -> str:
    for unit, p in (("B", 0), ("KiB", 10), ("MiB", 20), ("GiB", 30)):
        if n < (1 << (p + 10)):
            if p == 0:
                return f"{n} B"
            return f"{n / (1 << p):.2f} {unit}"
    return f"{n / (1 << 30):.2f} GiB"


class FlaKktBufferManager:
    """Persistent pool for `chunk_scaled_dot_kkt_fwd`'s `A` output tensor.

    Keyed by (B_max, T_max, H, BT, device_str, dtype_str).
    Single pool is shared across all GDN layers because layers execute
    sequentially within one forward pass — no race, pointer-stable, CUDA-
    graph-safe.
    """

    _A_POOLS: dict[tuple, torch.Tensor] = {}

    @classmethod
    def should_apply(cls) -> bool:
        """Same platform gate as TurboQuantBufferManager."""
        from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
        if not is_nvidia_cuda():
            return False
        if not is_sm_at_least(8, 0):
            return False
        return True

    @classmethod
    def get_or_create_pool(
        cls,
        B_max: int,
        T_max: int,
        H: int,
        BT: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return the persistent `(B_max, T_max, H, BT)` pool (create on
        first call, reuse afterwards). None on non-NVIDIA platforms.
        """
        if not cls.should_apply():
            return None
        key = (B_max, T_max, H, BT, str(device), str(dtype))
        if key not in cls._A_POOLS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            shape = (B_max, T_max, H, BT)
            cls._A_POOLS[key] = GPB.get_or_create(
                namespace=f"fla_kkt_a|{key}",
                shape=shape, dtype=dtype, device=device, zero_init=False,
            )
            bytes_ = 1
            for d in shape:
                bytes_ *= d
            bytes_ *= dtype.itemsize if hasattr(dtype, "itemsize") else (
                torch.empty((), dtype=dtype).element_size()
            )
            log.info(
                "[P39a FLA KKT] allocated A pool %s (%s) on %s",
                list(shape), _humanize(bytes_), device,
            )
        return cls._A_POOLS[key]

    # Pool-key mapping: (H, BT, device_str, dtype_str) → pool tensor.
    # Keyed WITHOUT B/T so a single pool grows in-place as B/T demands
    # increase. This is the fast path — avoids keeping multiple pools
    # for the same (H, BT) just because early calls saw smaller batch
    # or chunk sizes.
    _A_POOLS_BY_SHAPE: dict[tuple, torch.Tensor] = {}
    # Track the GenesisPreallocBuffer namespace for each key so that on
    # pool-grow we can release the old namespace (prevents GPB registry
    # from accumulating stale pools across grows).
    _GPB_NAMESPACES: dict[tuple, str] = {}

    @classmethod
    def acquire(
        cls,
        B: int,
        T: int,
        H: int,
        BT: int,
        device: torch.device,
        dtype: torch.dtype,
        max_T: Optional[int] = None,
        max_B: Optional[int] = None,
    ) -> torch.Tensor:
        """Return `(B, T, H, BT)` view into the persistent pool.

        Pool is keyed by `(H, BT, device, dtype)` only; on first call at
        a given shape it's sized to `(max(B, max_B or B), max(T, max_T
        or T), H, BT)`; subsequent calls with larger `B` or `T` replace
        the pool with a grown copy (old pool is GC'd when refcount
        drops — safe in prefill, not CUDA-graph-captured).

        P39b reserve-before-cudagraph: callers SHOULD pass `max_T`
        and `max_B` hints so the pool grows to its final size on the
        first call (typically warmup with small batch). After that,
        the pool pointer is stable and subsequent calls are pure views.
        If hints aren't passed, we fall back to auto-grow behaviour
        (pointer-swap on growth — OK for prefill, BROKEN for
        CUDA-graph-captured regions).

        On non-NVIDIA / pre-Ampere platforms returns a fresh
        `torch.empty` — preserves upstream behaviour.
        """
        if not cls.should_apply():
            return torch.empty(
                (B, T, H, BT), device=device, dtype=dtype,
            )

        key = (H, BT, str(device), str(dtype))
        pool = cls._A_POOLS_BY_SHAPE.get(key)
        needed_B = max(B, max_B or 0)
        needed_T = max(T, max_T or 0)

        # Grow the pool if necessary. On the first call at this shape we
        # allocate (needed_B, needed_T, H, BT). On subsequent calls we
        # only re-allocate if the current pool is smaller.
        if pool is None or pool.shape[0] < needed_B or pool.shape[1] < needed_T:
            new_B = needed_B if pool is None else max(pool.shape[0], needed_B)
            new_T = needed_T if pool is None else max(pool.shape[1], needed_T)
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

            # Release any prior GPB entry at this key so the registry
            # doesn't accumulate stale tensors on each grow. Safe —
            # Python GC will reclaim once the old tensor's refcount drops.
            prev_entry_ns = cls._GPB_NAMESPACES.get(key)
            if prev_entry_ns is not None:
                try:
                    GPB.release(prev_entry_ns)
                except Exception:
                    # Best-effort: older GPB versions may not expose
                    # release(); fall through — pool tensor still gets
                    # GC'd when this_A_POOLS dict drops the ref below.
                    pass

            shape = (new_B, new_T, H, BT)
            ns = f"fla_kkt_a_by_shape|{key}"
            new_pool = GPB.get_or_create(
                namespace=ns,
                shape=shape, dtype=dtype, device=device, zero_init=False,
            )
            cls._A_POOLS_BY_SHAPE[key] = new_pool
            cls._GPB_NAMESPACES[key] = ns
            # Keep the legacy size-keyed cache in sync so
            # get_registry_info() reports consistently.
            cls._A_POOLS[(new_B, new_T, H, BT, str(device), str(dtype))] = new_pool

            # Compute bytes for the info log. `dtype.itemsize` is the
            # preferred API but some torch versions don't expose it on
            # the type object — fall back to a 0-d tensor probe.
            elem_bytes = (
                dtype.itemsize if hasattr(dtype, "itemsize")
                else torch.empty((), dtype=dtype).element_size()
            )
            total = 1
            for d in shape:
                total *= d
            total *= elem_bytes
            log.info(
                "[P39a FLA KKT] grew A pool → %s (%s) on %s",
                list(shape), _humanize(total), device,
            )
            pool = new_pool

        return pool[:B, :T, :H, :BT]

    @classmethod
    def get_registry_info(cls) -> dict:
        entries = []
        total = 0
        for key, t in cls._A_POOLS.items():
            b = t.element_size() * t.numel()
            total += b
            entries.append({
                "key": key, "bytes": b, "human": _humanize(b),
            })
        return {
            "num_pools": len(entries),
            "total_bytes": total,
            "total_human": _humanize(total),
            "entries": entries,
        }

    @classmethod
    def clear_for_tests(cls) -> None:
        cls._A_POOLS.clear()
        cls._A_POOLS_BY_SHAPE.clear()
        cls._GPB_NAMESPACES.clear()
