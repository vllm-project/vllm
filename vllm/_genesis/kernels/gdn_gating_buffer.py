# SPDX-License-Identifier: Apache-2.0
"""P46 — Persistent buffers for `fused_gdn_gating` `g` / `beta_output`.

Problem
-------
`vllm/model_executor/layers/mamba/gdn_linear_attn.py:1195-1196`
allocates two tensors per call:

    g = torch.empty(1, batch, num_heads, dtype=torch.float32, ...)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, ...)

Qwen3.6-35B-A3B has 48 GDN-bearing layers. On decode (batch=1) these
are TINY (~kilobytes each). Bytes saved per step = negligible. But
the ALLOCATOR overhead — two fresh `torch.empty` calls per layer per
step × 48 layers × 250 tok/s = **~24 000 allocator ops/sec** just
for these two tensors.

Fix
---
Module-level pool keyed by `(batch, num_heads, dtype, device)`. On
first call at a given shape, allocate both `g` and `beta_output`
persistent. Subsequent calls with the same shape (overwhelmingly
common for stable workloads — batch size + head count rarely shift
per forward) return the SAME tensors, kernel writes into them in-place.
No `.zero_()` needed — Triton kernel unconditionally writes every
position.

CUDA graph safety
-----------------
- Pool allocation happens at FIRST call → if that's during warmup
  (before capture), pointer is stable across all captured graphs.
- `_resolve_default_shape_once()` helper lets wiring pre-warm
  during profile_run so the pool is profiler-visible AND pointer
  already stable before any capture.
- Pool "grows" (i.e. re-allocates into different key) only if a
  DIFFERENT (batch, num_heads, dtype) is seen — at which point the
  OLD pool stays live under the old key; we don't mutate the existing
  tensor, so any captured graph holding it continues to work.

Scope notes
-----------
- byte-exact output vs upstream: ✅ yes — Triton kernel writes all
  positions unconditionally, allocated-content doesn't matter (as
  with `torch.empty`).
- Platform guard: NVIDIA CUDA SM 8.0+ (same as rest of P2x).
- Default-on: this patch has NO semantic change, only allocator
  reduction. Active by default on NVIDIA.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.7 implementation (default-on on NVIDIA)
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.gdn_gating_buffer")


class GdnGatingBufferManager:
    """Module-level pool for `fused_gdn_gating`'s two output tensors.

    Thread-safety: class-level mutable dicts; each TP worker process
    has its own state (vLLM uses spawn). No cross-process sync needed.

    Lifecycle per shape:
      first call at (B, H, dt_g, dt_b, device) → alloc g + beta_output
      second call same key → return cached tensors (identity `is`)
      different key → alloc new pair under new key, old pair stays live
    """

    _G_POOLS: dict[tuple, torch.Tensor] = {}
    _BETA_POOLS: dict[tuple, torch.Tensor] = {}

    @classmethod
    def should_apply(cls) -> bool:
        """Same platform gate as rest of P2x — NVIDIA SM ≥ 8.0."""
        from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
        if not is_nvidia_cuda():
            return False
        if not is_sm_at_least(8, 0):
            return False
        return True

    @classmethod
    def acquire_g(
        cls,
        batch: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the persistent `g` buffer for this shape key.

        Shape returned: `(1, batch, num_heads)` — matches upstream
        `fused_gdn_gating` contract line-for-line.

        On platform-skip returns a fresh `torch.empty` (preserves
        upstream semantics on CPU / ROCm / pre-Ampere).
        """
        if not cls.should_apply():
            return torch.empty(
                (1, batch, num_heads), dtype=dtype, device=device,
            )
        key = (batch, num_heads, str(dtype), str(device))
        t = cls._G_POOLS.get(key)
        if t is None:
            t = torch.empty(
                (1, batch, num_heads), dtype=dtype, device=device,
            )
            cls._G_POOLS[key] = t
            log.info(
                "[P46] allocated persistent `g` buffer (1,%d,%d) dtype=%s "
                "device=%s", batch, num_heads, dtype, device,
            )
        return t

    @classmethod
    def acquire_beta(
        cls,
        batch: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the persistent `beta_output` buffer for this shape key.

        Shape: `(1, batch, num_heads)`; dtype matches caller's `b.dtype`
        (typically FP16 or BF16).
        """
        if not cls.should_apply():
            return torch.empty(
                (1, batch, num_heads), dtype=dtype, device=device,
            )
        key = (batch, num_heads, str(dtype), str(device))
        t = cls._BETA_POOLS.get(key)
        if t is None:
            t = torch.empty(
                (1, batch, num_heads), dtype=dtype, device=device,
            )
            cls._BETA_POOLS[key] = t
            log.info(
                "[P46] allocated persistent `beta_output` buffer (1,%d,%d) "
                "dtype=%s device=%s", batch, num_heads, dtype, device,
            )
        return t

    @classmethod
    def get_registry_info(cls) -> dict:
        """Diagnostic snapshot — used by `memory_metrics.py`."""
        def _bytes(t: torch.Tensor) -> int:
            return t.element_size() * t.numel()
        g_bytes = sum(_bytes(t) for t in cls._G_POOLS.values())
        beta_bytes = sum(_bytes(t) for t in cls._BETA_POOLS.values())
        return {
            "num_g_pools": len(cls._G_POOLS),
            "num_beta_pools": len(cls._BETA_POOLS),
            "total_bytes": g_bytes + beta_bytes,
            "g_entries": [
                {"key": k, "bytes": _bytes(t)}
                for k, t in cls._G_POOLS.items()
            ],
            "beta_entries": [
                {"key": k, "bytes": _bytes(t)}
                for k, t in cls._BETA_POOLS.items()
            ],
        }

    @classmethod
    def clear_for_tests(cls) -> None:
        cls._G_POOLS.clear()
        cls._BETA_POOLS.clear()
