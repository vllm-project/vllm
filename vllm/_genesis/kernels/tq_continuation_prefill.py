# SPDX-License-Identifier: Apache-2.0
"""TurboQuant _continuation_prefill peak-memory helper (Patch 20).

Problem (JartX/vllm#11 — fixes #40420)
--------------------------------------
Long-prefill OOM crashes engine at ~185–234k actual tokens well below the
configured max-model-len on TurboQuant paths. Two redundant peak-memory
spikes in `_continuation_prefill`:

1. FP32 intermediate for inverse Hadamard rotation.
   Upstream: `k_flat.float() @ Pi` allocates a `cached_len * Hk * D * 4B`
   FP32 transient, followed by an FP16 copy. The FP32 widening has no
   accuracy benefit — keys were already reconstructed from 3–4 bit MSE
   indices, so FP16 roundoff on the rotation is orders of magnitude below
   the quantization noise already in the cache.

2. Redundant `.contiguous()` on transposed K/V views before `torch.cat`.
   `torch.cat` produces contiguous output regardless of input contiguity,
   so upstream's `.contiguous()` doubled the transient footprint for no
   reason.

Fix (this module)
-----------------
Provides `continuation_prefill_fp16_rotate(k_cached, Pi, Hk, D, cached_len)`
that:
  - Casts `Pi` to fp16 exactly once and caches it on the Pi.device
  - Does a single fp16 matmul (half the bytes of fp32 path)
  - Returns a non-contiguous transposed view — caller passes to torch.cat
    which materializes the final contiguous tensor with a single allocation

Verification (jhsmith409, RTX 5090 32 GiB + turboquant_4bit_nc):
    245k / 255k / 260k target tokens — CRASH → PASS, NIAH 45/45 unchanged.
Our setup (2× A5000 48 GiB + turboquant_k8v4):
    cliff at ~234k on max-model-len=262144 → post-patch target ≥ 250k.

Platform compatibility
----------------------
  NVIDIA CUDA   ✅ primary target (TurboQuant is CUDA-only)
  AMD ROCm      💤 TurboQuant not ported
  Intel XPU     💤 TurboQuant not ported
  CPU           💤 no TurboQuant kernel

Credits
-------
  - @JartX — TurboQuant author, JartX/vllm#11 original fix
  - @jhsmith409 — RTX 5090 32 GiB validation
  - Genesis Patch 20 — mirror into runtime patcher

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.tq_continuation_prefill")


# Cache Pi_half per (device, Pi.data_ptr()) so we only cast once ever.
# Key: (str(device), data_ptr); value: fp16 tensor.
_PI_HALF_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def get_pi_half(Pi: torch.Tensor) -> torch.Tensor:
    """Return an fp16 view/copy of Pi, cached per (device, data_ptr).

    If Pi is already fp16, return it directly (no copy).

    Args:
        Pi: Hadamard rotation matrix, any floating dtype, typically fp16
            or fp32, shape (D, D).

    Returns:
        fp16 tensor, same shape, same device as Pi.
    """
    if Pi.dtype == torch.float16:
        return Pi

    key = (str(Pi.device), Pi.data_ptr())
    cached = _PI_HALF_CACHE.get(key)
    if cached is not None and cached.shape == Pi.shape:
        return cached

    half = Pi.to(torch.float16)
    _PI_HALF_CACHE[key] = half
    return half


def continuation_prefill_fp16_rotate(
    k_cached: torch.Tensor,
    Pi: torch.Tensor,
    Hk: int,
    D: int,
    cached_len: int,
) -> torch.Tensor:
    """Rotate cached MSE keys back to original space in FP16 (no FP32 transient).

    This replaces the upstream 4-step FP32 path:

        k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D).float()
        k_flat = k_flat @ Pi
        k_cached_trim = (
            k_flat.to(torch.float16).reshape(Hk, cached_len, D).transpose(0, 1)
        )

    with a 2-step FP16 path:

        Pi_half = get_pi_half(Pi)                          # cached
        k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)  # already fp16
        return (k_flat @ Pi_half).reshape(Hk, cached_len, D).transpose(0, 1)

    Peak memory halved (no fp32 transient), no accuracy loss (keys were
    already 3–4 bit quantized — fp16 rounding is in the noise floor).

    Args:
        k_cached: Shape (1, Hk, max_len, D), fp16.
        Pi: Hadamard rotation, (D, D), fp16 or fp32.
        Hk: Per-rank num KV heads (after TP).
        D: head size.
        cached_len: Number of tokens currently populated in k_cached.

    Returns:
        Shape (cached_len, Hk, D), fp16, non-contiguous view —
        caller should torch.cat this (which materializes contiguous).
    """
    Pi_half = get_pi_half(Pi)
    k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
    k_flat = k_flat @ Pi_half
    return k_flat.reshape(Hk, cached_len, D).transpose(0, 1)


def continuation_prefill_k_view_fp8(
    k_cached: torch.Tensor,
    cached_len: int,
) -> torch.Tensor:
    """K-side view for the FP8 key branch (no rotation needed, no .contiguous()).

    Upstream emitted `.transpose(0, 1).contiguous()` on the FP8 branch; the
    `.contiguous()` is redundant since the caller concatenates via torch.cat
    which produces contiguous output regardless.

    Returns:
        Shape (cached_len, Hk, D), fp16/fp8, non-contiguous view.
    """
    return k_cached[0, :, :cached_len, :].transpose(0, 1)


def continuation_prefill_v_view(
    v_cached: torch.Tensor,
    cached_len: int,
) -> torch.Tensor:
    """V-side view (no rotation ever, no .contiguous())."""
    return v_cached[0, :, :cached_len, :].transpose(0, 1)


def should_apply() -> bool:
    """Platform guard — TurboQuant is NVIDIA CUDA + SM 8.0+ only."""
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def clear_pi_half_cache() -> None:
    """Clear the Pi_half cache — TESTS ONLY."""
    _PI_HALF_CACHE.clear()


def get_cache_info() -> dict:
    """Diagnostic info for observability."""
    total_bytes = 0
    entries = []
    for key, t in _PI_HALF_CACHE.items():
        b = t.element_size() * t.numel()
        total_bytes += b
        entries.append({
            "device": key[0],
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "bytes": b,
        })
    return {
        "num_entries": len(entries),
        "total_bytes": total_bytes,
        "entries": entries,
    }
