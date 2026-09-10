# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public router for the complete RDNA4 block-scaled FP8 GEMM family."""

import torch

from .rdna4_fp8_blockscale_common import SCALE_K
from .rdna4_fp8_blockscale_prefill import (
    PrefillConfig,
    _get_prefill_module,
)
from .rdna4_fp8_blockscale_small_m import _run_small_m, _use_tiled_decode_split
from .rdna4_fp8_blockscale_wave32 import (
    Wave32Config,
    _get_wave32_module,
    select_wave32_config,
)
from .runtime import run_compiled


def validate_tensors(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the tensor contract shared by every RDNA4 implementation."""
    tensors = (a, weight, a_scale, weight_scale)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must be on the GPU")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must be rank two")
    if a.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError("RDNA4 block-FP8 operands must use torch.float8_e4m3fn")
    if a_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise TypeError("RDNA4 block-FP8 scales must use torch.float32")
    if any(t.device != a.device for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must share a device")
    arch = getattr(torch.cuda.get_device_properties(a.device), "gcnArchName", "")
    if not (arch.startswith("gfx1200") or arch.startswith("gfx1201")):
        raise ValueError(
            f"RDNA4 block-FP8 route requires gfx1200 or gfx1201, got {arch!r}"
        )

    m, k = a.shape
    n, weight_k = weight.shape
    if m <= 0:
        raise ValueError(f"RDNA4 block-FP8 requires positive M, got {m}")
    if n <= 0 or n % SCALE_K:
        raise ValueError(
            f"RDNA4 block-FP8 requires positive N divisible by 128, got {n}"
        )
    if k <= 0 or k % SCALE_K:
        raise ValueError(
            f"RDNA4 block-FP8 requires positive K divisible by 128, got {k}"
        )
    if weight_k != k:
        raise ValueError(f"weight K mismatch: A has K={k}, weight has K={weight_k}")
    if (
        not a.is_contiguous()
        or not a_scale.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("A and both scale tensors must be contiguous")
    if weight.stride(1) != 1:
        raise ValueError("weight must have contiguous K rows (weight.stride(1) == 1)")
    if weight.stride(0) < k or weight.stride(0) % 4:
        raise ValueError("weight row stride must be at least K and divisible by 4")
    if tuple(a_scale.shape) != (m, k // SCALE_K):
        raise ValueError(
            f"activation scale shape must be {(m, k // SCALE_K)}, "
            f"got {tuple(a_scale.shape)}"
        )
    if tuple(weight_scale.shape) != (n // SCALE_K, k // SCALE_K):
        raise ValueError(
            f"weight scale shape must be {(n // SCALE_K, k // SCALE_K)}, "
            f"got {tuple(weight_scale.shape)}"
        )
    # Raw-buffer descriptors hold a 32-bit byte count; scale spans are smaller.
    if max(m * k, n * weight.stride(0), 2 * m * n) >= 1 << 32:
        raise ValueError("RDNA4 block-FP8 buffer spans must be smaller than 4 GiB")
    return m, n, k


def rdna4_fp8_block_scaled_mm(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Run the complete RDNA4 block-scaled FP8 stack for any positive M."""
    m, n, k = validate_tensors(a, weight, a_scale, weight_scale)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    stream = torch.cuda.current_stream(a.device)
    config: PrefillConfig | Wave32Config
    if m <= 16 and n >= 12288:
        config = PrefillConfig(32, 128, 1, 2, 4, k_rotate=2)
        module = _get_prefill_module(n, k, weight.stride(0), config)
        run_compiled(module, a, weight, a_scale, weight_scale, out, m, stream)
        return out
    if 17 <= m <= 64 and n > 4096:
        # Reuse weights across the short-prefill rows and stagger K traversal
        # across N tiles. Runtime M avoids one compiled kernel per token count.
        if n * k <= 64 * 1024 * 1024:
            config = Wave32Config(
                tile_m=32 if m <= 32 else 64,
                group_m=1,
                a_prefetch=2,
                b_prefetch=2,
                k_rotate=2,
                reg_m=1 if m <= 32 else 2,
            )
            module = _get_wave32_module(n, k, weight.stride(0), config)
        else:
            config = PrefillConfig(32 if m <= 32 else 64, 128, 1, 2, 2, k_rotate=2)
            module = _get_prefill_module(n, k, weight.stride(0), config)
        run_compiled(module, a, weight, a_scale, weight_scale, out, m, stream)
        return out
    if m <= 64 or _use_tiled_decode_split(m, n, k):
        return _run_small_m(a, weight, a_scale, weight_scale, out, stream, m, n, k)

    if m <= 512 and k <= 512:
        config = Wave32Config(
            tile_m=64,
            group_m=8,
            a_prefetch=2,
            b_prefetch=2,
            k_rotate=2,
        )
        module = _get_wave32_module(n, k, weight.stride(0), config)
        run_compiled(module, a, weight, a_scale, weight_scale, out, m, stream)
        return out

    config = select_wave32_config(m, n, k)
    module = _get_wave32_module(n, k, weight.stride(0), config)
    run_compiled(module, a, weight, a_scale, weight_scale, out, m, stream)
    return out


__all__ = [
    "rdna4_fp8_block_scaled_mm",
]
