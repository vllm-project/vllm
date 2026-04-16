# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Walsh-Hadamard transform and Randomized Hadamard Transform (RHT).

Used by INT2 (Hadamard + Lloyd-Max centroids) and INT4 (single RHT +
asymmetric quantization) per-token-head KV cache backends.

Three-tier dispatch for ``fast_hadamard_transform``:
  1. Hadacore CUDA Tensor Core kernel (sm_80+).
  2. Triton MMA matmul kernel (CUDA fallback + ROCm MFMA/WMMA path).
  3. PyTorch butterfly (CPU and any GPU/dtype combo Triton can't take).
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# Hadacore (CUDA tensor core kernel) availability check
# ---------------------------------------------------------------------------
# Hadacore's CUDA impl is only registered when built for sm_80+, but the
# schema def is unconditional — on ROCm ``hasattr`` is True yet dispatch
# would crash, so we also gate on ``is_cuda()``.
_HADACORE_AVAILABLE: bool | None = None


def _hadacore_available() -> bool:
    global _HADACORE_AVAILABLE
    if _HADACORE_AVAILABLE is None:
        _HADACORE_AVAILABLE = current_platform.is_cuda() and hasattr(
            torch.ops._C, "hadacore_transform"
        )
    return _HADACORE_AVAILABLE


# ---------------------------------------------------------------------------
# Cached Hadamard matrices (one per (size, dtype, device) tuple)
# ---------------------------------------------------------------------------
_HADAMARD_MATRIX_CACHE: dict[tuple[int, torch.dtype, str], torch.Tensor] = {}


def _get_hadamard_matrix(
    d: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (d, dtype, str(device))
    cached = _HADAMARD_MATRIX_CACHE.get(key)
    if cached is None:
        H = torch.ones(1, 1, dtype=torch.float32, device=device)
        while H.shape[0] < d:
            H = torch.cat(
                [
                    torch.cat([H, H], dim=1),
                    torch.cat([H, -H], dim=1),
                ],
                dim=0,
            )
        cached = H.to(dtype).contiguous()
        _HADAMARD_MATRIX_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Triton MMA Hadamard kernel (Tier 2)
# ---------------------------------------------------------------------------
@triton.jit
def _hadamard_mma_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    n_rows,
    stride_x_row: tl.int64,
    stride_x_col: tl.int64,
    stride_o_row: tl.int64,
    stride_o_col: tl.int64,
    BLOCK_M: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    row_mask = rows < n_rows

    x = tl.load(
        x_ptr + rows[:, None] * stride_x_row + cols[None, :] * stride_x_col,
        mask=row_mask[:, None],
        other=0.0,
    )
    H = tl.load(h_ptr + cols[:, None] * D + cols[None, :])

    out = tl.dot(x, H, out_dtype=tl.float32).to(x.dtype)

    tl.store(
        out_ptr + rows[:, None] * stride_o_row + cols[None, :] * stride_o_col,
        out,
        mask=row_mask[:, None],
    )


# H is D×D bf16 = 2·D² bytes of LDS.  AMD CDNA has 64 KiB LDS, so D ≤ 128
# (32 KiB) leaves room for input + accumulator.  Larger D falls back.
_TRITON_HADAMARD_MIN_D = 16
_TRITON_HADAMARD_MAX_D = 128


def _triton_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    orig_shape = x.shape
    orig_dtype = x.dtype

    work_dtype = torch.bfloat16 if orig_dtype == torch.float32 else orig_dtype
    x2d = x.contiguous().to(work_dtype).reshape(-1, d)
    out2d = torch.empty_like(x2d)
    n_rows = x2d.shape[0]
    H_mat = _get_hadamard_matrix(d, work_dtype, x.device)

    BLOCK_M = 16
    grid = (triton.cdiv(n_rows, BLOCK_M),)
    # num_stages=1: the kernel has no loop, so default 3-stage pipelining
    # would triple-buffer H and blow the AMD LDS budget.
    _hadamard_mma_kernel[grid](
        x2d,
        H_mat,
        out2d,
        n_rows,
        x2d.stride(0),
        x2d.stride(1),
        out2d.stride(0),
        out2d.stride(1),
        BLOCK_M=BLOCK_M,
        D=d,
        num_stages=1,
        num_warps=4,
    )
    return out2d.reshape(orig_shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard Transform along the last dimension.

    H_d × x where H_d × H_d = d × I.  Last dim must be a power of 2.

    Three-tier dispatch:
      1. Hadacore CUDA Tensor Core kernel (sm_80+).
      2. Triton MMA matmul kernel (CUDA fallback + ROCm MFMA/WMMA path).
      3. PyTorch butterfly (CPU and any GPU/dtype combo Triton can't take).
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Requires power-of-2 dim, got {d}"

    # Tier 1 — hadacore on CUDA.
    if _hadacore_available() and 0 < d <= (1 << 15):
        from vllm import _custom_ops as ops

        # hadacore returns x @ (H/√d); rescale to the unnormalized H × x
        # convention the INT2/INT4 scale math is calibrated to.
        rescale = d**0.5
        if x.dtype in (torch.float16, torch.bfloat16):
            y = ops.hadacore_transform(x.contiguous().clone(), inplace=True)
            return y * rescale
        # fp32 → bf16 round-trip; precision loss is irrelevant before
        # INT2/INT4 quantization.
        orig_dtype = x.dtype
        x_bf16 = x.contiguous().to(torch.bfloat16)
        y_bf16 = ops.hadacore_transform(x_bf16, inplace=True)
        return y_bf16.to(orig_dtype) * rescale

    # Tier 2 — Triton MMA kernel (covers ROCm via MFMA/WMMA codegen, and
    # also CUDA when hadacore is unavailable).
    if (
        x.is_cuda
        and _TRITON_HADAMARD_MIN_D <= d <= _TRITON_HADAMARD_MAX_D
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return _triton_hadamard_transform(x)

    # Tier 3 — PyTorch butterfly (CPU / unsupported dtype / D < 16).
    h = 1
    while h < d:
        xv = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = xv[..., 0, :]
        b = xv[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(x.shape)
        h <<= 1
    return x


# ---------------------------------------------------------------------------
# Randomized Hadamard Transform (used by INT4)
# ---------------------------------------------------------------------------
# Deterministic ±1 signs for Randomized Hadamard Transform.
# RHT = H × D × x  (sign flip + Hadamard).  Breaks residual structure
# in KV vectors, improving quantization quality.
_RHT_SIGNS_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _get_rht_signs(d: int, round_idx: int, device: torch.device) -> torch.Tensor:
    """Return a cached deterministic ±1 sign vector of length *d*."""
    key = (d, round_idx, str(device))
    if key not in _RHT_SIGNS_CACHE:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x9E3779B9 + round_idx * 0x517CC1B7)
        signs = (
            2.0 * torch.bernoulli(torch.full((d,), 0.5, device="cpu"), generator=gen)
            - 1.0
        )
        _RHT_SIGNS_CACHE[key] = signs.to(device)
    return _RHT_SIGNS_CACHE[key]


def single_rht(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Single Randomized Hadamard Transform: H × D₁ × x.

    Used by INT4 per-token-head quantization to gaussianize data
    before asymmetric quantization.
    """
    d = x.shape[-1]
    d1 = _get_rht_signs(d, 0, x.device)
    if inverse:
        return fast_hadamard_transform(x) * d1
    else:
        return fast_hadamard_transform(x * d1)


# Backwards-compat alias (was a private name in the old location).
_single_rht = single_rht
