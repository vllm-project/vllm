# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for solve_tril Triton kernels (FLA/GDN block-triangular inverse).

Tests the three kernels dispatched by solve_tril() based on block size (BT):
  - solve_tril_16x16_kernel        (BT=16)
  - merge_16x16_to_32x32_inverse   (BT=32)
  - merge_16x16_to_64x64_inverse   (BT=64)

Each kernel computes (I + A)^{-1} where A is strictly lower triangular,
stored in chunked layout [B, T, H, BT]. Validated against torch.linalg.inv.
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def make_strictly_lower_tri(
    B: int, T: int, H: int, BT: int, dtype: torch.dtype
) -> torch.Tensor:
    """Create a strictly lower triangular A in chunked layout [B, T, H, BT].

    Each contiguous BT-row slice (within one chunk) represents a BT×BT
    strictly lower triangular matrix. Values are scaled down for stability.
    """
    A = torch.randn(B, T, H, BT, dtype=dtype, device=DEVICE) * 0.1
    NT = T // BT
    with torch.no_grad():
        for b in range(B):
            for t_idx in range(NT):
                ts = t_idx * BT
                for h in range(H):
                    for i in range(BT):
                        A[b, ts + i, h, i:] = 0.0
    return A


def solve_tril_ref(
    A: torch.Tensor, output_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Reference: compute (I + A)^{-1} via torch.linalg.inv per block.

    A: [B, T, H, BT] — chunked strictly lower triangular.
    """
    B, T, H, BT = A.shape
    NT = T // BT
    Ai = torch.zeros_like(A, dtype=output_dtype, device=A.device)

    for b in range(B):
        for t_idx in range(NT):
            ts = t_idx * BT
            for h in range(H):
                block = torch.zeros(BT, BT, device=A.device, dtype=torch.float32)
                for i in range(BT):
                    block[i, :BT] = A[b, ts + i, h, :BT].float()

                M = torch.eye(BT, device=A.device, dtype=torch.float32) + block
                M_inv = torch.linalg.inv(M)

                for i in range(BT):
                    Ai[b, ts + i, h, :BT] = M_inv[i, :BT].to(output_dtype)
    return Ai


# ─── BT=16: solve_tril_16x16_kernel ─────────────────────────────────────


@pytest.mark.parametrize(
    "B,T,H,dtype",
    [
        (1, 16, 1, torch.float32),
        (1, 32, 2, torch.float32),
        (2, 64, 4, torch.float32),
        (2, 64, 8, torch.float32),
        (1, 16, 1, torch.bfloat16),
        (1, 32, 4, torch.bfloat16),
        (2, 64, 4, torch.bfloat16),
    ],
)
def test_solve_tril_16x16(B: int, T: int, H: int, dtype: torch.dtype) -> None:
    """Verify solve_tril_16x16_kernel against torch.linalg.inv reference."""
    BT = 16
    A = make_strictly_lower_tri(B, T, H, BT, dtype)

    result = solve_tril(A, output_dtype=torch.float32)
    ref = solve_tril_ref(A, output_dtype=torch.float32)

    torch.testing.assert_close(result.cpu(), ref.cpu(), atol=1e-4, rtol=1e-3)


# ─── BT=32: merge_16x16_to_32x32_inverse_kernel ─────────────────────────


@pytest.mark.parametrize(
    "B,T,H,dtype",
    [
        (1, 32, 1, torch.float32),
        (1, 64, 2, torch.float32),
        (2, 64, 4, torch.float32),
        (2, 128, 8, torch.float32),
        (1, 32, 1, torch.bfloat16),
        (1, 64, 4, torch.bfloat16),
        (2, 64, 4, torch.bfloat16),
    ],
)
def test_solve_tril_32x32(B: int, T: int, H: int, dtype: torch.dtype) -> None:
    """Verify merge_16x16_to_32x32_inverse_kernel against reference."""
    BT = 32
    A = make_strictly_lower_tri(B, T, H, BT, dtype)

    result = solve_tril(A, output_dtype=torch.float32)
    ref = solve_tril_ref(A, output_dtype=torch.float32)

    torch.testing.assert_close(result.cpu(), ref.cpu(), atol=1e-3, rtol=1e-2)


# ─── BT=64: merge_16x16_to_64x64_inverse_kernel ─────────────────────────


@pytest.mark.parametrize(
    "B,T,H,dtype",
    [
        (1, 64, 1, torch.float32),
        (1, 128, 2, torch.float32),
        (2, 128, 4, torch.float32),
        (2, 256, 8, torch.float32),
        (1, 64, 1, torch.bfloat16),
        (1, 128, 4, torch.bfloat16),
        (2, 128, 4, torch.bfloat16),
    ],
)
def test_solve_tril_64x64(B: int, T: int, H: int, dtype: torch.dtype) -> None:
    """Verify merge_16x16_to_64x64_inverse_kernel against reference."""
    BT = 64
    A = make_strictly_lower_tri(B, T, H, BT, dtype)

    result = solve_tril(A, output_dtype=torch.float32)
    ref = solve_tril_ref(A, output_dtype=torch.float32)

    torch.testing.assert_close(result.cpu(), ref.cpu(), atol=1e-3, rtol=1e-2)
