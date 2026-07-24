# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for chunk_local_cumsum_scalar and chunk_local_cumsum_vector
Triton kernels (GDN/FLA, kernels #14 and #15).

Source: vllm/model_executor/layers/fla/ops/cumsum.py
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.cumsum import (
    chunk_local_cumsum_scalar,
    chunk_local_cumsum_vector,
)
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="requires CUDA or XPU",
)

DEVICE = torch.device("xpu") if current_platform.is_xpu() else torch.device("cuda")


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def chunk_local_cumsum_scalar_ref(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
) -> torch.Tensor:
    B, T, H = g.shape
    BT = chunk_size
    NT = (T + BT - 1) // BT
    o = torch.zeros(B, T, H, dtype=torch.float32, device=g.device)
    for b in range(B):
        for h in range(H):
            for t_idx in range(NT):
                ts, te = t_idx * BT, min((t_idx + 1) * BT, T)
                chunk = g[b, ts:te, h].float()
                o[b, ts:te, h] = (
                    -chunk.cumsum(0) + chunk.sum() + chunk if reverse
                    else chunk.cumsum(0)
                )
    return o


def chunk_local_cumsum_vector_ref(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
) -> torch.Tensor:
    B, T, H, _ = g.shape
    BT = chunk_size
    NT = (T + BT - 1) // BT
    idx_full = torch.arange(BT, device=g.device)
    o = torch.zeros_like(g, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for t_idx in range(NT):
                ts, te = t_idx * BT, min((t_idx + 1) * BT, T)
                idx = idx_full[: te - ts]
                chunk = g[b, ts:te, h, :].float()
                mask = (
                    (idx[:, None] <= idx[None, :]).float() if reverse
                    else (idx[:, None] >= idx[None, :]).float()
                )
                o[b, ts:te, h, :] = mask @ chunk
    return o


# ---------------------------------------------------------------------------
# chunk_local_cumsum_scalar tests (kernel #14)
# ---------------------------------------------------------------------------

SCALAR_CONFIGS = [
    pytest.param(1, 64,  4, 64, False, id="B1_T64_H4_BT64_fwd"),
    pytest.param(1, 64,  4, 64, True,  id="B1_T64_H4_BT64_rev"),
    pytest.param(1, 128, 4, 64, False, id="B1_T128_H4_BT64"),
    pytest.param(2, 64,  4, 64, False, id="B2_T64_H4_BT64"),
    pytest.param(1, 64,  8, 64, False, id="B1_T64_H8_BT64"),
    pytest.param(1, 192, 4, 64, False, id="B1_T192_H4_BT64"),
    pytest.param(1, 64,  4, 32, False, id="B1_T64_H4_BT32"),
    pytest.param(2, 256, 8, 64, False, id="B2_T256_H8_BT64"),
]


@requires_gpu
@pytest.mark.parametrize("B,T,H,BT,reverse", SCALAR_CONFIGS)
@torch.inference_mode()
def test_cumsum_scalar_correctness(B, T, H, BT, reverse):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)
    o_tri = chunk_local_cumsum_scalar(g, chunk_size=BT, reverse=reverse)
    o_ref = chunk_local_cumsum_scalar_ref(g, chunk_size=BT, reverse=reverse)
    torch.testing.assert_close(o_tri.float(), o_ref.float(), rtol=1e-4, atol=1e-4)


@requires_gpu
@torch.inference_mode()
def test_cumsum_scalar_bfloat16():
    torch.manual_seed(7)
    g = torch.randn(1, 128, 4, device=DEVICE, dtype=torch.bfloat16)
    o_tri = chunk_local_cumsum_scalar(g, chunk_size=64, output_dtype=torch.float32)
    o_ref = chunk_local_cumsum_scalar_ref(g, chunk_size=64)
    torch.testing.assert_close(o_tri.float(), o_ref.float(), rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# chunk_local_cumsum_vector tests (kernel #15)
# ---------------------------------------------------------------------------

VECTOR_CONFIGS = [
    pytest.param(1, 64,  4, 32, 64, False, id="B1_T64_H4_S32_BT64_fwd"),
    pytest.param(1, 64,  4, 32, 64, True,  id="B1_T64_H4_S32_BT64_rev"),
    pytest.param(1, 128, 4, 32, 64, False, id="B1_T128_H4_S32_BT64"),
    pytest.param(2, 64,  4, 32, 64, False, id="B2_T64_H4_S32_BT64"),
    pytest.param(1, 64,  8, 16, 64, False, id="B1_T64_H8_S16_BT64"),
    pytest.param(1, 64,  4, 64, 64, False, id="B1_T64_H4_S64_BT64"),
    pytest.param(1, 192, 4, 32, 64, False, id="B1_T192_H4_S32_BT64"),
    pytest.param(1, 64,  4, 32, 32, False, id="B1_T64_H4_S32_BT32"),
    pytest.param(2, 256, 4, 32, 64, False, id="B2_T256_H4_S32_BT64"),
]


@requires_gpu
@pytest.mark.parametrize("B,T,H,S,BT,reverse", VECTOR_CONFIGS)
@torch.inference_mode()
def test_cumsum_vector_correctness(B, T, H, S, BT, reverse):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, S, device=DEVICE, dtype=torch.float32)
    o_tri = chunk_local_cumsum_vector(g, chunk_size=BT, reverse=reverse)
    o_ref = chunk_local_cumsum_vector_ref(g, chunk_size=BT, reverse=reverse)
    torch.testing.assert_close(o_tri.float(), o_ref.float(), rtol=1e-3, atol=1e-3)


@requires_gpu
@torch.inference_mode()
def test_cumsum_vector_bfloat16():
    torch.manual_seed(7)
    g = torch.randn(1, 128, 4, 32, device=DEVICE, dtype=torch.bfloat16)
    o_tri = chunk_local_cumsum_vector(g, chunk_size=64, output_dtype=torch.float32)
    o_ref = chunk_local_cumsum_vector_ref(g, chunk_size=64)
    torch.testing.assert_close(o_tri.float(), o_ref.float(), rtol=5e-3, atol=5e-3)
