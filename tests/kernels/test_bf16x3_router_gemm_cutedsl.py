# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SM100 BF16x3 router GEMM."""

import pytest
import torch

from vllm.utils.import_utils import has_cutedsl


def _requires_sm100_cutedsl():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("bf16x3 router GEMM requires SM100-class GPU")
    if not has_cutedsl():
        pytest.skip("cutedsl (cutlass) not installed")


@pytest.mark.parametrize(
    ("num_tokens", "hidden_dim", "num_experts"),
    [
        (48, 6144, 128),
        (96, 3072, 256),
        (129, 3072, 17),
        # long K: the large-M split-K heuristic leaves chains of 15 and 32
        # K-tiles, above num_tmem_acc (multi-chunk epilogue); the odd-experts
        # case below covers the small-M kernel's multi-chunk path (12 K-tiles)
        (1024, 8192, 256),
        (2048, 8192, 256),
        # small-M path, including several BM tiles above 128 tokens (M2/HV4)
        (64, 6144, 128),
        (128, 2816, 256),
        (200, 3072, 256),
        (256, 2816, 256),
        # large-M path, cta_group=1, from each shape's switch point
        (129, 6144, 128),
        (300, 2816, 256),
        (512, 6144, 128),
        (512, 3072, 256),
        # large-M path, cta_group=2
        (1024, 6144, 128),
        (2304, 4096, 192),
        (2304, 2816, 256),
        # partial 256-token pair-tile tail
        (8200, 6144, 128),
        # odd experts: large-M ineligible at any token count, small-M path
        (4096, 3072, 17),
    ],
)
def test_bf16x3_router_gemm_matches_reference(
    num_tokens: int, hidden_dim: int, num_experts: int
):
    _requires_sm100_cutedsl()
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
        bf16x3_router_gemm,
    )

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device="cuda")
    # Match the observed router weight scale
    w *= 0.053
    out = bf16x3_router_gemm(x, w)
    # FP64 reference: the FP32 reference itself drifts by ~5e-6 at N=2048
    ref = torch.nn.functional.linear(x.double(), w.double())

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    assert torch.mean(torch.abs(out.double() - ref)).item() < 5e-6
