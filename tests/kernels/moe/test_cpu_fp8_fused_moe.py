# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU FP8 W8A16 block-scaled fused MoE kernel."""

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

import vllm._custom_ops as ops  # noqa: E402

if not hasattr(torch.ops._C, "fused_experts_cpu"):
    pytest.skip("fused_experts_cpu op not available", allow_module_level=True)


BLOCK_SIZE = [128, 128]  # [block_n, block_k]

# FP8 weight generation parameters
_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_SCALE = _FP8_INFO.max  # 448.0
FACTOR_FOR_SCALE = 1e-3

# Tolerance for FP8 W8A16
FP8_W8A16_ATOL = 1e-2
FP8_W8A16_RTOL = 1e-2


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _block_dequant_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-dequantize FP8 weight [E, N, K] → float [E, N, K].

    Each (block_n × block_k) tile is multiplied by its per-block scale.
    """
    E, N, K = weight.shape
    block_n, block_k = block_size
    pad_N = (block_n - N % block_n) % block_n
    pad_K = (block_k - K % block_k) % block_k

    if pad_N > 0 or pad_K > 0:
        weight = F.pad(weight, (0, pad_K, 0, pad_N))

    n_tiles = math.ceil(N / block_n)
    k_tiles = math.ceil(K / block_k)

    weight_block = (
        weight.view(E, n_tiles, block_n, k_tiles, block_k)
        .permute(0, 1, 3, 2, 4)
        .float()
        .contiguous()
    )
    weight_scaled = (
        (weight_block * scales.view(E, n_tiles, k_tiles, 1, 1))
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if pad_N > 0 or pad_K > 0:
        weight_scaled = weight_scaled.view(E, N + pad_N, K + pad_K)
        weight_scaled = weight_scaled[..., :N, :K].contiguous()
    else:
        weight_scaled = weight_scaled.view(E, N, K)
    return weight_scaled


def ref_w8a16_block_fp8_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Reference FP8 W8A16 block-scaled fused MoE in pure torch.

    Steps:
      1. Block-dequant FP8 weights → float
      2. For each expert: matmul → SiLU+Mul → matmul
      3. Weighted sum across top-k experts
    """
    B, D = a.shape
    topk = topk_ids.size(1)

    w1_dq = _block_dequant_weight(w1, w1_s, block_size)
    w2_dq = _block_dequant_weight(w2, w2_s, block_size)

    a_exp = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).float()
    out = torch.zeros(B * topk, w2_dq.shape[1], dtype=torch.float32)

    topk_weight_flat = topk_weight.view(-1)
    topk_ids_flat = topk_ids.view(-1)

    for i in range(w1_dq.shape[0]):
        mask = topk_ids_flat == i
        if mask.sum():
            ic0 = torch.matmul(a_exp[mask], w1_dq[i].transpose(0, 1))
            ic1 = _silu_and_mul(ic0)
            out[mask] = torch.matmul(ic1, w2_dq[i].transpose(0, 1))

    return (
        (out.view(B, -1, w2_dq.shape[1]) * topk_weight_flat.view(B, -1, 1))
        .sum(dim=1)
        .to(a.dtype)
    )


def _make_fp8_moe_weights(
    E: int,
    N: int,
    K: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random FP8 MoE weights with random block scales.

    Weight generation follows SGLang: ``randn * FP8_SCALE`` → clamp → cast.
    Scales are small random values (``FACTOR_FOR_SCALE``), independent of
    the actual weight magnitudes — this is sufficient to test the kernel's
    block-dequant + matmul correctness.

    Returns: (w1, w2, w1_s, w2_s)
    """
    block_n, block_k = block_size

    w1 = (
        (torch.randn(E, 2 * N, K) * FP8_SCALE)
        .clamp(min=-FP8_SCALE, max=FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        (torch.randn(E, K, N) * FP8_SCALE)
        .clamp(min=-FP8_SCALE, max=FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )

    w1_s = (
        torch.randn(E, math.ceil(2 * N / block_n), math.ceil(K / block_k))
        * FACTOR_FOR_SCALE
    )
    w2_s = (
        torch.randn(E, math.ceil(K / block_n), math.ceil(N / block_k))
        * FACTOR_FOR_SCALE
    )
    return w1, w2, w1_s, w2_s


def _prepack_experts(w: torch.Tensor) -> torch.Tensor:
    """VNNI-prepack each expert's weight via ``convert_weight_packed``."""
    return torch.stack(
        [torch.ops._C.convert_weight_packed(w[e]) for e in range(w.shape[0])]
    )


NUM_TOKENS = [1, 2, 64, 121]
# (M, intermediate_size N, hidden_size K, num_experts E, topk)
MoE_CONFIGS = [
    (256, 512, 8, 2),
    (256, 512, 8, 4),
    (512, 256, 8, 2),
    (512, 256, 8, 4),
    (512, 512, 8, 2),
    (512, 512, 8, 4),
    (768, 2048, 8, 2),
    (768, 2048, 8, 4),
    (768, 2048, 128, 8),
]
SEEDS = [0]


@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", MoE_CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_w8a16_block_fp8_cpu_fused_moe(M, N, K, E, topk, seed):
    """Test fused_experts_cpu FP8 W8A16 against dequantised torch reference."""
    set_random_seed(seed)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    w1, w2, w1_s, w2_s = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_w8a16_block_fp8_moe(
        a,
        w1,
        w2,
        w1_s,
        w2_s,
        topk_weight,
        topk_ids,
        BLOCK_SIZE,
    )

    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)

    # Test inplace=False against reference
    out = ops.fused_experts_cpu(
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        False,
        False,
        True,
        w1_s,
        w2_s,
        BLOCK_SIZE,
        None,
        None,
        True,
    )
    torch.testing.assert_close(
        ref_out.bfloat16(),
        out,
        atol=FP8_W8A16_ATOL,
        rtol=FP8_W8A16_RTOL,
    )

    # Test inplace=True produces identical output
    out_inplace = ops.fused_experts_cpu(
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        True,
        False,
        True,
        w1_s,
        w2_s,
        BLOCK_SIZE,
        None,
        None,
        True,
    )
    torch.testing.assert_close(out_inplace, out, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
