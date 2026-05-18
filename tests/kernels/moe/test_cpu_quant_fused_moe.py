# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU quantized fused MoE kernels (FP8 W8A16 / INT8 W8A8)."""

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


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _prepack_experts(w: torch.Tensor) -> torch.Tensor:
    """VNNI-prepack expert weights via ``convert_weight_packed``."""
    return torch.ops._C.convert_weight_packed(w)


# ===========================================================================
# FP8 W8A16 block-scaled fused MoE
# ===========================================================================

BLOCK_SIZE = [128, 128]  # [block_n, block_k]

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_SCALE = _FP8_INFO.max  # 448.0
FACTOR_FOR_SCALE = 1e-3


def _block_dequant_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-dequantize FP8 weight [E, N, K] -> float [E, N, K]."""
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
    """Reference FP8 W8A16 block-scaled fused MoE in pure torch."""
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
    """Generate random FP8 MoE weights with random block scales."""
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


FP8_NUM_TOKENS = [1, 2, 64, 121]
FP8_MOE_CONFIGS = [
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


@pytest.mark.parametrize("M", FP8_NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", FP8_MOE_CONFIGS)
@pytest.mark.parametrize("seed", [0])
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
        a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, BLOCK_SIZE
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
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )
    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)

    # Test inplace=True produces identical output
    out_inplace = ops.fused_experts_cpu(
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        True,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )
    torch.testing.assert_close(out_inplace, out, atol=0, rtol=0)


# ===========================================================================
# INT8 W8A8 per-channel MoE
# ===========================================================================


def _quantize_per_channel(w):
    """Symmetric per-channel INT8 quantisation. w: [N, K] -> (int8, scale)."""
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    w_q = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_q, scale.float()


def _quantize_per_token(x):
    """Symmetric per-token INT8 quantisation. x: [M, K] -> (int8, scale)."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale.float()


def _ref_int8_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids):
    """Reference INT8 W8A8 per-channel fused MoE in pure torch."""
    B, D = a.shape
    topk = topk_ids.size(1)

    out = torch.zeros(B, topk, w2.shape[1], dtype=torch.float32)
    for b in range(B):
        for t in range(topk):
            eid = topk_ids[b, t].item()

            x = a[b : b + 1].float()
            x_q, x_s = _quantize_per_token(x)
            ic = torch.matmul(x_q.float(), w1[eid].float().t())
            ic = ic * x_s * w1_s[eid].view(1, -1)
            ic = _silu_and_mul(ic)

            ic_q, ic_s = _quantize_per_token(ic)
            oc = torch.matmul(ic_q.float(), w2[eid].float().t())
            oc = oc * ic_s * w2_s[eid].view(1, -1)
            out[b, t] = oc.squeeze(0)

    result = (out * topk_weight.unsqueeze(-1)).sum(dim=1)
    return result.to(a.dtype)


def _make_int8_moe_weights(E, N, K):
    factor = 1e-2
    w1_f = (torch.randn(E, 2 * N, K) - 0.5) * 2
    w2_f = (torch.randn(E, K, N) - 0.5) * 2

    w1_q_list, w1_s_list = [], []
    w2_q_list, w2_s_list = [], []
    for e in range(E):
        q, s = _quantize_per_channel(w1_f[e])
        w1_q_list.append(q)
        w1_s_list.append(s)
        q, s = _quantize_per_channel(w2_f[e])
        w2_q_list.append(q)
        w2_s_list.append(s)

    return (
        torch.stack(w1_q_list),
        torch.stack(w2_q_list),
        torch.stack(w1_s_list) * factor,
        torch.stack(w2_s_list) * factor,
    )


INT8_NUM_TOKENS = [1, 2, 64, 121]
INT8_MOE_CONFIGS = [
    # (N, K, E, topk)
    (256, 512, 8, 2),
    (512, 256, 8, 2),
    (512, 512, 8, 4),
    (768, 2048, 8, 2),
]


@pytest.mark.parametrize("M", INT8_NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", INT8_MOE_CONFIGS)
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("is_vnni", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_int8_w8a8_cpu_fused_moe(M, N, K, E, topk, seed, is_vnni, inplace):
    """Test fused_experts_cpu INT8 W8A8 against torch reference."""
    set_random_seed(seed)

    a = torch.randn(M, K, dtype=torch.bfloat16) / (0.5 * K**0.5)
    w1_q, w2_q, w1_s, w2_s = _make_int8_moe_weights(E, N, K)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = _ref_int8_moe(a, w1_q, w2_q, w1_s, w2_s, topk_weight, topk_ids)

    w1 = _prepack_experts(w1_q) if is_vnni else w1_q
    w2 = _prepack_experts(w2_q) if is_vnni else w2_q

    out = ops.fused_experts_cpu(
        a.clone(),
        w1,
        w2,
        topk_weight,
        topk_ids,
        inplace,
        ops.CPUQuantMethod.INT8_W8A8,
        w1_s,
        w2_s,
        None,
        None,
        None,
        is_vnni,
    )
    torch.testing.assert_close(
        ref_out.bfloat16(),
        out,
        atol=2e-1,
        rtol=2e-1,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
