# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU MXFP4 W4A16 fused MoE kernel."""

import sys

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops  # noqa: E402
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if not hasattr(torch.ops._C, "fused_experts_cpu"):
    pytest.skip("fused_experts_cpu op not available", allow_module_level=True)

# Tolerance for MXFP4 W4A16
MXFP4_ATOL = 1e-2
MXFP4_RTOL = 1e-2


class MXFP4QuantizeUtil:
    """MXFP4 quantization utility."""

    E2M1_max = 6.0
    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
    block_size = 32

    @classmethod
    def quantize(cls, input: torch.Tensor) -> tuple:
        """Quantize BF16 tensor to MXFP4 packed uint8 format."""

        def cast_fp4(x):
            sign = torch.sign(x)
            sign_bit = (2 - sign) // 2
            ord_ = torch.sum(
                (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1
            )
            fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
            return fp4_val

        def fuse_uint4_to_uint8(x):
            left_side = x[..., 0::2]
            right_side = x[..., 1::2]
            new_data = right_side.clone() << 4
            new_data[..., : left_side.shape[-1]] += left_side
            return new_data

        original_shape = input.shape
        input = input.view(-1, cls.block_size)
        input_amax = input.abs().max(dim=-1, keepdim=True).values
        descale = input_amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=descale.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

        input = (input / torch.exp2(e8m0_scale)).view(original_shape)
        input_q = cast_fp4(input)
        input_q = fuse_uint4_to_uint8(input_q)
        e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
        return input_q, e8m0_scale

    @classmethod
    def dequantize(cls, quantized_data, dtype: torch.dtype, scale):
        """Dequantize MXFP4 packed tensor back to float."""

        def unfuse_uint8_to_uint4(x):
            left_side = x & 0x0F
            right_side = (x >> 4) & 0x0F
            shape = list(x.shape)
            shape[-1] = shape[-1] * 2
            result = torch.zeros(shape, dtype=torch.uint8, device=x.device)
            result[..., 0::2] = left_side
            result[..., 1::2] = right_side
            return result

        e8m0_scale = scale
        x_unfused = unfuse_uint8_to_uint4(quantized_data)
        sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to(torch.float32)
        magnitude = (x_unfused & 0b0111).to(torch.long)
        values = torch.tensor(cls.E2M1_values, device=quantized_data.device)
        original_shape = magnitude.shape
        x_float = values[magnitude.reshape(-1)].reshape(original_shape)
        x_float = sign.float() * x_float
        x_float = x_float.reshape(-1, cls.block_size)
        scale_factor = torch.exp2(e8m0_scale.float() - 127)
        scale_factor = scale_factor.reshape(-1, 1)
        x_float = x_float * scale_factor
        return x_float.reshape(original_shape).to(dtype)


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _swiglu(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """SwigLU activation used in GPT-OSS.
    Input is interleaved: [gate_0, up_0, gate_1, up_1, ...] in last dim.
    Formula: min(gate, limit) * sigmoid(alpha * min(gate, limit))
            * (clamp(up, -limit, limit) + 1)
    """
    gate = x[..., 0::2]  # even indices
    up = x[..., 1::2]  # odd indices
    gate_clamped = torch.clamp(gate, max=limit)
    up_clamped = torch.clamp(up, min=-limit, max=limit)
    return gate_clamped * torch.sigmoid(alpha * gate_clamped) * (up_clamped + 1)


def ref_mxfp4_fused_moe(
    a: torch.Tensor,
    w1_dq: torch.Tensor,
    w2_dq: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Reference MXFP4 fused MoE with SiLU activation."""
    B, D = a.shape
    a_f = a.float()
    out = torch.zeros(B * topk, w2_dq.shape[1], dtype=torch.float32)
    topk_ids_flat = topk_ids.view(-1)

    for i in range(w1_dq.shape[0]):
        mask = topk_ids_flat == i
        if mask.sum() == 0:
            continue
        # expand for topk
        token_indices = torch.where(mask)[0]
        source_indices = token_indices // topk
        ic0 = torch.matmul(a_f[source_indices], w1_dq[i].float().T)
        ic1 = _silu_and_mul(ic0)
        out[mask] = torch.matmul(ic1, w2_dq[i].float().T)

    return (out.view(B, topk, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def ref_mxfp4_fused_moe_gptoss(
    a: torch.Tensor,
    w1_dq: torch.Tensor,
    w2_dq: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """Reference MXFP4 fused MoE with SwigLU+bias (GPT-OSS style)."""
    B, D = a.shape
    topk = topk_ids.shape[1]
    a_f = a.float()
    E = w1_dq.shape[0]
    out = torch.zeros(B * topk, w2_dq.shape[1], dtype=torch.float32)
    topk_ids_flat = topk_ids.view(-1)

    for i in range(E):
        mask = topk_ids_flat == i
        if mask.sum() == 0:
            continue
        token_indices = torch.where(mask)[0]
        source_indices = token_indices // topk
        ic0 = torch.matmul(a_f[source_indices], w1_dq[i].float().T)
        ic0 = ic0 + w1_bias[i].float()
        ic1 = _swiglu(ic0, alpha, limit)
        ic2 = torch.matmul(ic1, w2_dq[i].float().T)
        ic2 = ic2 + w2_bias[i].float()
        out[mask] = ic2

    return (out.view(B, topk, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def _prepack_mxfp4_experts(
    w: torch.Tensor, w_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """VNNI-prepack MXFP4 weights and repack scales."""
    packed_w = torch.stack(
        [torch.ops._C.convert_weight_packed(w[e]) for e in range(w.shape[0])]
    )
    packed_s = torch.ops._C.convert_scale_packed(w_scale)
    return packed_w, packed_s


NUM_TOKENS = [1, 2, 32, 121]
# (intermediate_size N, hidden_size K, num_experts E, topk)
MOE_CONFIGS = [
    (128, 128, 4, 2),
    (256, 256, 8, 4),
    (352, 256, 8, 4),
    (512, 320, 8, 4),
]
SEEDS = [0]


@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", MOE_CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_mxfp4_cpu_fused_moe(M, N, K, E, topk, seed):
    """Test fused_experts_mxfp4_cpu against dequantized torch reference."""
    set_random_seed(seed)
    dtype = torch.bfloat16

    a = torch.randn(M, K, dtype=dtype) / 10

    # Generate and quantize weights
    w1_bf16 = torch.randn(E, 2 * N, K, dtype=dtype) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

    # Routing
    score = torch.randn(M, E, dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    # Reference
    ref_out = ref_mxfp4_fused_moe(a, w1dq, w2dq, topk_weight, topk_ids, topk)

    # Pack weights for kernel
    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)

    # Kernel
    out = ops.fused_experts_cpu(
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        False,  # inplace
        ops.CPUQuantMethod.MXFP4,
        pw1s,  # w1_scale
        pw2s,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
    )

    torch.testing.assert_close(
        ref_out.bfloat16(), out, atol=MXFP4_ATOL, rtol=MXFP4_RTOL
    )


@pytest.mark.parametrize("M", [1, 32])
@pytest.mark.parametrize("N,K,E,topk", [(128, 128, 4, 2), (64, 64, 4, 2)])
@pytest.mark.parametrize("seed", SEEDS)
def test_mxfp4_cpu_fused_moe_bias_swiglu(M, N, K, E, topk, seed):
    """Test fused_experts_mxfp4_cpu with bias and SwigLU activation (GPT-OSS)."""
    set_random_seed(seed)
    dtype = torch.bfloat16
    alpha = 1.702
    limit = 7.0

    a = torch.randn(M, K, dtype=dtype) / 10

    # Generate and quantize weights
    w1_bf16 = torch.randn(E, 2 * N, K, dtype=dtype) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)
    w1_b = torch.randn(E, 2 * N, dtype=torch.float32) / 10

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)
    w2_b = torch.randn(E, K, dtype=torch.float32) / 10

    # Routing
    score = torch.randn(M, E, dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    # Reference
    ref_out = ref_mxfp4_fused_moe_gptoss(
        a, w1dq, w2dq, w1_b, w2_b, topk_weight, topk_ids, alpha, limit
    )

    # Pack weights for kernel
    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)

    # Kernel
    out = ops.fused_experts_cpu(
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        False,  # inplace
        ops.CPUQuantMethod.MXFP4,
        pw1s,  # w1_scale
        pw2s,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
        w1_bias=w1_b,
        w2_bias=w2_b,
        alpha=alpha,
        limit=limit,
    )

    torch.testing.assert_close(
        ref_out.bfloat16(), out, atol=MXFP4_ATOL, rtol=MXFP4_RTOL
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
