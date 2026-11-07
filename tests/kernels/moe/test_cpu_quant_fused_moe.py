# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU quantized fused MoE kernels."""

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

from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (  # noqa: E402
    prepare_fp8_w8a8_moe_layer_for_cpu,
)


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]


def _gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d]) * x[..., d:]


def _prepack_experts(w: torch.Tensor) -> torch.Tensor:
    """VNNI-prepack expert weights via ``convert_weight_packed``."""
    return torch.ops._C.convert_weight_packed(w)


def _deterministic_expert_routes(
    block_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create top-1 routes with exact per-expert block occupancies."""
    topk_ids = torch.cat(
        [
            torch.full((size,), expert, dtype=torch.int32)
            for expert, size in enumerate(block_sizes)
        ]
    )
    topk_ids = topk_ids.view(-1, 1)
    topk_weights = torch.ones(topk_ids.shape, dtype=torch.float32)
    return topk_weights, topk_ids


# ===========================================================================
# FP8 W8A16 MoE
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
    act_fn=_silu_and_mul,
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
            ic1 = act_fn(ic0)
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

    # Test writing into a fresh output buffer against reference
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )
    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)

    # Test writing into the input tensor itself (aliased in-place) produces
    # an identical result
    a_inplace = a.clone()
    ops.fused_experts_cpu(
        a_inplace,
        a_inplace,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )
    torch.testing.assert_close(a_inplace, out, atol=0, rtol=0)


@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("N,K,E,topk", [(256, 512, 8, 2), (768, 2048, 8, 4)])
def test_w8a16_block_fp8_cpu_fused_moe_gelu_tanh(M, N, K, E, topk):
    """Test fused_experts_cpu FP8 W8A16 with GELU-tanh activation."""
    set_random_seed(0)

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
        act_fn=_gelu_tanh_and_mul,
    )

    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)

    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
        act_method=ops.CPUActMethod.GELU_TANH_AND_MUL,
    )
    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


def test_w8a16_block_fp8_cpu_fused_moe_small_expert_blocks():
    """Test FP8 BRGEMM at exact and multi-block expert boundaries."""
    set_random_seed(0)
    block_sizes = (4, 5, 33)
    N, K, E = 128, 128, len(block_sizes)
    M = sum(block_sizes)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    w1, w2, w1_s, w2_s = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)
    topk_weight, topk_ids = _deterministic_expert_routes(block_sizes)

    ref_out = ref_w8a16_block_fp8_moe(
        a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, BLOCK_SIZE
    )
    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


# ===========================================================================
# MXFP4 W4A16 MoE
# ===========================================================================


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


def _swiglu(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """SwigLU activation used in GPT-OSS.
    Input is interleaved: [gate_0, up_0, gate_1, up_1, ...] in last dim.
    """
    gate = x[..., 0::2]
    up = x[..., 1::2]
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
    act_fn=_silu_and_mul,
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
        token_indices = torch.where(mask)[0]
        source_indices = token_indices // topk
        ic0 = torch.matmul(a_f[source_indices], w1_dq[i].float().T)
        ic1 = act_fn(ic0)
        out[mask] = torch.matmul(ic1, w2_dq[i].float().T)

    return (out.view(B, topk, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def ref_mxfp4_fused_moe_gptoss(
    a: torch.Tensor,
    w1_dq: torch.Tensor,
    w2_dq: torch.Tensor,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
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
        if w1_bias is not None:
            ic0 = ic0 + w1_bias[i].float()
        ic1 = _swiglu(ic0, alpha, limit)
        ic2 = torch.matmul(ic1, w2_dq[i].float().T)
        if w2_bias is not None:
            ic2 = ic2 + w2_bias[i].float()
        out[mask] = ic2

    return (out.view(B, topk, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def _prepack_mxfp4_experts(
    w: torch.Tensor, w_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """VNNI-prepack MXFP4 weights and repack scales."""
    packed_w = torch.ops._C.convert_weight_packed(w)
    packed_s = torch.ops._C.convert_scale_packed(w_scale)
    return packed_w, packed_s


def _quantize_per_token_reference(
    x: torch.Tensor,
    dtype: torch.dtype,
    *,
    round_through_fp16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows to FP8 with dynamic per-row FP32 scales."""
    max_value = torch.finfo(dtype).max
    x_float = x.float()
    scales = (x_float.abs().amax(dim=-1) / max_value).clamp_min(
        torch.finfo(torch.float32).eps
    )
    quantized = (x_float * scales.reciprocal().unsqueeze(-1)).clamp(
        min=-max_value, max=max_value
    )
    if round_through_fp16:
        quantized = quantized.to(torch.float16)
    return quantized.to(dtype), scales


def _ref_cpu_fp8_activation_moe(
    hidden_states: torch.Tensor,
    w1_dequantized: torch.Tensor,
    w2_dequantized: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    alpha: float | None = None,
    limit: float | None = None,
    *,
    input_quantization: tuple[torch.Tensor, torch.Tensor],
    mxfp4: bool = False,
    act_fn=_silu_and_mul,
) -> torch.Tensor:
    """Reference FP8 MoE with FP32 intermediates and BF16 MXFP4 tiny operands."""
    batch_size, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    route_count = batch_size * topk
    route_ids = topk_ids.reshape(-1)
    route_sources = torch.arange(batch_size).repeat_interleave(topk)
    input_q, input_scales = input_quantization

    w1_float = w1_dequantized.float()
    w2_float = w2_dequantized.float()

    routed_out = torch.empty(
        route_count, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )
    for expert in range(w1_dequantized.shape[0]):
        route_indices = torch.where(route_ids == expert)[0]
        if route_indices.numel() == 0:
            continue

        source_indices = route_sources[route_indices]
        gate_up = input_q[source_indices].float() @ w1_float[expert].T
        gate_up *= input_scales[source_indices].unsqueeze(-1)
        tiny = mxfp4 and route_indices.numel() <= 4
        if tiny:
            tiny_input = (
                input_q[source_indices].float()
                * input_scales[source_indices].unsqueeze(-1)
            ).to(torch.bfloat16)
            gate_up = tiny_input.float() @ w1_float[expert].T
        if w1_bias is not None:
            gate_up += w1_bias[expert]
        if not mxfp4:
            gate_up = gate_up.to(torch.bfloat16)

        if alpha is None:
            activated = act_fn(gate_up.float())
        else:
            assert limit is not None
            activated = _swiglu(gate_up.float(), alpha, limit)
        if not mxfp4:
            activated = activated.to(torch.bfloat16)

        activated_q, activated_scales = _quantize_per_token_reference(
            activated, torch.float8_e4m3fn
        )
        down = activated_q.float() @ w2_float[expert].T
        down *= activated_scales.unsqueeze(-1)
        if tiny:
            down = activated.to(torch.bfloat16).float() @ w2_float[expert].T
        if w2_bias is not None:
            down += w2_bias[expert]

        routed_out[route_indices] = (
            down.to(torch.bfloat16).float()
            * topk_weights.reshape(-1)[route_indices].unsqueeze(-1)
        ).to(torch.bfloat16)

    return (
        routed_out.view(batch_size, topk, hidden_size)
        .float()
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


requires_cpu_native_e4m3 = pytest.mark.skipif(
    not ops.cpu_native_fp8_supported(torch.float8_e4m3fn),
    reason="requires native E4M3 BRGEMM support",
)


def _run_mxfp4_w4a8(
    a: torch.Tensor,
    pw1: torch.Tensor,
    pw2: torch.Tensor,
    pw1s: torch.Tensor,
    pw2s: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    alpha: float | None = None,
    limit: float | None = None,
    w1_block_metadata: torch.Tensor | None = None,
    w2_block_metadata: torch.Tensor | None = None,
    w1_channel_metadata: torch.Tensor | None = None,
    w2_channel_metadata: torch.Tensor | None = None,
    act_method: ops.CPUActMethod | None = None,
) -> torch.Tensor:
    """Quantize activations to E4M3 and run MXFP4 W4A8."""
    if act_method is None:
        act_method = (
            ops.CPUActMethod.SWIGLU
            if alpha is not None
            else ops.CPUActMethod.SILU_AND_MUL
        )
    hidden_states, a_scale = torch.ops._C.quantize_fp8e4m3_vec(a, True, None)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        hidden_states,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4_W4A8,
        pw1s,
        pw2s,
        None,
        None,
        None,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        alpha=alpha,
        limit=limit,
        a1_scale=a_scale,
        act_method=act_method,
        w1_block_metadata=w1_block_metadata,
        w2_block_metadata=w2_block_metadata,
        w1_channel_metadata=w1_channel_metadata,
        w2_channel_metadata=w2_channel_metadata,
    )
    return out


MXFP4_NUM_TOKENS = [1, 2, 32, 121]
MXFP4_MOE_CONFIGS = [
    (128, 128, 4, 2),
    (256, 256, 8, 4),
    (352, 256, 8, 4),
    (512, 320, 8, 4),
]


@pytest.mark.parametrize("M", MXFP4_NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk", MXFP4_MOE_CONFIGS)
@pytest.mark.parametrize("seed", [0])
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
    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,  # w1_scale
        pw2s,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M", [1, 32])
@pytest.mark.parametrize("N,K,E,topk", [(128, 128, 4, 2), (256, 256, 8, 4)])
def test_mxfp4_cpu_fused_moe_gelu_tanh(M, N, K, E, topk):
    """Test fused_experts_mxfp4_cpu with GELU-tanh activation."""
    set_random_seed(0)
    dtype = torch.bfloat16

    a = torch.randn(M, K, dtype=dtype) / 10

    w1_bf16 = torch.randn(E, 2 * N, K, dtype=dtype) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

    score = torch.randn(M, E, dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_mxfp4_fused_moe(
        a, w1dq, w2dq, topk_weight, topk_ids, topk, act_fn=_gelu_tanh_and_mul
    )

    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)

    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,  # w1_scale
        pw2s,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
        act_method=ops.CPUActMethod.GELU_TANH_AND_MUL,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "quant_method,weight_scale_byte,activation_name",
    [
        pytest.param(ops.CPUQuantMethod.MXFP4, None, "silu", id="w4a16_silu"),
        pytest.param(
            ops.CPUQuantMethod.MXFP4_W4A8,
            None,
            "silu",
            marks=requires_cpu_native_e4m3,
            id="w4a8_silu",
        ),
        pytest.param(
            ops.CPUQuantMethod.MXFP4_W4A8,
            "mixed",
            "silu",
            marks=requires_cpu_native_e4m3,
            id="w4a8_mixed_scales",
        ),
        pytest.param(
            ops.CPUQuantMethod.MXFP4_W4A8,
            None,
            "gelu",
            marks=requires_cpu_native_e4m3,
            id="w4a8_gelu",
        ),
        pytest.param(
            ops.CPUQuantMethod.MXFP4_W4A8,
            None,
            "gelu_tanh",
            marks=requires_cpu_native_e4m3,
            id="w4a8_gelu_tanh",
        ),
    ],
)
def test_mxfp4_cpu_fused_moe_small_expert_blocks(
    quant_method, weight_scale_byte, activation_name
):
    """Check row scales, K-block rescaling, tiny inputs, and expert cache changes."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    activation, moe_activation, act_fn = {
        "silu": (
            ops.CPUActMethod.SILU_AND_MUL,
            MoEActivation.SILU,
            _silu_and_mul,
        ),
        "gelu": (
            ops.CPUActMethod.GELU_AND_MUL,
            MoEActivation.GELU,
            _gelu_and_mul,
        ),
        "gelu_tanh": (
            ops.CPUActMethod.GELU_TANH_AND_MUL,
            MoEActivation.GELU_TANH,
            _gelu_tanh_and_mul,
        ),
    }[activation_name]
    set_random_seed(0)
    block_sizes = (1, 4, 5, 33)
    N, K, E = 128, 256, len(block_sizes)
    M = sum(block_sizes)
    dtype = torch.bfloat16

    a = torch.randn(M, K, dtype=dtype) / 10
    a[1].zero_()
    a[2] *= 0.01
    a[3] *= 4

    w1_bf16 = torch.randn(E, 2 * N, K, dtype=dtype) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    if weight_scale_byte == "mixed":
        w1s[..., :4] = 140
        w1s[..., 4:] = 126
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    if weight_scale_byte == "mixed":
        w2s[..., 0] = 140
        w2s[..., 1:] = 126
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)
    topk_weight, topk_ids = _deterministic_expert_routes(block_sizes)
    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
    if quant_method == ops.CPUQuantMethod.MXFP4:
        ref_out = ref_mxfp4_fused_moe(
            a, w1dq, w2dq, topk_weight, topk_ids, 1, act_fn=act_fn
        )
        out = torch.empty_like(a)
        ops.fused_experts_cpu(
            out,
            a,
            pw1,
            pw2,
            topk_weight,
            topk_ids,
            quant_method,
            pw1s,
            pw2s,
            None,
            None,
            None,
            act_method=activation,
        )
    else:
        from tests.kernels.moe.utils import make_dummy_moe_config
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            CPUExpertsMxfp4W4A8,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTokenSym,
            kMxfp4Static,
        )

        moe_config = make_dummy_moe_config(
            num_experts=E,
            experts_per_token=topk_ids.shape[1],
            hidden_dim=K,
            intermediate_size=N,
            max_num_tokens=M,
            activation=moe_activation,
        )
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

        moe_config.routing_method = RoutingMethodType.Default
        supported, reason = CPUExpertsMxfp4W4A8.is_supported_config(
            CPUExpertsMxfp4W4A8,
            moe_config,
            kMxfp4Static,
            kFp8DynamicTokenSym,
            CPUExpertsMxfp4W4A8.activation_format(),
        )
        assert supported, (
            f"expected MXFP4 W4A8 experts to support {activation_name}, got {reason}"
        )
        ref_out = _ref_cpu_fp8_activation_moe(
            a,
            w1dq,
            w2dq,
            topk_weight,
            topk_ids,
            input_quantization=_quantize_e4m3_per_token(a),
            mxfp4=True,
            act_fn=act_fn,
        )
        w1_block_metadata = torch.ops._C.mxfp4_moe_scale_metadata(pw1s)
        w2_block_metadata = torch.ops._C.mxfp4_moe_scale_metadata(pw2s)
        assert torch.all(w1_block_metadata[..., 2] == 1)
        assert torch.all(
            w2_block_metadata[..., 2] == (0 if weight_scale_byte == "mixed" else 1)
        )
        out = _run_mxfp4_w4a8(
            a,
            pw1,
            pw2,
            pw1s,
            pw2s,
            topk_weight,
            topk_ids,
            w1_block_metadata=w1_block_metadata,
            w2_block_metadata=w2_block_metadata,
            act_method=activation,
        )

        if weight_scale_byte == "mixed":
            runtime_metadata_out = _run_mxfp4_w4a8(
                a,
                pw1,
                pw2,
                pw1s,
                pw2s,
                topk_weight,
                topk_ids,
                act_method=activation,
            )
            assert torch.equal(out, runtime_metadata_out)
    for ref_block, out_block in zip(ref_out.split(block_sizes), out.split(block_sizes)):
        atol = (
            1e-2
            if quant_method == ops.CPUQuantMethod.MXFP4
            else 1e-2 * ref_block.abs().max().item()
        )
        torch.testing.assert_close(ref_block, out_block, atol=atol, rtol=1e-2)
    assert torch.equal(out[1], torch.zeros_like(out[1]))


@requires_cpu_native_e4m3
def test_mxfp4_tiny_experts_keep_unquantized_swiglu():
    """Tiny experts round FP32 SwiGLU to BF16; large experts quantize to E4M3."""
    n = k = 32
    a = torch.zeros(9, k, dtype=torch.bfloat16)
    topk_weights, topk_ids = _deterministic_expert_routes((4, 5))
    q1, s1 = MXFP4QuantizeUtil.quantize(torch.zeros(2, 2 * n, k, dtype=a.dtype))
    q2, s2 = MXFP4QuantizeUtil.quantize(
        torch.eye(n, dtype=a.dtype).expand(2, n, n).clone()
    )
    pw1, ps1 = _prepack_mxfp4_experts(q1, s1.reshape(2, 2 * n, k // 32))
    pw2, ps2 = _prepack_mxfp4_experts(q2, s2.reshape(2, k, n // 32))
    bias = torch.zeros(2, 2 * n, dtype=torch.float32)
    bias[:, :4] = torch.tensor([4.305, 2.729, 0.8, 1.49])
    actual = _run_mxfp4_w4a8(
        a,
        pw1,
        pw2,
        ps1,
        ps2,
        topk_weights,
        topk_ids,
        w1_bias=bias,
        alpha=1.702,
        limit=7.0,
    )
    expected = torch.zeros_like(a)
    expected[:4, :2] = torch.tensor([16.0, 1.5859375])
    expected[4:, :2] = torch.tensor([16.0, 1.578125])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_cpu_native_e4m3
def test_mxfp4_tiny_and_large_experts_propagate_fp8_nan():
    """Direct E4M3 inputs with finite scales retain both NaN encodings."""
    n = k = 32
    a = torch.zeros(9, k, dtype=torch.float8_e4m3fn)
    a.view(torch.uint8)[0, 0] = 127
    a.view(torch.uint8)[4, 0] = 255
    topk_weights, topk_ids = _deterministic_expert_routes((4, 5))
    w1 = torch.zeros(2, 2 * n, k // 2, dtype=torch.uint8)
    w2 = torch.zeros(2, k, n // 2, dtype=torch.uint8)
    s1 = torch.full((2, 2 * n, k // 32), 127, dtype=torch.uint8)
    s2 = torch.full((2, k, n // 32), 127, dtype=torch.uint8)
    pw1, ps1 = _prepack_mxfp4_experts(w1, s1)
    pw2, ps2 = _prepack_mxfp4_experts(w2, s2)
    out = torch.empty(9, k, dtype=torch.bfloat16)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weights,
        topk_ids,
        ops.CPUQuantMethod.MXFP4_W4A8,
        ps1,
        ps2,
        None,
        None,
        None,
        a1_scale=torch.ones(9),
    )
    assert out[[0, 4]].isnan().all()
    assert torch.equal(out[[1, 2, 3, 5, 6, 7, 8]], torch.zeros(7, k))


@requires_cpu_native_e4m3
@pytest.mark.parametrize("use_channel_metadata", [False, True])
def test_mxfp4_w4a8_bias_fusion_preserves_fp8_rounding(use_channel_metadata):
    """Keep scaled bias fused before SwiGLU crosses an FP8 rounding midpoint."""
    m, n, k = 5, 32, 32
    a = torch.ones(m, k, dtype=torch.bfloat16)
    a[:, 0] = 2
    w1q = torch.zeros(1, 2 * n, k // 2, dtype=torch.uint8)
    w1q[:, 0] = 0x11
    w1s = torch.full((1, 2 * n, k // 32), 133, dtype=torch.uint8)
    w2q, w2s = MXFP4QuantizeUtil.quantize(torch.eye(n, dtype=a.dtype).unsqueeze(0))
    pw1, ps1 = _prepack_mxfp4_experts(w1q, w1s)
    pw2, ps2 = _prepack_mxfp4_experts(w2q, w2s.reshape(1, k, n // 32))
    bias = torch.zeros(1, 2 * n, dtype=torch.float32)
    bias[0, 0] = -1054.9375
    bias[0, 2] = 14
    actual = _run_mxfp4_w4a8(
        a,
        pw1,
        pw2,
        ps1,
        ps2,
        torch.ones(m, 1),
        torch.zeros(m, 1, dtype=torch.int32),
        w1_bias=bias,
        alpha=0.0,
        limit=64.0,
        w1_block_metadata=torch.ops._C.mxfp4_moe_scale_metadata(ps1),
        w2_block_metadata=torch.ops._C.mxfp4_moe_scale_metadata(ps2),
        w1_channel_metadata=(
            torch.ops._C.mxfp4_moe_channel_metadata(ps1)
            if use_channel_metadata
            else None
        ),
        w2_channel_metadata=(
            torch.ops._C.mxfp4_moe_channel_metadata(ps2)
            if use_channel_metadata
            else None
        ),
    )
    expected = torch.zeros_like(a)
    # Separate multiply/add rounds the first output to 0.5.
    expected[:, 0] = 0.5625
    expected[:, 1] = 7
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_cpu_native_e4m3
@pytest.mark.parametrize(
    "source_scale_byte,use_channel_metadata",
    [
        pytest.param(129, False, id="block_spread_11"),
        pytest.param(128, False, id="block_spread_12"),
        pytest.param(126, False, id="block_spread_14"),
        pytest.param(125, False, id="block_spread_15"),
        pytest.param(129, True, id="channel_spread_11"),
        pytest.param(128, True, id="channel_spread_12"),
    ],
)
def test_mxfp4_w4a8_e4m3_decodes_codes_across_partial_stripes(
    source_scale_byte, use_channel_metadata
):
    """Check signed nibbles, channel scales, E4M3 rounding, and partial blocks."""
    M, N, K, E = 5, 96, 160, 1
    a = torch.zeros(M, K, dtype=torch.bfloat16)
    w1q = torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8)
    w1s = torch.full((E, 2 * N, K // 32), 127, dtype=torch.uint8)
    w2q = torch.zeros(E, K, N // 2, dtype=torch.uint8)
    codes = torch.arange(K, dtype=torch.uint8) % 16
    high_codes = torch.zeros(K, dtype=torch.int64)
    high_codes[128:] = (codes[128:].long() + 5) % 16
    w2q[0, :, 0] = (codes.to(torch.int64) | (high_codes << 4)).to(torch.uint8)
    w2s = torch.full((E, K, N // 32), source_scale_byte, dtype=torch.uint8)
    # A single panel in the partial column group sets the entire first stripe's
    # normalization, including panels owned by other threads.
    w2s[:, 64:96, -1] = 140
    if use_channel_metadata:
        w2s[:, 128:, :] = torch.arange(110, 142, dtype=torch.uint8).unsqueeze(-1)
    w1_bias = torch.zeros(E, 2 * N)
    w1_bias[:, 0::2] = 7
    w2_bias = torch.zeros(E, K)
    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
    w1_block_metadata = torch.ops._C.mxfp4_moe_scale_metadata(pw1s)
    w2_block_metadata = torch.ops._C.mxfp4_moe_scale_metadata(pw2s)
    w2_channel_metadata = (
        torch.ops._C.mxfp4_moe_channel_metadata(pw2s) if use_channel_metadata else None
    )
    out = _run_mxfp4_w4a8(
        a,
        pw1,
        pw2,
        pw1s,
        pw2s,
        torch.ones(M, 1),
        torch.zeros(M, 1, dtype=torch.int32),
        w1_bias,
        w2_bias,
        1.702,
        7.0,
        w1_block_metadata=w1_block_metadata,
        w2_block_metadata=w2_block_metadata,
        w2_channel_metadata=w2_channel_metadata,
    )
    fp4_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    block_max = torch.cat(
        (w2s[0, :128].max().expand(128), w2s[0, 128:].max().expand(K - 128))
    ).to(torch.int64)
    # Eligible channels use their own maximum; the mixed panel falls back to
    # the shared block maximum when any channel exceeds the spread limit.
    decode_max = (
        w2s[0].max(dim=-1).values.to(torch.int64) if use_channel_metadata else block_max
    )
    normalization = torch.exp2((w2s[0, :, 0].long() - decode_max + 6).float())
    decoded_low = (fp4_values[codes.long()] * normalization).to(torch.float8_e4m3fn)
    decoded_high = (fp4_values[high_codes] * normalization).to(torch.float8_e4m3fn)
    decoded = (decoded_low.float() + decoded_high.float()) * torch.exp2(
        (decode_max - 133).float()
    )
    # Interleaved SwiGLU rounds to BF16 7, whose E4M3 row quantization is exact.
    expected = (7 * decoded).to(torch.bfloat16)
    torch.testing.assert_close(out, expected.expand(M, K), atol=0, rtol=0)
    if source_scale_byte == 128 and not use_channel_metadata:
        runtime_metadata_out = _run_mxfp4_w4a8(
            a,
            pw1,
            pw2,
            pw1s,
            pw2s,
            torch.ones(M, 1),
            torch.zeros(M, 1, dtype=torch.int32),
            w1_bias,
            w2_bias,
            1.702,
            7.0,
        )
        assert torch.equal(out, runtime_metadata_out)


@requires_cpu_native_e4m3
def test_mxfp4_w4a8_padded_tp_shards():
    """Zero-padded intermediate shards preserve both projections and TP reduction."""
    set_random_seed(0)
    N, K, E, tp = 96, 128, 2, 2
    topk_weights, topk_ids = _deterministic_expert_routes((4, 33))
    a = torch.randn(topk_ids.shape[0], K, dtype=torch.bfloat16) / 10
    w1 = torch.randn(E, 2 * N, K, dtype=torch.bfloat16) / 10
    w2 = torch.randn(E, K, N, dtype=torch.bfloat16) / 10
    b1 = torch.randn(E, 2 * N) / 10
    b2 = torch.randn(E, K) / 10
    shard = N // tp
    padding = (32 - shard % 32) % 32
    actual = torch.zeros_like(a, dtype=torch.float32)
    expected = torch.zeros_like(actual)
    input_quantization = _quantize_e4m3_per_token(a)
    for rank in range(tp):
        start, end = rank * shard, (rank + 1) * shard
        shard_w1 = torch.cat(
            (
                F.pad(w1[:, start:end], (0, 0, 0, padding)),
                F.pad(w1[:, N + start : N + end], (0, 0, 0, padding)),
            ),
            dim=1,
        )
        shard_b1 = torch.cat(
            (
                F.pad(b1[:, start:end], (0, padding)),
                F.pad(b1[:, N + start : N + end], (0, padding)),
            ),
            dim=1,
        )
        shard_w2 = F.pad(w2[:, :, start:end], (0, padding))
        q1, s1 = MXFP4QuantizeUtil.quantize(shard_w1)
        q2, s2 = MXFP4QuantizeUtil.quantize(shard_w2)
        s1 = s1.reshape(E, 2 * (shard + padding), K // 32)
        s2 = s2.reshape(E, K, (shard + padding) // 32)
        pw1, ps1 = _prepack_mxfp4_experts(q1, s1)
        pw2, ps2 = _prepack_mxfp4_experts(q2, s2)
        shard_b2 = b2 if rank == 0 else None
        actual += _run_mxfp4_w4a8(
            a,
            pw1,
            pw2,
            ps1,
            ps2,
            topk_weights,
            topk_ids,
            shard_b1,
            shard_b2,
        ).float()
        expected += _ref_cpu_fp8_activation_moe(
            a,
            MXFP4QuantizeUtil.dequantize(q1, a.dtype, s1),
            MXFP4QuantizeUtil.dequantize(q2, a.dtype, s2),
            topk_weights,
            topk_ids,
            shard_b1,
            shard_b2,
            input_quantization=input_quantization,
            mxfp4=True,
        ).float()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@requires_cpu_native_e4m3
@pytest.mark.parametrize(
    "metadata_op",
    [
        torch.ops._C.mxfp4_moe_scale_metadata,
        torch.ops._C.mxfp4_moe_channel_metadata,
    ],
    ids=["block", "channel"],
)
def test_mxfp4_w4a8_metadata_rejects_reserved_e8m0_nan_scale(metadata_op):
    """Reject E8M0's reserved NaN code during metadata preparation."""
    packed_scales = torch.full((1, 32, 1), 255, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="reserved E8M0 NaN"):
        metadata_op(packed_scales)


@requires_cpu_native_e4m3
@pytest.mark.parametrize("M", [1, 5], ids=["tiny", "brgemm"])
@pytest.mark.parametrize("projection", ["w1", "w2"])
def test_mxfp4_w4a8_rejects_reserved_e8m0_nan_scale(M, projection):
    """Reject E8M0's reserved NaN code before decoding compressed weights."""
    N, K, E = 32, 32, 1
    a = torch.zeros(M, K, dtype=torch.bfloat16)
    w1q = torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8)
    w1s = torch.full((E, 2 * N, K // 32), 127, dtype=torch.uint8)
    w2q = torch.zeros(E, K, N // 2, dtype=torch.uint8)
    w2s = torch.full((E, K, N // 32), 127, dtype=torch.uint8)
    (w1s if projection == "w1" else w2s).fill_(255)
    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)

    with pytest.raises(RuntimeError, match="reserved E8M0 NaN"):
        _run_mxfp4_w4a8(
            a,
            pw1,
            pw2,
            pw1s,
            pw2s,
            torch.ones(M, 1),
            torch.zeros(M, 1, dtype=torch.int32),
        )


# Both E2M1 zero codes: 0b0000 (+0.0) and 0b1000 (-0.0), in both nibbles.
MXFP4_ZERO_BYTES = [0x00, 0x88, 0x08, 0x80]
# 0 and 255 are the ends of the E8M0 range; 127 is the identity scale.
MXFP4_E8M0_VALUES = [0, 1, 127, 200, 254, 255]


@pytest.mark.parametrize("zero_byte", MXFP4_ZERO_BYTES)
@pytest.mark.parametrize("e8m0", MXFP4_E8M0_VALUES)
def test_mxfp4_cpu_zero_codes_stay_zero(zero_byte, e8m0):
    """A zero E2M1 code stays zero for every E8M0 exponent.

    The unpack applies the block scale as an integer add on the bf16 exponent
    field, which is exact for every value in the E2M1 codebook except the two
    zeros: 0x0000 and 0x8000 have no exponent to shift, so adding to them
    produces a small finite number instead of zero. Both are special-cased, and
    this is the invariant that special case exists for.

    Worth pinning separately from ``test_mxfp4_cpu_fused_moe``: there the zero
    codes are a small fraction of random weights and a broken special case
    would stay inside the 1e-2 tolerance for the low exponents. Here every
    weight is a zero, so the output is exactly zero or it is not.
    """
    N, K, E, M = 64, 64, 2, 4
    dtype = torch.bfloat16
    set_random_seed(0)

    a = torch.randn(M, K, dtype=dtype)
    w1q = torch.full((E, 2 * N, K // 2), zero_byte, dtype=torch.uint8)
    w1s = torch.full((E, 2 * N, K // 32), e8m0, dtype=torch.uint8)
    # w2 is ordinary: the zeros have to survive the first GEMM and the
    # activation, and a nonzero w2 is what would expose it if they did not.
    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)

    topk_weight = torch.ones((M, 1), dtype=torch.float32)
    topk_ids = torch.zeros((M, 1), dtype=torch.int32)

    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,
        pw2s,
        None,
        None,
        None,
    )

    # silu(0) * 0 = 0, so the whole layer collapses to exactly zero. Not
    # assert_close: any nonzero output here is a wrong unpack, not rounding.
    assert torch.equal(out, torch.zeros_like(out)), (
        f"zero code 0x{zero_byte:02x} with e8m0={e8m0} produced "
        f"max |out| = {out.abs().max().item()}"
    )


# Narrower than MXFP4_E8M0_VALUES on purpose: this test keeps nonzero weights,
# and 6.0 * 2**(255-127) is not representable in bf16, so the ends of the E8M0
# range would compare inf against inf and prove nothing. The all-zero test above
# is the one that can reach them.
MXFP4_E8M0_FINITE = [107, 127, 137]


@pytest.mark.parametrize("e8m0", MXFP4_E8M0_FINITE)
def test_mxfp4_cpu_zero_codes_mixed_with_nonzero(e8m0):
    """Zeros and nonzeros in the same 32-element scale block.

    The zero check is per lane, not per block: this fails if it is ever
    rewritten as a whole-block branch. Uses a single shared exponent so the
    reference is a plain power of two.
    """
    N, K, E, M = 64, 64, 2, 4
    dtype = torch.bfloat16
    set_random_seed(0)

    a = torch.randn(M, K, dtype=dtype)
    # Alternate a zero byte and a nonzero one along K, so every scale block
    # holds both kinds.
    pattern = torch.tensor([0x88, 0x21], dtype=torch.uint8).repeat(K // 4)
    w1q = pattern.view(1, 1, -1).expand(E, 2 * N, K // 2).contiguous()
    w1s = torch.full((E, 2 * N, K // 32), e8m0, dtype=torch.uint8)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

    w2_bf16 = torch.randn(E, K, N, dtype=dtype) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

    topk_weight = torch.ones((M, 1), dtype=torch.float32)
    topk_ids = torch.zeros((M, 1), dtype=torch.int32)
    ref_out = ref_mxfp4_fused_moe(a, w1dq, w2dq, topk_weight, topk_ids, 1)

    pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
    pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.MXFP4,
        pw1s,
        pw2s,
        None,
        None,
        None,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M", [1, 32])
@pytest.mark.parametrize(
    "use_w1_bias,use_w2_bias", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize(
    "quant_method", [ops.CPUQuantMethod.MXFP4, ops.CPUQuantMethod.FP8_W8A16]
)
def test_cpu_weight_only_fused_moe_bias_swiglu(
    M, use_w1_bias, use_w2_bias, quant_method
):
    """Apply projection biases independently in tiny and batched GPT-OSS experts."""
    set_random_seed(0)
    N, K, E, topk = 128, 128, 4, 2
    dtype = torch.bfloat16
    alpha = 1.702
    limit = 7.0

    a = torch.randn(M, K, dtype=dtype) / 10

    w1_b = torch.randn(E, 2 * N, dtype=torch.float32) / 10 if use_w1_bias else None
    w2_b = torch.randn(E, K, dtype=torch.float32) / 10 if use_w2_bias else None
    if quant_method == ops.CPUQuantMethod.MXFP4:
        w1q, w1s = MXFP4QuantizeUtil.quantize(
            torch.randn(E, 2 * N, K, dtype=dtype) / 10
        )
        w1s = w1s.reshape(E, 2 * N, K // 32)
        w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)
        w2q, w2s = MXFP4QuantizeUtil.quantize(torch.randn(E, K, N, dtype=dtype) / 10)
        w2s = w2s.reshape(E, K, N // 32)
        w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)
        pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
        pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
        block_size = None
    else:
        w1q, w2q, w1s, w2s = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)
        w1dq = _block_dequant_weight(w1q, w1s, BLOCK_SIZE)
        w2dq = _block_dequant_weight(w2q, w2s, BLOCK_SIZE)
        pw1, pw2 = _prepack_experts(w1q), _prepack_experts(w2q)
        pw1s, pw2s = w1s, w2s
        block_size = BLOCK_SIZE

    # Routing
    score = torch.randn(M, E, dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    # Reference
    ref_out = ref_mxfp4_fused_moe_gptoss(
        a, w1dq, w2dq, w1_b, w2_b, topk_weight, topk_ids, alpha, limit
    )

    # Kernel
    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        quant_method,
        pw1s,  # w1_scale
        pw2s,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        block_size,
        w1_bias=w1_b,
        w2_bias=w2_b,
        alpha=alpha,
        limit=limit,
        act_method=ops.CPUActMethod.SWIGLU,
    )

    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


# ===========================================================================
# INT4 W4A16 MoE
# ===========================================================================


def _pack_int4_gptq(w_int4: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values [N, K] → [N, K//8] int32 along K dim (GPTQ format)."""
    N, K = w_int4.shape
    assert K % 8 == 0
    w = w_int4.to(torch.int32)
    w_packed = torch.zeros(N, K // 8, dtype=torch.int32)
    for j in range(8):
        w_packed |= (w[:, j::8] & 0xF) << (j * 4)
    return w_packed


def _pack_int4_awq(w_int4: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values [..., N] → [..., N//8] int32 along last dim (AWQ format)."""
    # AWQ packing bitshifts: indices {0,4,1,5,2,6,3,7} * 4 bits each
    _AWQ_BITSHIFTS = [0, 16, 4, 20, 8, 24, 12, 28]

    N = w_int4.shape[-1]
    assert N % 8 == 0
    w = w_int4.to(torch.int32)
    w_packed = torch.zeros(*w.shape[:-1], N // 8, dtype=torch.int32)
    for j, shift in enumerate(_AWQ_BITSHIFTS):
        w_packed |= (w[..., j::8] & 0xF) << shift
    return w_packed


def _ref_int4_moe(
    a: torch.Tensor,
    w1_int4: torch.Tensor,
    w2_int4: torch.Tensor,
    w1_zeros: torch.Tensor | None,
    w2_zeros: torch.Tensor | None,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int,
    act_fn=_silu_and_mul,
) -> torch.Tensor:
    """Reference INT4 W4A16 group-quantized fused MoE in pure torch."""
    B = a.shape[0]
    topk = topk_ids.size(1)
    K_out = a.shape[1]

    out = torch.zeros(B, topk, K_out, dtype=torch.float32)
    for b in range(B):
        for t in range(topk):
            eid = topk_ids[b, t].item()
            x = a[b : b + 1].float()

            # Dequantize w1: [K, 2*N], groups along K (input dim)
            K_dim = w1_int4.shape[1]
            w1_dq = torch.zeros(K_dim, w1_int4.shape[2], dtype=torch.float32)
            for g in range(w1_s.shape[1]):
                k_start = g * group_size
                k_end = min((g + 1) * group_size, K_dim)
                zp = w1_zeros[eid, g, :].float() if w1_zeros is not None else 8.0
                w1_dq[k_start:k_end, :] = (
                    w1_int4[eid, k_start:k_end, :].float() - zp
                ) * w1_s[eid, g, :].float()

            ic = torch.matmul(x, w1_dq)  # [1, K] @ [K, 2*N] → [1, 2*N]
            ic = act_fn(ic)  # [1, N]

            # Dequantize w2: [N, K], groups along N (input dim)
            N_dim = w2_int4.shape[1]
            w2_dq = torch.zeros(N_dim, w2_int4.shape[2], dtype=torch.float32)
            for g in range(w2_s.shape[1]):
                n_start = g * group_size
                n_end = min((g + 1) * group_size, N_dim)
                zp = w2_zeros[eid, g, :].float() if w2_zeros is not None else 8.0
                w2_dq[n_start:n_end, :] = (
                    w2_int4[eid, n_start:n_end, :].float() - zp
                ) * w2_s[eid, g, :].float()

            oc = torch.matmul(ic, w2_dq)  # [1, N] @ [N, K] → [1, K]
            out[b, t] = oc.squeeze(0)

    return (out * topk_weight.unsqueeze(-1)).sum(dim=1).to(a.dtype)


def _make_int4_moe_weights(E, N, K, group_size, quant_algo):
    """Create INT4 MoE weights in GPTQ or AWQ packed format.

    Canonical layout (input × output):
      w1_int4: [E, K, 2*N]  w2_int4: [E, N, K]

    GPTQ packed (pack transposed weight along input/K dim):
      w1_packed: [E, K//8, 2*N]  w2_packed: [E, N//8, K]
      zeros: actual int4 zero points, same packing as weights

    AWQ packed (pack along output/N dim):
      w1_packed: [E, K, 2*N//8]  w2_packed: [E, N, K//8]
      zeros: actual int4 zero points, same packing as weights

    Returns:
        w1_int4, w2_int4,
        w1_packed, w2_packed,
        w1_zeros, w2_zeros,
        w1_zeros_packed, w2_zeros_packed,
        w1_s, w2_s

    """
    w1_int4 = torch.randint(0, 16, (E, K, 2 * N), dtype=torch.int32)
    w2_int4 = torch.randint(0, 16, (E, N, K), dtype=torch.int32)

    num_groups_w1 = K // group_size
    num_groups_w2 = N // group_size
    w1_s = (
        torch.randn(E, num_groups_w1, 2 * N, dtype=torch.bfloat16) * 0.01
    ).abs() + 0.001
    w2_s = (torch.randn(E, num_groups_w2, K, dtype=torch.bfloat16) * 0.01).abs() + 0.001

    if quant_algo == ops.CPUQuantAlgo.GPTQ:
        # Pack: canonical [E, K, 2*N] → transpose [E, 2*N, K] → GPTQ pack
        # [E, 2*N, K//8] → transpose [E, K//8, 2*N]
        w1_t = w1_int4.transpose(1, 2).contiguous()  # [E, 2*N, K]
        w1_packed = (
            torch.stack([_pack_int4_gptq(w1_t[e]) for e in range(E)])
            .transpose(1, 2)
            .contiguous()
        )  # [E, K//8, 2*N]
        w2_t = w2_int4.transpose(1, 2).contiguous()  # [E, K, N]
        w2_packed = (
            torch.stack([_pack_int4_gptq(w2_t[e]) for e in range(E)])
            .transpose(1, 2)
            .contiguous()
        )  # [E, N//8, K]
        w1_zeros = w2_zeros = None
        w1_zeros_packed = torch.full(
            (E, num_groups_w1, 2 * N // 8), 0x77777777, dtype=torch.int32
        )
        w2_zeros_packed = torch.full(
            (E, num_groups_w2, K // 8), 0x77777777, dtype=torch.int32
        )
    else:  # AWQ
        # Asymmetric: actual zero points, packed along output dim.
        w1_zeros = torch.randint(1, 15, (E, num_groups_w1, 2 * N), dtype=torch.int32)
        w2_zeros = torch.randint(1, 15, (E, num_groups_w2, K), dtype=torch.int32)
        w1_packed = torch.stack(
            [_pack_int4_awq(w1_int4[e]) for e in range(E)]
        )  # [E, K, 2*N//8]
        w2_packed = torch.stack(
            [_pack_int4_awq(w2_int4[e]) for e in range(E)]
        )  # [E, N, K//8]
        w1_zeros_packed = torch.stack(
            [_pack_int4_awq(w1_zeros[e]) for e in range(E)]
        )  # [E, K//gs, 2*N//8]
        w2_zeros_packed = torch.stack(
            [_pack_int4_awq(w2_zeros[e]) for e in range(E)]
        )  # [E, N//gs, K//8]

    return (
        w1_int4,
        w2_int4,
        w1_packed,
        w2_packed,
        w1_zeros,
        w2_zeros,
        w1_zeros_packed,
        w2_zeros_packed,
        w1_s,
        w2_s,
    )


INT4_MOE_CONFIGS = [
    # (N, K, E, topk, group_size)
    (256, 512, 8, 2, 128),
    (512, 256, 8, 2, 128),
    (512, 512, 8, 4, 128),
    (768, 2048, 8, 2, 128),
]


@pytest.mark.parametrize("M", [1, 2, 64, 121])
@pytest.mark.parametrize("N,K,E,topk,group_size", INT4_MOE_CONFIGS)
@pytest.mark.parametrize("quant_algo", [ops.CPUQuantAlgo.GPTQ, ops.CPUQuantAlgo.AWQ])
@pytest.mark.parametrize("seed", [0])
def test_int4_w4a16_cpu_fused_moe(M, N, K, E, topk, group_size, quant_algo, seed):
    """Test fused_experts_cpu INT4 W4A16 for both GPTQ and AWQ quant formats."""
    set_random_seed(seed)

    a = torch.randn(M, K, dtype=torch.bfloat16) / (0.5 * K**0.5)
    (
        w1_int4,
        w2_int4,
        w1_packed,
        w2_packed,
        w1_zeros,
        w2_zeros,
        w1_zeros_packed,
        w2_zeros_packed,
        w1_s,
        w2_s,
    ) = _make_int4_moe_weights(E, N, K, group_size, quant_algo)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = _ref_int4_moe(
        a,
        w1_int4,
        w2_int4,
        w1_zeros,
        w2_zeros,
        w1_s,
        w2_s,
        topk_weight,
        topk_ids,
        group_size,
    )

    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        prepare_int4_moe_layer_for_cpu,
    )

    (blocked_w1, blocked_w2, blocked_s1, blocked_s2, blocked_z1, blocked_z2) = (
        prepare_int4_moe_layer_for_cpu(
            w1_packed,
            w2_packed,
            w1_s,
            w2_s,
            quant_algo=quant_algo,
            w13_zeros=w1_zeros_packed,
            w2_zeros=w2_zeros_packed,
        )
    )

    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        blocked_w1,
        blocked_w2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.INT4_W4A8,
        blocked_s1,
        blocked_s2,
        blocked_z1,
        blocked_z2,
        None,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        True,  # is_vnni
    )
    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("N,K,E,topk,group_size", [(256, 512, 8, 2, 128)])
@pytest.mark.parametrize("quant_algo", [ops.CPUQuantAlgo.GPTQ, ops.CPUQuantAlgo.AWQ])
def test_int4_w4a16_cpu_fused_moe_gelu_tanh(M, N, K, E, topk, group_size, quant_algo):
    """Test fused_experts_cpu INT4 W4A16 with GELU-tanh activation."""
    set_random_seed(0)

    a = torch.randn(M, K, dtype=torch.bfloat16) / (0.5 * K**0.5)
    (
        w1_int4,
        w2_int4,
        w1_packed,
        w2_packed,
        w1_zeros,
        w2_zeros,
        w1_zeros_packed,
        w2_zeros_packed,
        w1_s,
        w2_s,
    ) = _make_int4_moe_weights(E, N, K, group_size, quant_algo)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = _ref_int4_moe(
        a,
        w1_int4,
        w2_int4,
        w1_zeros,
        w2_zeros,
        w1_s,
        w2_s,
        topk_weight,
        topk_ids,
        group_size,
        act_fn=_gelu_tanh_and_mul,
    )

    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        prepare_int4_moe_layer_for_cpu,
    )

    (blocked_w1, blocked_w2, blocked_s1, blocked_s2, blocked_z1, blocked_z2) = (
        prepare_int4_moe_layer_for_cpu(
            w1_packed,
            w2_packed,
            w1_s,
            w2_s,
            quant_algo=quant_algo,
            w13_zeros=w1_zeros_packed,
            w2_zeros=w2_zeros_packed,
        )
    )

    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        blocked_w1,
        blocked_w2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.INT4_W4A8,
        blocked_s1,
        blocked_s2,
        blocked_z1,
        blocked_z2,
        None,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        True,  # is_vnni
        None,  # a1_scale
        ops.CPUActMethod.GELU_TANH_AND_MUL,
    )
    torch.testing.assert_close(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)


# ===========================================================================
# INT8 W8A8 MoE
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


def _ref_int8_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, act_fn=_silu_and_mul):
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
            ic = act_fn(ic)

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
    """Test fused_experts_cpu INT8 W8A8 against torch reference.

    ``inplace`` exercises writing into a fresh output buffer vs. aliasing
    the input tensor as the output buffer.
    """
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

    a_in = a.clone()
    out = a_in if inplace else torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        w1,
        w2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.INT8_W8A8,
        w1_s,
        w2_s,
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        is_vnni,
    )
    torch.testing.assert_close(
        ref_out.bfloat16(),
        out,
        atol=2e-1,
        rtol=2e-1,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_int8_w8a8_cpu_fused_moe_small_expert_blocks(dtype):
    """Test INT8 dispatch at exact and multi-block expert boundaries."""
    set_random_seed(0)
    block_sizes = (3, 4, 5, 33)
    N, K, E = 128, 128, len(block_sizes)
    M = sum(block_sizes)

    a = torch.randn(M, K, dtype=dtype) / (0.5 * K**0.5)
    w1_q, w2_q, w1_s, w2_s = _make_int8_moe_weights(E, N, K)
    topk_weight, topk_ids = _deterministic_expert_routes(block_sizes)

    ref_out = _ref_int8_moe(a, w1_q, w2_q, w1_s, w2_s, topk_weight, topk_ids)
    pw1, pw2 = _prepack_experts(w1_q), _prepack_experts(w2_q)
    out = torch.empty_like(a)
    ops.fused_experts_cpu(
        out,
        a,
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.INT8_W8A8,
        w1_s,
        w2_s,
        None,
        None,
        None,
        is_vnni=True,
    )

    torch.testing.assert_close(ref_out, out, atol=2e-1, rtol=2e-1)


@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("N,K,E,topk", [(256, 512, 8, 2), (768, 2048, 8, 2)])
@pytest.mark.parametrize("is_vnni", [False, True])
def test_int8_w8a8_cpu_fused_moe_gelu_tanh(M, N, K, E, topk, is_vnni):
    """Test fused_experts_cpu INT8 W8A8 with GELU-tanh activation."""
    set_random_seed(0)

    a = torch.randn(M, K, dtype=torch.bfloat16) / (0.5 * K**0.5)
    w1_q, w2_q, w1_s, w2_s = _make_int8_moe_weights(E, N, K)

    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    ref_out = _ref_int8_moe(
        a, w1_q, w2_q, w1_s, w2_s, topk_weight, topk_ids, act_fn=_gelu_tanh_and_mul
    )

    w1 = _prepack_experts(w1_q) if is_vnni else w1_q
    w2 = _prepack_experts(w2_q) if is_vnni else w2_q

    a_in = a.clone()
    out = torch.empty_like(a_in)
    ops.fused_experts_cpu(
        out,
        a_in,
        w1,
        w2,
        topk_weight,
        topk_ids,
        ops.CPUQuantMethod.INT8_W8A8,
        w1_s,
        w2_s,
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        is_vnni,
        None,  # a1_scale
        ops.CPUActMethod.GELU_TANH_AND_MUL,
    )
    torch.testing.assert_close(
        ref_out.bfloat16(),
        out,
        atol=2e-1,
        rtol=2e-1,
    )


# ===========================================================================
# FP8 W8A8 MoE
# ===========================================================================


def _quantize_e4m3_per_token(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the native quantizer and validate either supported E4M3 rounding."""
    expected_q, expected_scales = _quantize_per_token_reference(x, torch.float8_e4m3fn)
    avx10_q, _ = _quantize_per_token_reference(
        x, torch.float8_e4m3fn, round_through_fp16=True
    )
    native_q, native_scales = torch.ops._C.quantize_fp8e4m3_vec(x, True, None)
    native_bytes = native_q.view(torch.uint8)
    assert torch.equal(native_bytes, expected_q.view(torch.uint8)) or torch.equal(
        native_bytes, avx10_q.view(torch.uint8)
    )
    torch.testing.assert_close(native_scales, expected_scales, atol=0, rtol=0)
    return native_q, native_scales


@requires_cpu_native_e4m3
@pytest.mark.parametrize(
    "M,N,K,E,topk,biased,expert_blocks,activation_name",
    [
        (43, 128, 128, 4, 1, False, (1, 4, 5, 33), "silu"),
        (43, 128, 128, 4, 1, False, (1, 4, 5, 33), "gelu"),
        (43, 128, 128, 4, 1, False, (1, 4, 5, 33), "gelu_tanh"),
        (8, 256, 512, 8, 2, True, None, "silu"),
    ],
)
def test_fp8_w8a8_cpu_fused_moe(
    default_vllm_config, M, N, K, E, topk, biased, expert_blocks, activation_name
):
    """Check FP8 factory initialization, activations, and top-k reduction."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUExpertsFp8W8A8,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        make_fp8_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    activation, act_fn = {
        "silu": (MoEActivation.SILU, _silu_and_mul),
        "gelu": (MoEActivation.GELU, _gelu_and_mul),
        "gelu_tanh": (MoEActivation.GELU_TANH, _gelu_tanh_and_mul),
    }[activation_name]

    moe_config = make_dummy_moe_config(
        num_experts=E,
        experts_per_token=topk,
        hidden_dim=K,
        intermediate_size=N,
        max_num_tokens=M,
        activation=activation,
    )
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    moe_config.routing_method = RoutingMethodType.Default
    supported, reason = CPUExpertsFp8W8A8.is_supported_config(
        CPUExpertsFp8W8A8,
        moe_config,
        kFp8Static128BlockSym,
        kFp8Dynamic128Sym,
        CPUExpertsFp8W8A8.activation_format(),
    )
    assert supported, (
        f"expected FP8 W8A8 experts to support {activation_name}, got {reason}"
    )

    set_random_seed(42)

    w13, w2, w13_scale, w2_scale = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)
    w13_scale.abs_()
    w2_scale.abs_()
    (
        packed_w13,
        packed_w13_scale,
        packed_w2,
        packed_w2_scale,
    ) = prepare_fp8_w8a8_moe_layer_for_cpu(w13, w2, w13_scale, w2_scale)

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    if M > 1:
        hidden_states[1].zero_()
    if M > 2:
        hidden_states[2] *= 0.01
    if M > 3:
        hidden_states[3] *= 5
    x_fp8, x_scales = _quantize_e4m3_per_token(hidden_states)
    if expert_blocks is None:
        scores = torch.softmax(torch.randn(M, E), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, topk)
        topk_ids = topk_ids.to(torch.int32)
    else:
        topk_weights, topk_ids = _deterministic_expert_routes(expert_blocks)

    w1_bias = torch.randn(E, 2 * N, dtype=torch.bfloat16) / 10 if biased else None
    w2_bias = torch.randn(E, K, dtype=torch.bfloat16) / 10 if biased else None
    if w1_bias is not None:
        w1_bias[:, :4] = torch.tensor([8.0, -8.0, 8.0, 8.0])
    quant_config = make_fp8_moe_quant_config(
        Fp8MoeBackend.CPU_W8A8,
        packed_w13_scale,
        packed_w2_scale,
        None,
        None,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        block_shape=BLOCK_SIZE,
    )
    output = _run_cpu_fp8_modular_kernel(
        CPUExpertsFp8W8A8,
        quant_config,
        hidden_states,
        packed_w13,
        packed_w2,
        topk_weights,
        topk_ids,
        N,
        activation=activation,
    )

    ref = _ref_cpu_fp8_activation_moe(
        hidden_states,
        _block_dequant_weight(w13, w13_scale, BLOCK_SIZE),
        _block_dequant_weight(w2, w2_scale, BLOCK_SIZE),
        topk_weights,
        topk_ids.long(),
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        input_quantization=(x_fp8, x_scales),
        act_fn=act_fn,
    )
    torch.testing.assert_close(
        output,
        ref,
        atol=1e-2 * ref.float().abs().max().item(),
        rtol=1e-2,
    )


def _run_cpu_fp8_modular_kernel(
    experts_cls,
    quant_config,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    intermediate_size: int,
    activation=None,
) -> torch.Tensor:
    """Run CPU experts through modular prepare and finalize."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        make_fp8_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    if activation is None:
        activation = MoEActivation.SILU
    num_experts = w1.shape[0]
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk_ids.shape[1],
        hidden_dim=hidden_states.shape[1],
        intermediate_size=intermediate_size,
        max_num_tokens=hidden_states.shape[0],
        activation=activation,
    )
    if quant_config.use_mxfp4_w4a8:
        experts = experts_cls(moe_config, quant_config)
        experts.process_weights_after_loading(torch.nn.Module())
        assert experts.w1_block_metadata is not None
        assert experts.w2_block_metadata is not None
        kernel = FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)
    else:
        kernel = make_fp8_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=moe_config,
            experts_cls=experts_cls,
            fp8_backend=Fp8MoeBackend.CPU_W8A8,
        )
    return kernel.apply(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=activation,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


@requires_cpu_native_e4m3
@pytest.mark.parametrize(
    ("use_w1_bias", "use_w2_bias", "all_tiny"),
    [
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
    ids=["both_biases", "w1_bias_only", "w2_bias_only", "all_tiny_both_biases"],
)
def test_mxfp4_w4a8_cpu_modular_kernel_quantizes_inputs_and_reduces_topk(
    use_w1_bias, use_w2_bias, all_tiny
):
    """Check tiny and mixed GPT-OSS E4M3 blocks with bias and top-k reduction."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUExpertsMxfp4W4A8,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_quant_config,
    )

    set_random_seed(12)
    M, N, K, E = (4 if all_tiny else 5), 96, 160, 3
    alpha, limit = 1.702, 7.0
    hidden_states = torch.randn(M, K, dtype=torch.bfloat16) / 10
    hidden_states[0].zero_()
    hidden_states[1] *= 0.01
    hidden_states[2] *= 4
    w1_bf16 = torch.randn(E, 2 * N, K, dtype=torch.bfloat16) / 10
    w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
    w1s = w1s.reshape(E, 2 * N, K // 32)
    w1dq = MXFP4QuantizeUtil.dequantize(w1q, torch.bfloat16, w1s)
    w2_bf16 = torch.randn(E, K, N, dtype=torch.bfloat16) / 10
    w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
    w2s = w2s.reshape(E, K, N // 32)
    w2dq = MXFP4QuantizeUtil.dequantize(w2q, torch.bfloat16, w2s)
    w1_bias = torch.randn(E, 2 * N, dtype=torch.float32) / 10
    w1_bias[:, :4] = torch.tensor([8.0, -8.0, 8.0, 8.0])
    w2_bias = torch.randn(E, K, dtype=torch.float32) / 10
    w1_bias = w1_bias if use_w1_bias else None
    w2_bias = w2_bias if use_w2_bias else None
    packed_w1, packed_w1s = _prepack_mxfp4_experts(w1q, w1s)
    packed_w2, packed_w2s = _prepack_mxfp4_experts(w2q, w2s)
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.CPU_MXFP4_FP8,
        packed_w1s,
        packed_w2s,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        gemm1_alpha=alpha,
        gemm1_beta=1.0,
        swiglu_limit=limit,
    )
    topk_ids = torch.tensor(
        [[0, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
        dtype=torch.int32,
    )[:M]
    topk_weights = torch.tensor(
        [[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    )[:M]
    ref = _ref_cpu_fp8_activation_moe(
        hidden_states,
        w1dq,
        w2dq,
        topk_weights,
        topk_ids,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        input_quantization=_quantize_e4m3_per_token(hidden_states),
        mxfp4=True,
    )

    output = _run_cpu_fp8_modular_kernel(
        CPUExpertsMxfp4W4A8,
        quant_config,
        hidden_states,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        N,
        activation=MoEActivation.SWIGLUOAI,
    )
    torch.testing.assert_close(
        output,
        ref,
        atol=1e-2 * ref.float().abs().max().item(),
        rtol=1e-2,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
