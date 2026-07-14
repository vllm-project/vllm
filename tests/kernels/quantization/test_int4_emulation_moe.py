#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Int4EmulationTritonExperts MoE backend.

Tests the weight dequantization helpers (_unpack_and_dequant_int4_gptq,
_unpack_and_dequant_int4_awq) and full MoE forward pass
(_process_weights_emulation_gptq, _process_weights_emulation_awq)
for both symmetric and asymmetric zero-point cases.

Run `pytest tests/kernels/quantization/test_int4_emulation_moe.py`.
"""

import numpy
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe import (
    Int4EmulationTritonExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _process_weights_emulation_awq,
    _process_weights_emulation_gptq,
    _unpack_and_dequant_int4_awq,
    _unpack_and_dequant_int4_gptq,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    awq_pack,
    gptq_pack,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Int4EmulationTritonExperts requires CUDA.",
)

device = "cuda"

# (E, K, N, group_size)
SHAPES = [
    pytest.param(2, 64, 32, 32, id="tiny-gs32"),
    pytest.param(4, 128, 64, 64, id="small-gs64"),
    pytest.param(4, 256, 128, 128, id="medium-gs128"),
]

# (E, K, N, top_k, group_size, num_tokens)
E2E_CONFIGS = [
    pytest.param(4, 64, 32, 2, 32, 8, id="tiny"),
    pytest.param(8, 128, 64, 2, 64, 16, id="small"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_sym(w_fp: torch.Tensor, group_size: int):
    """Quantize [K, N] float to int4 symmetric (uint4b8), return q and scale."""
    K, N = w_fp.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w_grouped = w_fp.reshape(n_groups, group_size, N)
    scale = w_grouped.abs().amax(dim=1) / 7.0
    scale = scale.clamp(min=1e-6)
    w_quant = (w_grouped / scale.unsqueeze(1)).round().clamp(-8, 7)
    q = (w_quant + 8).to(torch.int32).reshape(K, N)
    return q, scale


def _quantize_asym(w_fp: torch.Tensor, group_size: int):
    """Quantize [K, N] float to uint4 asymmetric, return q, scale, zero."""
    K, N = w_fp.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w_grouped = w_fp.reshape(n_groups, group_size, N)
    wmin = w_grouped.amin(dim=1)
    wmax = w_grouped.amax(dim=1)
    scale = (wmax - wmin) / 15.0
    scale = scale.clamp(min=1e-6)
    zero = (-wmin / scale).round().clamp(0, 15).to(torch.int32)
    w_quant = ((w_grouped - wmin.unsqueeze(1)) / scale.unsqueeze(1)).round()
    q = w_quant.clamp(0, 15).to(torch.int32).reshape(K, N)
    return q, scale, zero


def _dequantize_ref(
    w_uint: torch.Tensor,
    scale: torch.Tensor,
    zero=None,
    output_dtype: torch.dtype = torch.bfloat16,
):
    """Reference dequant for a single [K, N] slice."""
    K, N = w_uint.shape
    n_groups = scale.shape[0]
    group_size = K // n_groups
    w = w_uint.reshape(n_groups, group_size, N).to(output_dtype)
    s = scale.unsqueeze(1).to(output_dtype)
    if zero is None:
        return ((w - 8) * s).reshape(K, N)
    z = zero.unsqueeze(1).to(output_dtype)
    return ((w - z) * s).reshape(K, N)


def _pack_gptq_zeros(zero: torch.Tensor, N: int) -> torch.Tensor:
    """Pack [n_groups, N] zeros into GPTQ format [n_groups, N//8] int32."""
    n_groups, _ = zero.shape
    z = zero.to(torch.int32).cpu().numpy().astype(numpy.uint32)
    packed = numpy.zeros((n_groups, N // 8), dtype=numpy.uint32)
    for i in range(8):
        packed |= z[:, i::8] << (i * 4)
    return torch.from_numpy(packed.astype(numpy.int32)).to(device)


def _pack_awq_zeros(zero: torch.Tensor, N: int) -> torch.Tensor:
    """Pack [n_groups, N] zeros into AWQ column format [n_groups, N//8] int32."""
    n_groups, _ = zero.shape
    interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    z = zero.to(torch.int32).cpu().numpy().astype(numpy.uint32)
    z_interleaved = z.reshape(-1, 8)[:, interleave].reshape(n_groups, N)
    packed = numpy.zeros((n_groups, N // 8), dtype=numpy.uint32)
    for i in range(8):
        packed |= z_interleaved[:, i::8] << (i * 4)
    return torch.from_numpy(packed.astype(numpy.int32)).to(device)


def _make_moe_config(E, K, N):
    return FusedMoEConfig(
        num_experts=E,
        experts_per_token=2,
        hidden_dim=K,
        intermediate_size=N,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )


def _make_gptq_moe_weights(E, K, N, group_size, asym=False):
    """Build GPTQ MoE weight tensors and per-expert float references."""
    torch.manual_seed(7)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

    w13_list, w13s, w13z, w13_ref = [], [], [], []
    w2_list, w2s, w2z, w2_ref = [], [], [], []

    for e in range(E):
        if asym:
            q13, s13, z13 = _quantize_asym(w13_fp[e], group_size)
            w13_list.append(gptq_pack(q13, 4, K, 2 * N))
            w13s.append(s13)
            w13z.append(_pack_gptq_zeros(z13, 2 * N))
            w13_ref.append(_dequantize_ref(q13, s13, z13))
            q2, s2, z2 = _quantize_asym(w2_fp[e], group_size)
            w2_list.append(gptq_pack(q2, 4, N, K))
            w2s.append(s2)
            w2z.append(_pack_gptq_zeros(z2, K))
            w2_ref.append(_dequantize_ref(q2, s2, z2))
        else:
            q13, s13 = _quantize_sym(w13_fp[e], group_size)
            w13_list.append(gptq_pack(q13, 4, K, 2 * N))
            w13s.append(s13)
            w13z.append(None)
            w13_ref.append(_dequantize_ref(q13, s13))
            q2, s2 = _quantize_sym(w2_fp[e], group_size)
            w2_list.append(gptq_pack(q2, 4, N, K))
            w2s.append(s2)
            w2z.append(None)
            w2_ref.append(_dequantize_ref(q2, s2))

    return (
        torch.stack(w13_list),
        torch.stack(w13s),
        torch.stack(w13z) if asym else None,
        torch.stack(w13_ref),  # [E, K, 2N]
        torch.stack(w2_list),
        torch.stack(w2s),
        torch.stack(w2z) if asym else None,
        torch.stack(w2_ref),  # [E, N, K]
    )


def _make_awq_moe_weights(E, K, N, group_size, asym=False):
    """Build AWQ MoE weight tensors and per-expert float references."""
    torch.manual_seed(8)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

    w13_list, w13s, w13z, w13_ref = [], [], [], []
    w2_list, w2s, w2z, w2_ref = [], [], [], []

    for e in range(E):
        if asym:
            q13, s13, z13 = _quantize_asym(w13_fp[e], group_size)
            w13_list.append(awq_pack(q13, 4, K, 2 * N))
            w13s.append(s13)
            w13z.append(_pack_awq_zeros(z13, 2 * N))
            w13_ref.append(_dequantize_ref(q13, s13, z13))
            q2, s2, z2 = _quantize_asym(w2_fp[e], group_size)
            w2_list.append(awq_pack(q2, 4, N, K))
            w2s.append(s2)
            w2z.append(_pack_awq_zeros(z2, K))
            w2_ref.append(_dequantize_ref(q2, s2, z2))
        else:
            q13, s13 = _quantize_sym(w13_fp[e], group_size)
            w13_list.append(awq_pack(q13, 4, K, 2 * N))
            w13s.append(s13)
            w13z.append(None)
            w13_ref.append(_dequantize_ref(q13, s13))
            q2, s2 = _quantize_sym(w2_fp[e], group_size)
            w2_list.append(awq_pack(q2, 4, N, K))
            w2s.append(s2)
            w2z.append(None)
            w2_ref.append(_dequantize_ref(q2, s2))

    return (
        torch.stack(w13_list),
        torch.stack(w13s),
        torch.stack(w13z) if asym else None,
        torch.stack(w13_ref),  # [E, K, 2N]
        torch.stack(w2_list),
        torch.stack(w2s),
        torch.stack(w2z) if asym else None,
        torch.stack(w2_ref),  # [E, N, K]
    )


def _run_emulation_forward(
    experts, w13_bf16, w2_bf16, hidden_states, topk_weights, topk_ids, E, K, N
):
    ws13_size = hidden_states.shape[0] * topk_ids.shape[1] * max(N, K)
    ws2_size = hidden_states.shape[0] * topk_ids.shape[1] * max(2 * N, K)
    workspace13 = torch.zeros(ws13_size, dtype=hidden_states.dtype, device=device)
    workspace2 = torch.zeros(ws2_size, dtype=hidden_states.dtype, device=device)
    output = torch.zeros(
        hidden_states.shape[0], K, dtype=hidden_states.dtype, device=device
    )
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w13_bf16,
        w2=w2_bf16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return output


# ---------------------------------------------------------------------------
# Tests: _unpack_and_dequant_int4_gptq
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_gptq_unpack_symmetric(E, K, N, group_size):
    """GPTQ symmetric unpacker matches reference."""
    torch.manual_seed(0)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list, ref_list = [], [], []
    for e in range(E):
        q, s = _quantize_sym(w_fp[e], group_size)
        packed_list.append(gptq_pack(q, 4, K, N))
        scale_list.append(s)
        ref_list.append(_dequantize_ref(q, s, output_dtype=torch.float32))

    w_packed = torch.stack(packed_list).to(device)
    scale = torch.stack(scale_list).to(device)
    ref = torch.stack(ref_list).to(device)

    out = _unpack_and_dequant_int4_gptq(
        w_packed, scale, None, transpose_output=False, output_dtype=torch.float32
    )

    assert out.shape == (E, K, N)
    assert torch.allclose(out, ref, atol=0), (
        f"max diff: {(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_gptq_unpack_asymmetric(E, K, N, group_size):
    """GPTQ asymmetric unpacker matches reference."""
    torch.manual_seed(1)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list, zero_list, ref_list = [], [], [], []
    for e in range(E):
        q, s, z = _quantize_asym(w_fp[e], group_size)
        packed_list.append(gptq_pack(q, 4, K, N))
        scale_list.append(s)
        zero_list.append(_pack_gptq_zeros(z, N))
        ref_list.append(_dequantize_ref(q, s, z, output_dtype=torch.float32))

    w_packed = torch.stack(packed_list).to(device)
    scale = torch.stack(scale_list).to(device)
    qzeros = torch.stack(zero_list).to(device)
    ref = torch.stack(ref_list).to(device)

    out = _unpack_and_dequant_int4_gptq(
        w_packed, scale, qzeros, transpose_output=False, output_dtype=torch.float32
    )

    assert out.shape == (E, K, N)
    assert torch.allclose(out, ref, atol=0), (
        f"max diff: {(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_gptq_unpack_transpose(E, K, N, group_size):
    """GPTQ transpose_output=True gives [E, N, K]."""
    torch.manual_seed(2)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list = [], []
    for e in range(E):
        q, s = _quantize_sym(w_fp[e], group_size)
        packed_list.append(gptq_pack(q, 4, K, N))
        scale_list.append(s)

    w_packed = torch.stack(packed_list).to(device)
    scale = torch.stack(scale_list).to(device)

    out_normal = _unpack_and_dequant_int4_gptq(w_packed, scale, None, False)
    out_transposed = _unpack_and_dequant_int4_gptq(w_packed, scale, None, True)

    assert out_transposed.shape == (E, N, K)
    assert torch.allclose(
        out_transposed, out_normal.permute(0, 2, 1).contiguous(), atol=0
    )


# ---------------------------------------------------------------------------
# Tests: _unpack_and_dequant_int4_awq
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_awq_unpack_symmetric(E, K, N, group_size):
    """AWQ symmetric unpacker matches reference."""
    torch.manual_seed(3)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list, ref_list = [], [], []
    for e in range(E):
        q, s = _quantize_sym(w_fp[e], group_size)
        packed_list.append(awq_pack(q, 4, K, N))
        scale_list.append(s)
        ref_list.append(_dequantize_ref(q, s, output_dtype=torch.float32))

    w_packed = torch.stack(packed_list).to(device)
    scale = torch.stack(scale_list).to(device)
    ref = torch.stack(ref_list).to(device)

    out = _unpack_and_dequant_int4_awq(
        w_packed, scale, None, transpose_output=False, output_dtype=torch.float32
    )

    assert out.shape == (E, K, N)
    assert torch.allclose(out, ref, atol=0), (
        f"max diff: {(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_awq_unpack_asymmetric(E, K, N, group_size):
    """AWQ asymmetric unpacker matches reference."""
    torch.manual_seed(4)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list, zero_list, ref_list = [], [], [], []
    for e in range(E):
        q, s, z = _quantize_asym(w_fp[e], group_size)
        packed_list.append(awq_pack(q, 4, K, N))
        scale_list.append(s)
        zero_list.append(_pack_awq_zeros(z, N))
        ref_list.append(_dequantize_ref(q, s, z, output_dtype=torch.float32))

    w_packed = torch.stack(packed_list).to(device)
    scale = torch.stack(scale_list).to(device)
    qzeros = torch.stack(zero_list).to(device)
    ref = torch.stack(ref_list).to(device)

    out = _unpack_and_dequant_int4_awq(
        w_packed, scale, qzeros, transpose_output=False, output_dtype=torch.float32
    )

    assert out.shape == (E, K, N)
    assert torch.allclose(out, ref, atol=0), (
        f"max diff: {(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_awq_unpack_transpose(E, K, N, group_size):
    """AWQ transpose_output=True gives [E, N, K]."""
    torch.manual_seed(5)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list = [], []
    for e in range(E):
        q, s = _quantize_sym(w_fp[e], group_size)
        packed_list.append(awq_pack(q, 4, K, N))
        scale_list.append(s)

    w_packed = torch.stack(packed_list).to(device)
    scale = torch.stack(scale_list).to(device)

    out_normal = _unpack_and_dequant_int4_awq(w_packed, scale, None, False)
    out_transposed = _unpack_and_dequant_int4_awq(w_packed, scale, None, True)

    assert out_transposed.shape == (E, N, K)
    assert torch.allclose(
        out_transposed, out_normal.permute(0, 2, 1).contiguous(), atol=0
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_awq_gptq_unpack_agree(E, K, N, group_size):
    """AWQ and GPTQ unpackers produce identical values for the same weights."""
    torch.manual_seed(6)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    gptq_list, awq_list, scale_list = [], [], []
    for e in range(E):
        q, s = _quantize_sym(w_fp[e], group_size)
        gptq_list.append(gptq_pack(q, 4, K, N))
        awq_list.append(awq_pack(q, 4, K, N))
        scale_list.append(s)

    scale = torch.stack(scale_list).to(device)
    out_gptq = _unpack_and_dequant_int4_gptq(
        torch.stack(gptq_list).to(device), scale, None, False, torch.float32
    )
    out_awq = _unpack_and_dequant_int4_awq(
        torch.stack(awq_list).to(device), scale, None, False, torch.float32
    )

    assert torch.allclose(out_gptq, out_awq, atol=0), (
        f"max diff: {(out_gptq - out_awq).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Tests: _process_weights_emulation_{gptq,awq}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_gptq_process_weights_shapes_and_values(E, K, N, group_size, asym):
    """_process_weights_emulation_gptq shapes and values match reference."""
    w13, w13s, w13z, w13_ref, w2, w2s, w2z, w2_ref = _make_gptq_moe_weights(
        E, K, N, group_size, asym
    )
    result = _process_weights_emulation_gptq(w13, w2, w13s, w2s, w13z, w2z)
    w13_out, w2_out = result[0], result[1]

    assert w13_out.shape == (E, 2 * N, K)
    assert w2_out.shape == (E, K, N)
    assert w13_out.dtype == torch.bfloat16
    assert w2_out.dtype == torch.bfloat16

    expected_w13 = w13_ref.permute(0, 2, 1)
    expected_w2 = w2_ref.permute(0, 2, 1)

    assert torch.allclose(w13_out.float(), expected_w13.float(), atol=0), (
        f"w13 max diff: {(w13_out.float() - expected_w13.float()).abs().max().item()}"
    )
    assert torch.allclose(w2_out.float(), expected_w2.float(), atol=0), (
        f"w2 max diff: {(w2_out.float() - expected_w2.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_awq_process_weights_shapes_and_values(E, K, N, group_size, asym):
    """_process_weights_emulation_awq shapes and values match reference."""
    w13, w13s, w13z, w13_ref, w2, w2s, w2z, w2_ref = _make_awq_moe_weights(
        E, K, N, group_size, asym
    )
    result = _process_weights_emulation_awq(w13, w2, w13s, w2s, w13z, w2z)
    w13_out, w2_out = result[0], result[1]

    assert w13_out.shape == (E, 2 * N, K)
    assert w2_out.shape == (E, K, N)
    assert w13_out.dtype == torch.bfloat16
    assert w2_out.dtype == torch.bfloat16

    expected_w13 = w13_ref.permute(0, 2, 1)
    expected_w2 = w2_ref.permute(0, 2, 1)

    assert torch.allclose(w13_out.float(), expected_w13.float(), atol=0), (
        f"w13 max diff: {(w13_out.float() - expected_w13.float()).abs().max().item()}"
    )
    assert torch.allclose(w2_out.float(), expected_w2.float(), atol=0), (
        f"w2 max diff: {(w2_out.float() - expected_w2.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_gptq_awq_process_weights_agree(E, K, N, group_size, asym):
    """AWQ and GPTQ process_weights produce identical dequantized tensors."""
    torch.manual_seed(9)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

    g13, g13s, g13z = [], [], []
    a13, a13s, a13z = [], [], []
    g2, g2s, g2z = [], [], []
    a2, a2s, a2z = [], [], []

    for e in range(E):
        if asym:
            q13, s13, z13 = _quantize_asym(w13_fp[e], group_size)
            q2, s2, z2 = _quantize_asym(w2_fp[e], group_size)
            g13z.append(_pack_gptq_zeros(z13, 2 * N))
            a13z.append(_pack_awq_zeros(z13, 2 * N))
            g2z.append(_pack_gptq_zeros(z2, K))
            a2z.append(_pack_awq_zeros(z2, K))
        else:
            q13, s13 = _quantize_sym(w13_fp[e], group_size)
            q2, s2 = _quantize_sym(w2_fp[e], group_size)
            g13z.append(None)
            a13z.append(None)
            g2z.append(None)
            a2z.append(None)

        g13.append(gptq_pack(q13, 4, K, 2 * N))
        a13.append(awq_pack(q13, 4, K, 2 * N))
        g13s.append(s13)
        a13s.append(s13)
        g2.append(gptq_pack(q2, 4, N, K))
        a2.append(awq_pack(q2, 4, N, K))
        g2s.append(s2)
        a2s.append(s2)

    gptq_res = _process_weights_emulation_gptq(
        torch.stack(g13),
        torch.stack(g2),
        torch.stack(g13s),
        torch.stack(g2s),
        torch.stack(g13z) if asym else None,
        torch.stack(g2z) if asym else None,
    )
    awq_res = _process_weights_emulation_awq(
        torch.stack(a13),
        torch.stack(a2),
        torch.stack(a13s),
        torch.stack(a2s),
        torch.stack(a13z) if asym else None,
        torch.stack(a2z) if asym else None,
    )

    assert torch.allclose(gptq_res[0].float(), awq_res[0].float(), atol=1e-3), (
        f"w13 max diff: {(gptq_res[0] - awq_res[0]).float().abs().max().item()}"
    )
    assert torch.allclose(gptq_res[1].float(), awq_res[1].float(), atol=1e-3), (
        f"w2 max diff: {(gptq_res[1] - awq_res[1]).float().abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# End-to-end MoE forward pass tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS)
def test_gptq_vs_awq_forward_agree(E, K, N, top_k, group_size, num_tokens):
    """GPTQ and AWQ emulation backends produce bit-identical forward outputs."""
    torch.manual_seed(42)
    moe_config = _make_moe_config(E, K, N)

    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

    g13, g13s, a13, a13s = [], [], [], []
    g2, g2s, a2, a2s = [], [], [], []

    for e in range(E):
        q13, s13 = _quantize_sym(w13_fp[e], group_size)
        q2, s2 = _quantize_sym(w2_fp[e], group_size)
        g13.append(gptq_pack(q13, 4, K, 2 * N))
        a13.append(awq_pack(q13, 4, K, 2 * N))
        g13s.append(s13)
        a13s.append(s13.clone())
        g2.append(gptq_pack(q2, 4, N, K))
        a2.append(awq_pack(q2, 4, N, K))
        g2s.append(s2)
        a2s.append(s2.clone())

    gptq_res = _process_weights_emulation_gptq(
        torch.stack(g13),
        torch.stack(g2),
        torch.stack(g13s),
        torch.stack(g2s),
        None,
        None,
    )
    awq_res = _process_weights_emulation_awq(
        torch.stack(a13),
        torch.stack(a2),
        torch.stack(a13s),
        torch.stack(a2s),
        None,
        None,
    )
    w13_gptq, w2_gptq = gptq_res[0], gptq_res[1]
    w13_awq, w2_awq = awq_res[0], awq_res[1]

    dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
    experts_gptq = Int4EmulationTritonExperts(
        moe_config, int4_w4a16_moe_quant_config(dummy_scale, dummy_scale)
    )
    experts_awq = Int4EmulationTritonExperts(
        moe_config, int4_w4a16_moe_quant_config(dummy_scale, dummy_scale)
    )

    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_gptq = _run_emulation_forward(
        experts_gptq, w13_gptq, w2_gptq, hidden_states, topk_weights, topk_ids, E, K, N
    )
    out_awq = _run_emulation_forward(
        experts_awq, w13_awq, w2_awq, hidden_states, topk_weights, topk_ids, E, K, N
    )

    assert torch.allclose(out_gptq, out_awq, atol=0), (
        f"max diff: {(out_gptq - out_awq).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS)
@pytest.mark.parametrize("fmt", ["gptq", "awq"])
def test_emulation_output_close_to_bf16_reference(
    E, K, N, top_k, group_size, num_tokens, fmt
):
    """Emulation output is close to a direct BF16 MoE forward."""
    torch.manual_seed(11)
    moe_config = _make_moe_config(E, K, N)

    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.bfloat16, device=device) * 0.02
    w2_fp = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.02

    packed13, scales13, packed2, scales2 = [], [], [], []
    for e in range(E):
        q13, s13 = _quantize_sym(w13_fp[e].float(), group_size)
        q2, s2 = _quantize_sym(w2_fp[e].float(), group_size)
        if fmt == "gptq":
            packed13.append(gptq_pack(q13, 4, K, 2 * N))
            packed2.append(gptq_pack(q2, 4, N, K))
        else:
            packed13.append(awq_pack(q13, 4, K, 2 * N))
            packed2.append(awq_pack(q2, 4, N, K))
        scales13.append(s13)
        scales2.append(s2)

    process_fn = (
        _process_weights_emulation_gptq
        if fmt == "gptq"
        else _process_weights_emulation_awq
    )
    res = process_fn(
        torch.stack(packed13),
        torch.stack(packed2),
        torch.stack(scales13),
        torch.stack(scales2),
        None,
        None,
    )
    w13_bf16, w2_bf16 = res[0], res[1]

    dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
    experts = Int4EmulationTritonExperts(
        moe_config, int4_w4a16_moe_quant_config(dummy_scale, dummy_scale)
    )

    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_emulation = _run_emulation_forward(
        experts, w13_bf16, w2_bf16, hidden_states, topk_weights, topk_ids, E, K, N
    )

    ref = torch.zeros(num_tokens, K, dtype=torch.bfloat16, device=device)
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m] @ w13_bf16[e].T
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up
            acc += w * (act @ w2_bf16[e].T).float()
        ref[m] = acc.bfloat16()

    rel_l2 = (
        torch.norm(out_emulation.float() - ref.float())
        / torch.norm(ref.float()).clamp(min=1e-6)
    ).item()
    assert rel_l2 < 0.15, f"[{fmt}] relative L2 = {rel_l2:.4f} (threshold 0.15)"
