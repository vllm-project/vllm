#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for TritonWNA16OTFExperts MoE backend.

TritonWNA16OTFExperts keeps weights in packed int4/int8 format and dequantizes
on-the-fly each forward pass (unlike Int4/Int8EmulationTritonExperts which
dequantize at load time to BF16).  Tests verify:

  - _process_weights_triton_wna16: weight layout for GPTQ/AWQ, sym and asym.
  - End-to-end forward: GPTQ sym/asym, AWQ sym/asym, int8 GPTQ sym vs reference.
  - EP correctness: expert_map output matches full forward.
  - TritonWNA16OTFExperts._supports_* returns correct values.
  - make_group_size_adjusted_weight_loader: w2 scale expansion, w13/weight passthrough.
  - OTF forward when N % group_size != 0 (e.g. N=192, gs=128, tp=8).

Run: pytest tests/kernels/quantization/test_triton_wna16_otf_moe.py
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
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16OTFExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _process_weights_emulation_awq,
    _process_weights_emulation_gptq,
    _process_weights_emulation_int8,
    _process_weights_triton_wna16,
    make_group_size_adjusted_weight_loader,
)
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    awq_pack,
    gptq_pack,
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
    kInt8Static,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="TritonWNA16OTFExperts requires CUDA/ROCm.",
)

device = "cuda"

# (E, K, N, group_size)
SHAPES_LIST = [
    pytest.param(2, 64, 32, 32, id="tiny-gs32"),
    pytest.param(4, 128, 64, 64, id="small-gs64"),
    pytest.param(4, 256, 128, 128, id="medium-gs128"),
]

# (E, K, N, top_k, group_size, num_tokens)
E2E_CONFIGS_LIST = [
    pytest.param(4, 64, 32, 2, 32, 8, id="tiny"),
    pytest.param(8, 128, 64, 2, 64, 16, id="small"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_sym_int4(w_fp: torch.Tensor, group_size: int):
    K, N = w_fp.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w_grouped = w_fp.reshape(n_groups, group_size, N)
    scale = w_grouped.abs().amax(dim=1) / 7.0
    scale = scale.clamp(min=1e-6)
    w_quant = (w_grouped / scale.unsqueeze(1)).round().clamp(-8, 7)
    q = (w_quant + 8).to(torch.int32).reshape(K, N)
    return q, scale.to(torch.float16)


def _quantize_asym_int4(w_fp: torch.Tensor, group_size: int):
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
    return q, scale.to(torch.float16), zero


def _dequantize_ref_int4(w_uint, scale, zero=None, output_dtype=torch.bfloat16):
    K, N = w_uint.shape
    n_groups = scale.shape[0]
    gs = K // n_groups
    w = w_uint.reshape(n_groups, gs, N).to(output_dtype)
    s = scale.unsqueeze(1).to(output_dtype)
    if zero is None:
        return ((w - 8) * s).reshape(K, N)
    z = zero.unsqueeze(1).to(output_dtype)
    return ((w - z) * s).reshape(K, N)


def _pack_gptq_zeros(zero: torch.Tensor, N: int) -> torch.Tensor:
    n_groups, _ = zero.shape
    z = zero.to(torch.int32).cpu().numpy().astype(numpy.uint32)
    packed = numpy.zeros((n_groups, N // 8), dtype=numpy.uint32)
    for i in range(8):
        packed |= z[:, i::8] << (i * 4)
    return torch.from_numpy(packed.astype(numpy.int32)).to(device)


def _pack_awq_zeros(zero: torch.Tensor, N: int) -> torch.Tensor:
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


class _FakeGPTQConfig:
    """Minimal quant config stub for _process_weights_triton_wna16 (int4)."""

    weight_bits = 4
    num_bits = 4


class _FakeGPTQInt8Config:
    """Minimal quant config stub for _process_weights_triton_wna16 (int8)."""

    weight_bits = 8
    num_bits = 8


def _make_gptq_tensors(E, K, N, group_size, asym=False):
    """Build GPTQ-format w13/w2 tensors [E, K//8, 2N] int32."""
    torch.manual_seed(42)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device) * 0.1
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device) * 0.1

    w13_list, w13s_list, w13z_list = [], [], []
    w2_list, w2s_list, w2z_list = [], [], []

    for e in range(E):
        if asym:
            q13, s13, z13 = _quantize_asym_int4(w13_fp[e], group_size)
            q2, s2, z2 = _quantize_asym_int4(w2_fp[e], group_size)
            w13z_list.append(_pack_gptq_zeros(z13, 2 * N))  # [gs, 2N//8]
            w2z_list.append(_pack_gptq_zeros(z2, K))  # [gs, K//8]
        else:
            q13, s13 = _quantize_sym_int4(w13_fp[e], group_size)
            q2, s2 = _quantize_sym_int4(w2_fp[e], group_size)
        w13_list.append(gptq_pack(q13, 4, K, 2 * N))  # [K//8, 2N]
        w13s_list.append(s13)  # [K//gs, 2N]
        w2_list.append(gptq_pack(q2, 4, N, K))  # [N//8, K]
        w2s_list.append(s2)  # [N//gs, K]

    w13 = torch.stack(w13_list)
    w2 = torch.stack(w2_list)
    w13_scale = torch.stack(w13s_list)
    w2_scale = torch.stack(w2s_list)
    w13_qzeros = torch.stack(w13z_list) if asym else None
    w2_qzeros = torch.stack(w2z_list) if asym else None
    return w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros, w13_fp, w2_fp


def _make_awq_config(group_size: int, zero_point: bool) -> AutoAWQConfig:
    return AutoAWQConfig(
        weight_bits=4,
        group_size=group_size,
        zero_point=zero_point,
        lm_head_quantized=False,
    )


def _make_awq_tensors(E, K, N, group_size, asym=False):
    """Build AWQ-format w13/w2 tensors and float references.

    AWQ packs along N: w13 [E, K, 2N//8] int32, w2 [E, N, K//8] int32.
    Scales are column-parallel: w13_scale [E, K//gs, 2N], w2_scale [E, N//gs, K].
    """
    torch.manual_seed(43)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device) * 0.1
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device) * 0.1

    w13_list, w13s_list, w13z_list = [], [], []
    w2_list, w2s_list, w2z_list = [], [], []

    for e in range(E):
        if asym:
            q13, s13, z13 = _quantize_asym_int4(w13_fp[e], group_size)
            q2, s2, z2 = _quantize_asym_int4(w2_fp[e], group_size)
            w13z_list.append(_pack_awq_zeros(z13, 2 * N))
            w2z_list.append(_pack_awq_zeros(z2, K))
        else:
            q13, s13 = _quantize_sym_int4(w13_fp[e], group_size)
            q2, s2 = _quantize_sym_int4(w2_fp[e], group_size)
        w13_list.append(awq_pack(q13, 4, K, 2 * N))  # [K, 2N//8]
        w13s_list.append(s13)  # [K//gs, 2N]
        w2_list.append(awq_pack(q2, 4, N, K))  # [N, K//8]
        w2s_list.append(s2)  # [N//gs, K]

    w13 = torch.stack(w13_list)
    w2 = torch.stack(w2_list)
    w13_scale = torch.stack(w13s_list)
    w2_scale = torch.stack(w2s_list)
    w13_qzeros = torch.stack(w13z_list) if asym else None
    w2_qzeros = torch.stack(w2z_list) if asym else None
    return w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros, w13_fp, w2_fp


def _quantize_sym_int8(w_fp: torch.Tensor, group_size: int):
    """Quantize [K, N] float to int8 symmetric (uint8b128)."""
    K, N = w_fp.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w_grouped = w_fp.reshape(n_groups, group_size, N)
    scale = w_grouped.abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-6)
    w_quant = (w_grouped / scale.unsqueeze(1)).round().clamp(-128, 127)
    q = (w_quant + 128).to(torch.int32).reshape(K, N)
    return q, scale.to(torch.float16)


def _make_gptq_int8_tensors(E, K, N, group_size):
    """Build GPTQ-format int8 w13/w2 tensors."""
    torch.manual_seed(44)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float32, device=device) * 0.1
    w2_fp = torch.randn(E, N, K, dtype=torch.float32, device=device) * 0.1

    w13_list, w13s_list = [], []
    w2_list, w2s_list = [], []
    for e in range(E):
        q13, s13 = _quantize_sym_int8(w13_fp[e], group_size)
        w13_list.append(gptq_pack(q13, 8, K, 2 * N))
        w13s_list.append(s13)
        q2, s2 = _quantize_sym_int8(w2_fp[e], group_size)
        w2_list.append(gptq_pack(q2, 8, N, K))
        w2s_list.append(s2)

    return (
        torch.stack(w13_list),
        torch.stack(w2_list),
        torch.stack(w13s_list),
        torch.stack(w2s_list),
        w13_fp,
        w2_fp,
    )


# ---------------------------------------------------------------------------
# Tests: _supports_* static methods
# ---------------------------------------------------------------------------


def test_supports_current_device():
    assert TritonWNA16OTFExperts._supports_current_device()


@pytest.mark.parametrize(
    "wk,ak,expected",
    [
        (kInt4Static, None, True),
        (kInt4Static32, None, True),
        (kInt4StaticAsym, None, True),
        (kInt4Static32Asym, None, True),
        (kInt8Static, None, True),
        (None, None, False),  # unquantized not supported
        (kInt4Static, kInt4Static, False),  # W4A4 not supported
    ],
)
def test_supports_quant_scheme(wk, ak, expected):
    assert TritonWNA16OTFExperts._supports_quant_scheme(wk, ak) == expected


def test_supports_no_act_and_mul():
    assert TritonWNA16OTFExperts._supports_no_act_and_mul() is True


def test_supports_activation():
    assert TritonWNA16OTFExperts._supports_activation(MoEActivation.SILU) is True
    assert TritonWNA16OTFExperts._supports_activation(MoEActivation.GELU) is True
    assert TritonWNA16OTFExperts._supports_activation(MoEActivation.GELU_TANH) is True
    assert TritonWNA16OTFExperts._supports_activation(MoEActivation.SWIGLUOAI) is True


# ---------------------------------------------------------------------------
# Tests: _process_weights_triton_wna16 layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES_LIST)
def test_gptq_sym_weight_layout(E, K, N, group_size):
    """GPTQ sym: repacked w1/w2 have correct shape and dtype."""
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros, _, _ = _make_gptq_tensors(
        E, K, N, group_size, asym=False
    )

    result = _process_weights_triton_wna16(
        _FakeGPTQConfig(), w13, w2, w13_scale, w2_scale, None, None
    )
    w1_out, w2_out, s1_out, s2_out = result[0], result[1], result[2], result[3]

    # Kernel expects [E, 2*N, K//2] uint8 and [E, K, N//2] uint8
    assert w1_out.shape == (E, 2 * N, K // 2), f"w1 shape: {w1_out.shape}"
    assert w2_out.shape == (E, K, N // 2), f"w2 shape: {w2_out.shape}"
    assert w1_out.dtype == torch.uint8
    assert w2_out.dtype == torch.uint8
    # Scales transposed: [E, 2*N, K//gs] and [E, K, N//gs]
    assert s1_out.shape == (E, 2 * N, K // group_size)
    assert s2_out.shape == (E, K, N // group_size)
    # No zero points for symmetric
    assert result[8] is None
    assert result[9] is None


@pytest.mark.parametrize("E, K, N, group_size", SHAPES_LIST)
def test_gptq_asym_weight_layout(E, K, N, group_size):
    """GPTQ asym: repacked w1/w2 and zero points have correct shapes."""
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros, _, _ = _make_gptq_tensors(
        E, K, N, group_size, asym=True
    )

    result = _process_weights_triton_wna16(
        _FakeGPTQConfig(), w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros
    )
    w1_out, w2_out = result[0], result[1]
    w1_zp, w2_zp = result[8], result[9]

    assert w1_out.shape == (E, 2 * N, K // 2)
    assert w2_out.shape == (E, K, N // 2)
    # Zero points: [E, 2*N//2, K//gs] = [E, N, K//gs]
    assert w1_zp is not None and w1_zp.dtype == torch.uint8
    assert w2_zp is not None and w2_zp.dtype == torch.uint8
    assert w1_zp.shape == (E, N, K // group_size), f"w1_zp shape: {w1_zp.shape}"
    assert w2_zp.shape == (E, K // 2, N // group_size), f"w2_zp shape: {w2_zp.shape}"


# ---------------------------------------------------------------------------
# Tests: _process_weights_triton_wna16 layout (AWQ)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES_LIST)
def test_awq_sym_weight_layout(E, K, N, group_size):
    """AWQ sym: repacked w1/w2 have correct shape and dtype."""
    w13, w2, w13_scale, w2_scale, _, _, _, _ = _make_awq_tensors(
        E, K, N, group_size, asym=False
    )
    result = _process_weights_triton_wna16(
        _make_awq_config(group_size, zero_point=False),
        w13,
        w2,
        w13_scale,
        w2_scale,
        None,
        None,
    )
    w1_out, w2_out, s1_out, s2_out = result[0], result[1], result[2], result[3]

    assert w1_out.shape == (E, 2 * N, K // 2), f"w1 shape: {w1_out.shape}"
    assert w2_out.shape == (E, K, N // 2), f"w2 shape: {w2_out.shape}"
    assert w1_out.dtype == torch.uint8
    assert w2_out.dtype == torch.uint8
    assert s1_out.shape == (E, 2 * N, K // group_size)
    assert s2_out.shape == (E, K, N // group_size)
    assert result[8] is None
    assert result[9] is None


@pytest.mark.parametrize("E, K, N, group_size", SHAPES_LIST)
def test_awq_asym_weight_layout(E, K, N, group_size):
    """AWQ asym: repacked w1/w2 and zero points have correct shapes."""
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros, _, _ = _make_awq_tensors(
        E, K, N, group_size, asym=True
    )
    result = _process_weights_triton_wna16(
        _make_awq_config(group_size, zero_point=True),
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_qzeros,
        w2_qzeros,
    )
    w1_out, w2_out = result[0], result[1]
    w1_zp, w2_zp = result[8], result[9]

    assert w1_out.shape == (E, 2 * N, K // 2)
    assert w2_out.shape == (E, K, N // 2)
    assert w1_zp is not None and w1_zp.dtype == torch.uint8
    assert w2_zp is not None and w2_zp.dtype == torch.uint8
    assert w1_zp.shape == (E, N, K // group_size), f"w1_zp shape: {w1_zp.shape}"
    assert w2_zp.shape == (E, K // 2, N // group_size), f"w2_zp shape: {w2_zp.shape}"


# ---------------------------------------------------------------------------
# Tests: OTF forward matches emulation reference
# ---------------------------------------------------------------------------


def _run_otf_forward(
    experts,
    w1_out,
    w2_out,
    w1_scale,
    w2_scale,
    w1_zp,
    w2_zp,
    group_size,
    hidden_states,
    topk_weights,
    topk_ids,
    E,
    K,
    N,
):
    """Run TritonWNA16OTFExperts.apply() and return output."""
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
        w1=w1_out,
        w2=w2_out,
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


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS_LIST)
def test_otf_forward_matches_emulation_sym(E, K, N, top_k, group_size, num_tokens):
    """OTF sym forward matches Int4Emulation reference output."""
    w13, w2, w13_scale, w2_scale, _, _, w13_fp, w2_fp = _make_gptq_tensors(
        E, K, N, group_size, asym=False
    )

    # Emulation: dequantize at load, use BF16 weights
    emul_result = _process_weights_emulation_gptq(
        w13, w2, w13_scale, w2_scale, None, None, output_dtype=torch.bfloat16
    )
    w13_bf16, w2_bf16 = emul_result[0], emul_result[1]
    # w13_bf16: [E, 2N, K], w2_bf16: [E, K, N]

    # OTF: keep quantized, build quant config for kernel
    otf_result = _process_weights_triton_wna16(
        _FakeGPTQConfig(), w13, w2, w13_scale, w2_scale, None, None
    )
    w1_out, w2_out = otf_result[0], otf_result[1]
    s1_out, s2_out = otf_result[2], otf_result[3]

    moe_config = _make_moe_config(E, K, N)
    quant_cfg = int4_w4a16_moe_quant_config(
        w1_scale=s1_out,
        w2_scale=s2_out,
        block_shape=[0, group_size],
    )
    experts = TritonWNA16OTFExperts(moe_config, quant_cfg)

    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_otf = _run_otf_forward(
        experts,
        w1_out,
        w2_out,
        s1_out,
        s2_out,
        None,
        None,
        group_size,
        hidden_states,
        topk_weights,
        topk_ids,
        E,
        K,
        N,
    )

    # Reference: dense float32 forward using the dequantized BF16 weights.
    # This is the ground truth -- it avoids kernel-vs-kernel numerical differences
    # and tests only quantization error (which for int4 gs=32..128 should be small).
    ref = torch.zeros(num_tokens, K, dtype=torch.float32, device=device)
    # w13_bf16: [E, 2N, K], w2_bf16: [E, K, N]
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m].float() @ w13_bf16[e].float().T  # [2N]
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up  # [N]
            acc += w * (act @ w2_bf16[e].float().T)  # [K]
        ref[m] = acc

    rel_l2 = (
        torch.norm(out_otf.float() - ref) / torch.norm(ref).clamp(min=1e-6)
    ).item()
    # 1% threshold: reflects int4 quantization error only. Tighter than the
    # raw quantization noise bound (~0.5% for K=64) to catch numerical bugs.
    assert rel_l2 < 0.01, f"OTF vs float32 ref rel_l2={rel_l2:.4f} (threshold 0.01)"


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS_LIST)
def test_otf_forward_matches_emulation_asym_gptq(
    E, K, N, top_k, group_size, num_tokens
):
    """OTF asym GPTQ forward matches Int4Emulation dequant reference."""
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros, _, _ = _make_gptq_tensors(
        E, K, N, group_size, asym=True
    )

    emul_result = _process_weights_emulation_gptq(
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_qzeros,
        w2_qzeros,
        output_dtype=torch.bfloat16,
    )
    w13_bf16, w2_bf16 = emul_result[0], emul_result[1]

    otf_result = _process_weights_triton_wna16(
        _FakeGPTQConfig(), w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros
    )
    w1_out, w2_out = otf_result[0], otf_result[1]
    s1_out, s2_out = otf_result[2], otf_result[3]
    zp1_out, zp2_out = otf_result[8], otf_result[9]

    moe_config = _make_moe_config(E, K, N)
    quant_cfg = int4_w4a16_moe_quant_config(
        w1_scale=s1_out,
        w2_scale=s2_out,
        w1_zp=zp1_out,
        w2_zp=zp2_out,
        block_shape=[0, group_size],
    )
    experts = TritonWNA16OTFExperts(moe_config, quant_cfg)

    torch.manual_seed(2)
    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_otf = _run_otf_forward(
        experts,
        w1_out,
        w2_out,
        s1_out,
        s2_out,
        zp1_out,
        zp2_out,
        group_size,
        hidden_states,
        topk_weights,
        topk_ids,
        E,
        K,
        N,
    )

    ref = torch.zeros(num_tokens, K, dtype=torch.float32, device=device)
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m].float() @ w13_bf16[e].float().T
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up
            acc += w * (act @ w2_bf16[e].float().T)
        ref[m] = acc

    rel_l2 = (
        torch.norm(out_otf.float() - ref) / torch.norm(ref).clamp(min=1e-6)
    ).item()
    assert rel_l2 < 0.01, f"asym GPTQ OTF vs ref rel_l2={rel_l2:.4f} (threshold 0.01)"


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS_LIST)
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_otf_forward_matches_emulation_awq(
    E, K, N, top_k, group_size, num_tokens, asym
):
    """OTF AWQ forward (sym and asym) matches Int4Emulation dequant reference."""
    w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros, _, _ = _make_awq_tensors(
        E, K, N, group_size, asym=asym
    )

    emul_result = _process_weights_emulation_awq(
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_qzeros if asym else None,
        w2_qzeros if asym else None,
        output_dtype=torch.bfloat16,
    )
    w13_bf16, w2_bf16 = emul_result[0], emul_result[1]

    otf_result = _process_weights_triton_wna16(
        _make_awq_config(group_size, zero_point=asym),
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_qzeros if asym else None,
        w2_qzeros if asym else None,
    )
    w1_out, w2_out = otf_result[0], otf_result[1]
    s1_out, s2_out = otf_result[2], otf_result[3]
    zp1_out, zp2_out = otf_result[8], otf_result[9]

    moe_config = _make_moe_config(E, K, N)
    quant_cfg = int4_w4a16_moe_quant_config(
        w1_scale=s1_out,
        w2_scale=s2_out,
        w1_zp=zp1_out,
        w2_zp=zp2_out,
        block_shape=[0, group_size],
    )
    experts = TritonWNA16OTFExperts(moe_config, quant_cfg)

    torch.manual_seed(3)
    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_otf = _run_otf_forward(
        experts,
        w1_out,
        w2_out,
        s1_out,
        s2_out,
        zp1_out,
        zp2_out,
        group_size,
        hidden_states,
        topk_weights,
        topk_ids,
        E,
        K,
        N,
    )

    ref = torch.zeros(num_tokens, K, dtype=torch.float32, device=device)
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m].float() @ w13_bf16[e].float().T
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up
            acc += w * (act @ w2_bf16[e].float().T)
        ref[m] = acc

    rel_l2 = (
        torch.norm(out_otf.float() - ref) / torch.norm(ref).clamp(min=1e-6)
    ).item()
    label = "asym" if asym else "sym"
    assert rel_l2 < 0.01, f"AWQ {label} OTF vs ref rel_l2={rel_l2:.4f} (threshold 0.01)"


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS_LIST)
def test_otf_int8_forward_matches_emulation(E, K, N, top_k, group_size, num_tokens):
    """OTF int8 forward matches Int8Emulation dequant reference."""
    w13, w2, w13_scale, w2_scale, w13_fp, w2_fp = _make_gptq_int8_tensors(
        E, K, N, group_size
    )

    emul_result = _process_weights_emulation_int8(
        w13, w2, w13_scale, w2_scale, output_dtype=torch.bfloat16
    )
    w13_bf16, w2_bf16 = emul_result[0], emul_result[1]

    otf_result = _process_weights_triton_wna16(
        _FakeGPTQInt8Config(), w13, w2, w13_scale, w2_scale, None, None
    )
    w1_out, w2_out = otf_result[0], otf_result[1]
    s1_out, s2_out = otf_result[2], otf_result[3]

    from vllm.model_executor.layers.fused_moe.config import int8_w8a16_moe_quant_config

    moe_config = _make_moe_config(E, K, N)
    quant_cfg = int8_w8a16_moe_quant_config(
        w1_scale=s1_out, w2_scale=s2_out, block_shape=[0, group_size]
    )
    experts = TritonWNA16OTFExperts(moe_config, quant_cfg)

    torch.manual_seed(4)
    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    ws13 = torch.zeros(
        num_tokens * top_k * max(N, K), dtype=torch.bfloat16, device=device
    )
    ws2 = torch.zeros(
        num_tokens * top_k * max(2 * N, K), dtype=torch.bfloat16, device=device
    )
    out_otf = torch.zeros(num_tokens, K, dtype=torch.bfloat16, device=device)
    experts.apply(
        output=out_otf,
        hidden_states=hidden_states,
        w1=w1_out,
        w2=w2_out,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=ws13,
        workspace2=ws2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    # Reference: dense float32 forward using the dequantized BF16 weights.
    ref = torch.zeros(num_tokens, K, dtype=torch.float32, device=device)
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m].float() @ w13_bf16[e].float().T
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up
            acc += w * (act @ w2_bf16[e].float().T)
        ref[m] = acc

    rel_l2 = (
        torch.norm(out_otf.float() - ref) / torch.norm(ref).clamp(min=1e-6)
    ).item()
    # 1% threshold: int8 quantization is finer than int4; deviation from
    # the dequantized reference reflects only kernel arithmetic rounding.
    assert rel_l2 < 0.01, f"int8 OTF vs ref rel_l2={rel_l2:.4f} (threshold 0.01)"


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS_LIST)
def test_otf_ep_matches_full_forward(E, K, N, top_k, group_size, num_tokens):
    """OTF with expert_map (EP) produces same sum as full forward on local experts."""
    # Split E experts across 2 "ranks" -- test rank 0 (experts 0..E//2-1)
    num_local = E // 2
    expert_map = torch.full((E,), -1, dtype=torch.int32, device=device)
    expert_map[:num_local] = torch.arange(num_local, dtype=torch.int32, device=device)

    w13, w2, w13_scale, w2_scale, _, _, _, _ = _make_gptq_tensors(
        E, K, N, group_size, asym=False
    )
    otf_result = _process_weights_triton_wna16(
        _FakeGPTQConfig(), w13, w2, w13_scale, w2_scale, None, None
    )
    # Use only local experts' weights
    w1_local = otf_result[0][:num_local]
    w2_local = otf_result[1][:num_local]
    s1_local = otf_result[2][:num_local]
    s2_local = otf_result[3][:num_local]

    moe_config_local = FusedMoEConfig(
        num_experts=E,
        experts_per_token=top_k,
        hidden_dim=K,
        intermediate_size=N,
        num_local_experts=num_local,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )
    quant_cfg = int4_w4a16_moe_quant_config(
        w1_scale=s1_local, w2_scale=s2_local, block_shape=[0, group_size]
    )
    experts = TritonWNA16OTFExperts(moe_config_local, quant_cfg)

    torch.manual_seed(1)
    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    # Route all tokens to local experts only
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(num_local, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    ws13 = torch.zeros(
        num_tokens * top_k * max(N, K), dtype=torch.bfloat16, device=device
    )
    ws2 = torch.zeros(
        num_tokens * top_k * max(2 * N, K), dtype=torch.bfloat16, device=device
    )
    out_ep = torch.zeros(num_tokens, K, dtype=torch.bfloat16, device=device)
    experts.apply(
        output=out_ep,
        hidden_states=hidden_states,
        w1=w1_local,
        w2=w2_local,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=ws13,
        workspace2=ws2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    # Compute reference without expert_map (all experts local, same routing)
    moe_config_full = _make_moe_config(E, K, N)
    quant_cfg_full = int4_w4a16_moe_quant_config(
        w1_scale=otf_result[2], w2_scale=otf_result[3], block_shape=[0, group_size]
    )
    experts_full = TritonWNA16OTFExperts(moe_config_full, quant_cfg_full)
    out_full = _run_otf_forward(
        experts_full,
        otf_result[0],
        otf_result[1],
        otf_result[2],
        otf_result[3],
        None,
        None,
        group_size,
        hidden_states,
        topk_weights,
        topk_ids,
        E,
        K,
        N,
    )

    # Same kernel, same weights, same token routing -- output must be bit-identical
    # (or differ only by floating-point reordering in reduction, which is < 1 ULP).
    torch.testing.assert_close(
        out_ep.float(),
        out_full.float(),
        atol=1e-4,
        rtol=1e-4,
        msg="EP (expert_map) output differs from full forward"
        " -- same routing should be identical",
    )


def test_triton_wna16_otf_precedes_emulation_classes():
    """TritonWNA16OTFExperts must be the first kernel class under EMULATION."""
    from vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe import (
        Int4EmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.int8_emulation_moe import (
        Int8EmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonWNA16OTFExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        WNA16MoEBackend,
        backend_to_kernel_cls,
    )

    classes = backend_to_kernel_cls(WNA16MoEBackend.EMULATION)
    assert classes[0] is TritonWNA16OTFExperts, (
        f"TritonWNA16OTFExperts must be first in EMULATION kernel list,"
        f" got {classes[0]}"
    )
    assert Int4EmulationTritonExperts in classes
    assert Int8EmulationTritonExperts in classes
    idx_otf = classes.index(TritonWNA16OTFExperts)
    idx_int4 = classes.index(Int4EmulationTritonExperts)
    idx_int8 = classes.index(Int8EmulationTritonExperts)
    assert idx_otf < idx_int4, (
        f"TritonWNA16OTFExperts (idx={idx_otf}) must precede "
        f"Int4EmulationTritonExperts (idx={idx_int4})"
    )
    assert idx_otf < idx_int8, (
        f"TritonWNA16OTFExperts (idx={idx_otf}) must precede "
        f"Int8EmulationTritonExperts (idx={idx_int8})"
    )


# ---------------------------------------------------------------------------
# Tests: make_group_size_adjusted_weight_loader
# ---------------------------------------------------------------------------


def test_group_size_adjusted_loader_passthrough_when_factor_is_one():
    """factor=1 returns the original loader unchanged."""
    sentinel = object()

    def original(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        return sentinel

    wrapped = make_group_size_adjusted_weight_loader(original, group_size_div_factor=1)
    assert wrapped is original


def test_group_size_adjusted_loader_expands_scale_rows():
    """w2 scale/zero-point rows are expanded; w13 scales and weights are unchanged."""
    calls = []

    def capturing_loader(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        calls.append((weight_name, loaded_weight.shape))

    wrapped = make_group_size_adjusted_weight_loader(
        capturing_loader, group_size_div_factor=2
    )

    scale = torch.zeros(4, 128, device=device)  # 4 scale groups
    zeros = torch.zeros(4, 16, device=device)  # 4 zero-point groups
    weight = torch.zeros(16, 128, device=device)  # weight tensor

    w13_scale = torch.zeros(2, 256, device=device)  # w13 scale: not expanded

    wrapped(None, scale, "w2_scales", "w2", 0)
    wrapped(None, zeros, "w2_qzeros", "w2", 0)
    wrapped(None, weight, "w2_qweight", "w2", 0)
    wrapped(None, w13_scale, "w13_scales", "w1", 0)

    # w2 scale and zeros expanded to 8 rows; w2 weight and w13 scale unchanged
    assert calls[0] == ("w2_scales", (8, 128)), f"w2 scale not expanded: {calls[0]}"
    assert calls[1] == ("w2_qzeros", (8, 16)), f"w2 zeros not expanded: {calls[1]}"
    assert calls[2] == ("w2_qweight", (16, 128)), (
        f"w2 weight wrongly changed: {calls[2]}"
    )
    assert calls[3] == ("w13_scales", (2, 256)), (
        f"w13 scale wrongly expanded: {calls[3]}"
    )


# ---------------------------------------------------------------------------
# Tests: OTF forward when N % group_size != 0 (group-size adjustment path)
# ---------------------------------------------------------------------------
# Triggered by e.g. QuantTrio-Qwen3-235B-A22B-GPTQ-Int8 with tp=8:
#   moe_intermediate_size=1536, N_per_tp=192, group_size=128 -> 192%128=64!=0
#   adjusted_gs=64, factor=2


@pytest.mark.parametrize(
    "E, K, N, top_k, group_size, num_tokens",
    [
        # N=192, gs=128: 192%128=64, adjusted_gs=64, factor=2 (the real trigger case)
        pytest.param(4, 256, 192, 2, 128, 8, id="N192_gs128"),
        # N=384, gs=256: 384%256=128, adjusted_gs=128, factor=2
        pytest.param(4, 256, 384, 2, 256, 8, id="N384_gs256"),
    ],
)
def test_otf_forward_n_not_divisible_by_group_size(
    E, K, N, top_k, group_size, num_tokens
):
    """OTF forward is correct when N % group_size != 0.

    Simulates what make_group_size_adjusted_weight_loader produces:
    w13 scales are from the checkpoint at original group_size (K divides evenly).
    w2 weights are quantized at adjusted_gs; w2 scales come from the checkpoint
    (quantized at group_size, N//group_size rows) then repeat_interleave(factor).
    """
    adjusted_gs = group_size
    while N % adjusted_gs != 0:
        adjusted_gs //= 2
    div_factor = group_size // adjusted_gs

    torch.manual_seed(7)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device) * 0.1
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device) * 0.1

    w13_list, w13s_list, w2_list, w2s_list = [], [], [], []
    for e in range(E):
        # w13: quantize at original group_size (K is always divisible)
        q13, s13 = _quantize_sym_int4(w13_fp[e], group_size)
        w13_list.append(gptq_pack(q13, 4, K, 2 * N))
        w13s_list.append(s13)  # [K//group_size, 2N] -- no expansion needed
        # w2: weights quantized at adjusted_gs; scales simulated from the
        # checkpoint (group_size) then expanded by repeat_interleave(factor),
        # matching exactly what make_group_size_adjusted_weight_loader produces.
        q2, s2_fine = _quantize_sym_int4(w2_fp[e], adjusted_gs)
        w2_list.append(gptq_pack(q2, 4, N, K))
        _, s2_ckpt = _quantize_sym_int4(w2_fp[e], group_size)  # checkpoint rows
        s2_expanded = s2_ckpt.repeat_interleave(div_factor, dim=0)
        w2s_list.append(s2_expanded)  # [N//adjusted_gs, K]

    w13 = torch.stack(w13_list)
    w2 = torch.stack(w2_list)
    w13_scale = torch.stack(w13s_list)
    w2_scale = torch.stack(w2s_list)

    otf_result = _process_weights_triton_wna16(
        _FakeGPTQConfig(), w13, w2, w13_scale, w2_scale, None, None
    )
    w1_out, w2_out = otf_result[0], otf_result[1]
    s1_out, s2_out = otf_result[2], otf_result[3]

    # gs_eff derived from repacked scale shapes
    gs_eff = w2_out.shape[1] // s1_out.shape[2] if s1_out.shape[2] > 0 else adjusted_gs

    moe_config = _make_moe_config(E, K, N)
    quant_cfg = int4_w4a16_moe_quant_config(
        w1_scale=s1_out, w2_scale=s2_out, block_shape=[0, gs_eff]
    )
    experts = TritonWNA16OTFExperts(moe_config, quant_cfg)

    torch.manual_seed(8)
    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_otf = _run_otf_forward(
        experts,
        w1_out,
        w2_out,
        s1_out,
        s2_out,
        None,
        None,
        gs_eff,
        hidden_states,
        topk_weights,
        topk_ids,
        E,
        K,
        N,
    )

    # Reference: dequantize using the same expanded scales
    emul_result = _process_weights_emulation_gptq(
        w13, w2, w13_scale, w2_scale, None, None, output_dtype=torch.bfloat16
    )
    w13_bf16, w2_bf16 = emul_result[0], emul_result[1]

    ref = torch.zeros(num_tokens, K, dtype=torch.float32, device=device)
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m].float() @ w13_bf16[e].float().T
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up
            acc += w * (act @ w2_bf16[e].float().T)
        ref[m] = acc

    rel_l2 = (
        torch.norm(out_otf.float() - ref) / torch.norm(ref).clamp(min=1e-6)
    ).item()
    assert rel_l2 < 0.01, (
        f"[N={N}, gs={group_size}, adj_gs={adjusted_gs}] "
        f"rel_l2={rel_l2:.4f} (threshold 0.01)"
    )


# ---------------------------------------------------------------------------
# Note: OTF forward with N < group_size (e.g. N=64, group_size=128, TP=8)
# ---------------------------------------------------------------------------
# This configuration is unsupported.  When N_per_rank < group_size, the
# checkpoint w2 scale has N_full//group_size rows total; TP sharding gives
# each rank N_full//group_size//tp_size = 0 rows, so there is no data to
# expand via repeat_interleave.  AutoGPTQMoEMethod and AutoAWQMoEMethod raise
# a ValueError at create_weights time, which is the correct behavior.
# The correct remedy is to reduce --tensor-parallel-size so that
# N_per_rank >= group_size.
# Correctness of the weight loader wrapper for the N % group_size != 0
# (but N >= group_size) case is covered by test_group_size_adjusted_loader_*.
