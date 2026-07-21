#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Int8EmulationTritonExperts MoE backend.

Int8EmulationTritonExperts is a fallback under the EMULATION backend,
after TritonWNA16OTFExperts. It handles kInt8Static (GPTQ int8 symmetric)
only -- no AWQ, no asymmetric.
Weights are dequantized from packed int8 to BF16/FP16 once at load time.

Tests:
  - _unpack_and_dequant_int8_gptq: PyTorch and Triton paths vs reference
  - _process_weights_emulation_int8: output shapes and values
  - End-to-end forward: vs dequantized reference and vs original float weights
  - _supports_quant_scheme: int8 accepted, int4 and W4A4 rejected
  - Position within EMULATION backend: comes after TritonWNA16OTFExperts

Run `pytest tests/kernels/quantization/test_int8_emulation_moe.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.int8_emulation_moe import (
    Int8EmulationTritonExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _process_weights_emulation_int8,
    _unpack_and_dequant_int8_gptq,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    kInt4Static,
    kInt8Static,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Int8EmulationTritonExperts requires CUDA/ROCm.",
)

device = "cuda"

# (E, K, N, group_size)
SHAPES = [
    pytest.param(2, 64, 32, 32, id="tiny-gs32"),
    pytest.param(4, 128, 64, 64, id="small-gs64"),
    pytest.param(4, 256, 128, 128, id="medium-gs128"),
]

E2E_CONFIGS = [
    pytest.param(4, 64, 32, 2, 32, 8, id="tiny"),
    pytest.param(8, 128, 64, 2, 64, 16, id="small"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_sym_int8(w_fp: torch.Tensor, group_size: int):
    """Quantize [K, N] float to int8 symmetric (uint8b128).

    Returns:
        q:     [K, N] int32, values in [0, 255]  (uint8b128: val + 128)
        scale: [K//gs, N] float16
    """
    K, N = w_fp.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w_grouped = w_fp.reshape(n_groups, group_size, N)
    scale = w_grouped.abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-6)
    w_quant = (w_grouped / scale.unsqueeze(1)).round().clamp(-128, 127)
    q = (w_quant + 128).to(torch.int32).reshape(K, N)
    return q, scale.to(torch.float16)


def _dequantize_ref_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
):
    """Reference dequant for a single [K, N] int8 slice.

    Multiplies in float32 then casts, matching both the PyTorch fallback
    and the Triton kernel arithmetic exactly.
    """
    K, N = q.shape
    n_groups = scale.shape[0]
    group_size = K // n_groups
    w = q.reshape(n_groups, group_size, N).to(torch.float32)
    s = scale.unsqueeze(1).to(torch.float32)
    return ((w - 128) * s).to(output_dtype).reshape(K, N)


def _pack_int8_gptq(q: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Pack [K, N] int32 (uint8 values 0-255) into [K//4, N] int32."""
    return gptq_pack(q, 8, K, N)


def _make_moe_config(E, K, N, in_dtype=torch.bfloat16):
    return FusedMoEConfig(
        num_experts=E,
        experts_per_token=2,
        hidden_dim=K,
        intermediate_size=N,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=in_dtype,
        device=device,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )


def _make_int8_moe_weights(E, K, N, group_size, output_dtype=torch.bfloat16):
    """Build w13/w2 in GPTQ int8 MoE format and return float references."""
    torch.manual_seed(5)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

    w13_list, w13s_list, w13_ref_list = [], [], []
    w2_list, w2s_list, w2_ref_list = [], [], []

    for e in range(E):
        q13, s13 = _quantize_sym_int8(w13_fp[e], group_size)
        w13_list.append(_pack_int8_gptq(q13, K, 2 * N))
        w13s_list.append(s13)
        w13_ref_list.append(_dequantize_ref_int8(q13, s13, output_dtype))

        q2, s2 = _quantize_sym_int8(w2_fp[e], group_size)
        w2_list.append(_pack_int8_gptq(q2, N, K))
        w2s_list.append(s2)
        w2_ref_list.append(_dequantize_ref_int8(q2, s2, output_dtype))

    return (
        torch.stack(w13_list),
        torch.stack(w13s_list),
        torch.stack(w13_ref_list),  # [E, K, 2N]
        torch.stack(w2_list),
        torch.stack(w2s_list),
        torch.stack(w2_ref_list),  # [E, N, K]
    )


def _run_emulation_forward(
    experts, w13_out, w2_out, hidden_states, topk_weights, topk_ids, E, K, N
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
        w1=w13_out,
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


# ---------------------------------------------------------------------------
# Tests: _unpack_and_dequant_int8_gptq (PyTorch path vs reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_unpack_int8_symmetric_values(E, K, N, group_size):
    """PyTorch unpacker produces values matching the per-expert reference."""
    torch.manual_seed(0)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list, ref_list = [], [], []
    for e in range(E):
        q, s = _quantize_sym_int8(w_fp[e], group_size)
        packed_list.append(_pack_int8_gptq(q, K, N))
        scale_list.append(s)
        ref_list.append(_dequantize_ref_int8(q, s, torch.float32))

    w_packed = torch.stack(packed_list)
    scale = torch.stack(scale_list)
    ref = torch.stack(ref_list)

    out = _unpack_and_dequant_int8_gptq(
        w_packed,
        scale,
        transpose_output=False,
        output_dtype=torch.float32,
        force_torch=True,
    )

    assert out.shape == (E, K, N)
    assert torch.allclose(out, ref, atol=0), (
        f"max diff: {(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_unpack_int8_transpose_output(E, K, N, group_size):
    """transpose_output=True gives [E, N, K] with correct values."""
    torch.manual_seed(1)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list = [], []
    for e in range(E):
        q, s = _quantize_sym_int8(w_fp[e], group_size)
        packed_list.append(_pack_int8_gptq(q, K, N))
        scale_list.append(s)

    w_packed = torch.stack(packed_list)
    scale = torch.stack(scale_list)

    out_normal = _unpack_and_dequant_int8_gptq(
        w_packed, scale, False, torch.float32, force_torch=True
    )
    out_transposed = _unpack_and_dequant_int8_gptq(
        w_packed, scale, True, torch.float32, force_torch=True
    )

    assert out_transposed.shape == (E, N, K)
    assert torch.allclose(
        out_transposed, out_normal.permute(0, 2, 1).contiguous(), atol=0
    )


@pytest.mark.parametrize(
    "output_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_unpack_int8_output_dtype(output_dtype):
    """Output dtype is honoured."""
    E, K, N, gs = 2, 64, 32, 32
    torch.manual_seed(2)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)
    packed_list, scale_list = [], []
    for e in range(E):
        q, s = _quantize_sym_int8(w_fp[e], gs)
        packed_list.append(_pack_int8_gptq(q, K, N))
        scale_list.append(s)
    out = _unpack_and_dequant_int8_gptq(
        torch.stack(packed_list),
        torch.stack(scale_list),
        False,
        output_dtype,
        force_torch=True,
    )
    assert out.dtype == output_dtype


# ---------------------------------------------------------------------------
# Tests: Triton vs PyTorch agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
@pytest.mark.parametrize(
    "output_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_triton_matches_torch(E, K, N, group_size, output_dtype):
    """Triton and PyTorch unpackers produce bit-identical results."""
    torch.manual_seed(3)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list = [], []
    for e in range(E):
        q, s = _quantize_sym_int8(w_fp[e], group_size)
        packed_list.append(_pack_int8_gptq(q, K, N))
        scale_list.append(s)

    w_packed = torch.stack(packed_list)
    scale = torch.stack(scale_list)

    out_torch = _unpack_and_dequant_int8_gptq(
        w_packed, scale, False, output_dtype, force_torch=True
    )
    out_triton = _unpack_and_dequant_int8_gptq(
        w_packed, scale, False, output_dtype, force_torch=False
    )

    assert out_triton.shape == out_torch.shape
    assert out_triton.dtype == output_dtype
    assert torch.allclose(out_triton, out_torch, atol=0), (
        f"max diff: {(out_triton - out_torch).abs().max().item()}"
    )


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
def test_triton_transpose_matches_torch(E, K, N, group_size):
    """Triton transposed output matches PyTorch transposed output."""
    torch.manual_seed(4)
    w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

    packed_list, scale_list = [], []
    for e in range(E):
        q, s = _quantize_sym_int8(w_fp[e], group_size)
        packed_list.append(_pack_int8_gptq(q, K, N))
        scale_list.append(s)

    w_packed = torch.stack(packed_list)
    scale = torch.stack(scale_list)

    out_torch = _unpack_and_dequant_int8_gptq(
        w_packed, scale, True, torch.bfloat16, force_torch=True
    )
    out_triton = _unpack_and_dequant_int8_gptq(
        w_packed, scale, True, torch.bfloat16, force_torch=False
    )

    assert out_triton.shape == (E, N, K)
    assert torch.allclose(out_triton, out_torch, atol=0), (
        f"max diff: {(out_triton - out_torch).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Tests: _process_weights_emulation_int8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, group_size", SHAPES)
@pytest.mark.parametrize(
    "output_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_process_weights_int8_shapes_and_values(E, K, N, group_size, output_dtype):
    """Output shapes match TritonExperts layout and values match reference."""
    w13, w13s, w13_ref, w2, w2s, w2_ref = _make_int8_moe_weights(
        E, K, N, group_size, output_dtype
    )

    result = _process_weights_emulation_int8(
        w13, w2, w13s, w2s, output_dtype=output_dtype
    )
    w13_out, w2_out = result[0], result[1]

    assert w13_out.shape == (E, 2 * N, K), f"w13 shape: {w13_out.shape}"
    assert w2_out.shape == (E, K, N), f"w2 shape: {w2_out.shape}"
    assert w13_out.dtype == output_dtype
    assert w2_out.dtype == output_dtype

    expected_w13 = w13_ref.permute(0, 2, 1)
    expected_w2 = w2_ref.permute(0, 2, 1)

    assert torch.allclose(w13_out, expected_w13, atol=1e-2), (
        f"w13 max diff: {(w13_out - expected_w13).abs().max().item()}"
    )
    assert torch.allclose(w2_out, expected_w2, atol=1e-2), (
        f"w2 max diff: {(w2_out - expected_w2).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# End-to-end MoE forward pass tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS)
@pytest.mark.parametrize(
    "model_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"]
)
def test_int8_emulation_forward_close_to_reference(
    E, K, N, top_k, group_size, num_tokens, model_dtype
):
    """Emulation output is close to a direct float dense MoE forward."""
    torch.manual_seed(10)
    moe_config = _make_moe_config(E, K, N, model_dtype)

    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device) * 0.02
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device) * 0.02

    packed_list13, scale_list13 = [], []
    packed_list2, scale_list2 = [], []
    for e in range(E):
        q13, s13 = _quantize_sym_int8(w13_fp[e], group_size)
        packed_list13.append(_pack_int8_gptq(q13, K, 2 * N))
        scale_list13.append(s13)
        q2, s2 = _quantize_sym_int8(w2_fp[e], group_size)
        packed_list2.append(_pack_int8_gptq(q2, N, K))
        scale_list2.append(s2)

    res = _process_weights_emulation_int8(
        torch.stack(packed_list13),
        torch.stack(packed_list2),
        torch.stack(scale_list13),
        torch.stack(scale_list2),
        output_dtype=model_dtype,
    )
    w13_out, w2_out = res[0], res[1]

    # Int8EmulationTritonExperts nulls out scale fields in __init__ since
    # weights are already dequantized before apply(); dummy value is fine.
    dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
    quant_cfg = int8_w8a16_moe_quant_config(dummy_scale, dummy_scale)
    experts = Int8EmulationTritonExperts(moe_config, quant_cfg)

    hidden_states = torch.randn(num_tokens, K, dtype=model_dtype, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device),
        dim=-1,
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_emulation = _run_emulation_forward(
        experts,
        w13_out,
        w2_out,
        hidden_states,
        topk_weights,
        topk_ids,
        E,
        K,
        N,
    )

    # Dense reference using the dequantized BF16 weights directly
    ref = torch.zeros(num_tokens, K, dtype=model_dtype, device=device)
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m] @ w13_out[e].T
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up
            acc += w * (act @ w2_out[e].T).float()
        ref[m] = acc.to(model_dtype)

    rel_l2 = (
        torch.norm(out_emulation.float() - ref.float())
        / torch.norm(ref.float()).clamp(min=1e-6)
    ).item()
    assert rel_l2 < 0.01, f"[{model_dtype}] relative L2 = {rel_l2:.4f} (threshold 0.01)"


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS)
def test_int8_emulation_fp16_bf16_agree(E, K, N, top_k, group_size, num_tokens):
    """FP16 and BF16 emulation produce close outputs for the same weights."""
    torch.manual_seed(11)

    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device) * 0.02
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device) * 0.02

    packed_list13, scale_list13 = [], []
    packed_list2, scale_list2 = [], []
    for e in range(E):
        q13, s13 = _quantize_sym_int8(w13_fp[e], group_size)
        packed_list13.append(_pack_int8_gptq(q13, K, 2 * N))
        scale_list13.append(s13)
        q2, s2 = _quantize_sym_int8(w2_fp[e], group_size)
        packed_list2.append(_pack_int8_gptq(q2, N, K))
        scale_list2.append(s2)

    w13_packed = torch.stack(packed_list13)
    w13_scales = torch.stack(scale_list13)
    w2_packed = torch.stack(packed_list2)
    w2_scales = torch.stack(scale_list2)

    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device),
        dim=-1,
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    # Sample hidden_states once in float32, cast to each dtype -- both runs
    # see the same input values so the diff measures only dtype arithmetic.
    hidden_states_f32 = torch.randn(num_tokens, K, dtype=torch.float32, device=device)

    outputs = {}
    for dtype in (torch.bfloat16, torch.float16):
        res = _process_weights_emulation_int8(
            w13_packed,
            w2_packed,
            w13_scales,
            w2_scales,
            output_dtype=dtype,
        )
        # dummy scale: nulled out by Int8EmulationTritonExperts.__init__
        dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
        quant_cfg = int8_w8a16_moe_quant_config(dummy_scale, dummy_scale)
        experts = Int8EmulationTritonExperts(
            _make_moe_config(E, K, N, dtype), quant_cfg
        )
        outputs[dtype] = _run_emulation_forward(
            experts,
            res[0],
            res[1],
            hidden_states_f32.to(dtype),
            topk_weights,
            topk_ids,
            E,
            K,
            N,
        ).float()

    rel_l2 = (
        torch.norm(outputs[torch.bfloat16] - outputs[torch.float16])
        / torch.norm(outputs[torch.bfloat16]).clamp(min=1e-6)
    ).item()
    assert rel_l2 < 0.01, f"bf16 vs fp16 relative L2 = {rel_l2:.4f} (threshold 0.01)"


@pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS)
def test_int8_emulation_output_close_to_float_weights(
    E, K, N, top_k, group_size, num_tokens
):
    """Emulation output is close to a forward using the original float weights."""
    torch.manual_seed(15)
    moe_config = _make_moe_config(E, K, N)

    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float32, device=device) * 0.02
    w2_fp = torch.randn(E, N, K, dtype=torch.float32, device=device) * 0.02

    packed_list13, scale_list13 = [], []
    packed_list2, scale_list2 = [], []
    for e in range(E):
        q13, s13 = _quantize_sym_int8(w13_fp[e], group_size)
        packed_list13.append(_pack_int8_gptq(q13, K, 2 * N))
        scale_list13.append(s13)
        q2, s2 = _quantize_sym_int8(w2_fp[e], group_size)
        packed_list2.append(_pack_int8_gptq(q2, N, K))
        scale_list2.append(s2)

    res = _process_weights_emulation_int8(
        torch.stack(packed_list13),
        torch.stack(packed_list2),
        torch.stack(scale_list13),
        torch.stack(scale_list2),
        output_dtype=torch.bfloat16,
    )
    w13_out, w2_out = res[0], res[1]

    # Int8EmulationTritonExperts nulls out scale fields; dummy value is fine.
    dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
    quant_cfg = int8_w8a16_moe_quant_config(dummy_scale, dummy_scale)
    experts = Int8EmulationTritonExperts(moe_config, quant_cfg)

    hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    out_emulation = _run_emulation_forward(
        experts, w13_out, w2_out, hidden_states, topk_weights, topk_ids, E, K, N
    )

    # Reference: dense forward with original float32 weights (no quantization).
    ref = torch.zeros(num_tokens, K, dtype=torch.float32, device=device)
    for m in range(num_tokens):
        acc = torch.zeros(K, dtype=torch.float32, device=device)
        for k in range(top_k):
            e = topk_ids[m, k].item()
            w = topk_weights[m, k].item()
            gate_up = hidden_states[m].float() @ w13_fp[e]  # [K] @ [K, 2N] = [2N]
            gate, up = gate_up.chunk(2)
            act = F.silu(gate) * up
            acc += w * (act @ w2_fp[e])  # [N] @ [N, K] = [K]
        ref[m] = acc

    rel_l2 = (
        torch.norm(out_emulation.float() - ref) / torch.norm(ref).clamp(min=1e-6)
    ).item()
    # 5% threshold: int8 quantization error is ~0.4% per layer; two GEMM layers
    # accumulate to ~1-2% rel_l2 at group_size=32..128 for these small sizes.
    assert rel_l2 < 0.05, (
        f"int8 emulation vs float weights rel_l2={rel_l2:.4f} (threshold 0.05)"
    )


# ---------------------------------------------------------------------------
# Tests: _supports_quant_scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wk,ak,expected",
    [
        (kInt8Static, None, True),  # int8 GPTQ symmetric: supported
        (kInt4Static, None, False),  # int4: handled by OTF or Int4Emulation, not here
        (kInt8Static, kInt8Static, False),  # W8A8: not supported (W-only)
        (None, None, False),  # unquantized: not supported
    ],
)
def test_supports_quant_scheme(wk, ak, expected):
    assert Int8EmulationTritonExperts._supports_quant_scheme(wk, ak) == expected


# ---------------------------------------------------------------------------
# Tests: position within EMULATION backend
# ---------------------------------------------------------------------------


def test_int8_emulation_position_in_emulation_backend():
    """Int8EmulationTritonExperts must come after TritonWNA16OTFExperts in EMULATION."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonWNA16OTFExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        WNA16MoEBackend,
        backend_to_kernel_cls,
    )

    classes = backend_to_kernel_cls(WNA16MoEBackend.EMULATION)
    assert Int8EmulationTritonExperts in classes
    idx_int8 = classes.index(Int8EmulationTritonExperts)
    idx_otf = classes.index(TritonWNA16OTFExperts)
    assert idx_otf < idx_int8, (
        f"TritonWNA16OTFExperts (idx={idx_otf}) must precede "
        f"Int8EmulationTritonExperts (idx={idx_int8})"
    )
