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

# ---------------------------------------------------------------------------
# Shape parameters
# ---------------------------------------------------------------------------

# (E, K, N, group_size)
# K = hidden_size, N = intermediate_size
SHAPES = [
    pytest.param(2, 64, 32, 32, id="tiny-gs32"),
    pytest.param(4, 128, 64, 64, id="small-gs64"),
    pytest.param(4, 256, 128, 128, id="medium-gs128"),
]


# ---------------------------------------------------------------------------
# Helpers: quantize / pack for GPTQ and AWQ
# ---------------------------------------------------------------------------


def _quantize_sym(w_fp: torch.Tensor, group_size: int):
    """Quantize [K, N] float16 to int4 symmetric (uint4b8), return q_w and scale."""
    K, N = w_fp.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w_grouped = w_fp.reshape(n_groups, group_size, N)
    scale = w_grouped.abs().amax(dim=1) / 7.0  # [n_groups, N]
    scale = scale.clamp(min=1e-6)
    w_quant = (w_grouped / scale.unsqueeze(1)).round().clamp(-8, 7)  # int4 range
    # Store as uint4b8: shift by +8 so values are in [0, 15]
    w_uint = (w_quant + 8).to(torch.int32)  # [n_groups, gs, N], values 0..15
    return w_uint.reshape(K, N), scale  # [K, N] int32, [n_groups, N] float16


def _quantize_asym(w_fp: torch.Tensor, group_size: int):
    """Quantize [K, N] float16 to uint4 asymmetric, return q_w, scale, and zero."""
    K, N = w_fp.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w_grouped = w_fp.reshape(n_groups, group_size, N)
    wmin = w_grouped.amin(dim=1)  # [n_groups, N]
    wmax = w_grouped.amax(dim=1)
    scale = (wmax - wmin) / 15.0  # [n_groups, N]
    scale = scale.clamp(min=1e-6)
    zero = (-wmin / scale).round().clamp(0, 15).to(torch.int32)  # [n_groups, N]
    w_quant = ((w_grouped - wmin.unsqueeze(1)) / scale.unsqueeze(1)).round()
    w_quant = w_quant.clamp(0, 15).to(torch.int32)  # [n_groups, gs, N]
    return w_quant.reshape(K, N), scale, zero  # [K,N], [ng,N] fp16, [ng,N] int32


def _dequantize_ref(
    w_uint: torch.Tensor,
    scale: torch.Tensor,
    zero=None,
    output_dtype: torch.dtype = torch.bfloat16,
):
    """Reference dequant for a single [K, N] slice.

    output_dtype must match the dtype passed to the unpacker so that both
    sides of the comparison perform arithmetic in the same precision.
    """
    K, N = w_uint.shape
    n_groups = scale.shape[0]
    group_size = K // n_groups
    w = w_uint.reshape(n_groups, group_size, N).to(output_dtype)
    s = scale.unsqueeze(1).to(output_dtype)  # [ng, 1, N]
    if zero is None:
        out = (w - 8) * s
    else:
        z = zero.unsqueeze(1).to(output_dtype)
        out = (w - z) * s
    return out.reshape(K, N)


def _pack_gptq_zeros(zero: torch.Tensor, N: int):
    """Pack asymmetric zeros [n_groups, N] into GPTQ format [n_groups, N//8] int32."""
    n_groups, _ = zero.shape
    z = zero.to(torch.int32).cpu().numpy().astype(numpy.uint32)
    packed = numpy.zeros((n_groups, N // 8), dtype=numpy.uint32)
    for i in range(8):
        packed |= z[:, i::8] << (i * 4)
    return torch.from_numpy(packed.astype(numpy.int32)).to(device)


def _pack_awq_zeros(zero: torch.Tensor, N: int):
    """Pack asymmetric zeros [n_groups, N] into AWQ column format [n_groups, N//8]."""
    # AWQ zero packing mirrors awq_pack: interleave then pack along N
    n_groups, _ = zero.shape
    interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    z = zero.to(torch.int32).cpu().numpy().astype(numpy.uint32)
    z_interleaved = z.reshape(-1, 8)[:, interleave].reshape(n_groups, N)
    packed = numpy.zeros((n_groups, N // 8), dtype=numpy.uint32)
    for i in range(8):
        packed |= z_interleaved[:, i::8] << (i * 4)
    return torch.from_numpy(packed.astype(numpy.int32)).to(device)


# ---------------------------------------------------------------------------
# Unit tests: _unpack_and_dequant_int4_gptq
# ---------------------------------------------------------------------------


class TestUnpackDequantGPTQ:
    """Test the GPTQ unpacker against a numpy reference."""

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    def test_symmetric(self, E, K, N, group_size):
        """GPTQ symmetric: uint4b8 bias, no zero-points."""
        torch.manual_seed(0)
        w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

        # Quantize per expert and pack
        packed_list, scale_list = [], []
        ref_list = []
        for e in range(E):
            q, s = _quantize_sym(w_fp[e], group_size)  # [K,N], [K//gs,N]
            packed_list.append(gptq_pack(q, 4, K, N))  # [K//8, N] int32
            scale_list.append(s)
            ref_list.append(_dequantize_ref(q, s, output_dtype=torch.float32))

        w_packed = torch.stack(packed_list).to(device)  # [E, K//8, N]
        scale = torch.stack(scale_list).to(device)  # [E, K//gs, N]
        ref = torch.stack(ref_list).to(device)  # [E, K, N]

        # transpose_output=False → [E, K, N]
        out = _unpack_and_dequant_int4_gptq(
            w_packed, scale, None, transpose_output=False, output_dtype=torch.float32
        )

        assert out.shape == (E, K, N)
        assert torch.allclose(out, ref, atol=0), (
            f"max diff: {(out - ref).abs().max().item()}"
        )

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    def test_asymmetric(self, E, K, N, group_size):
        """GPTQ asymmetric: packed zero-points in [E, K//gs, N//8]."""
        torch.manual_seed(1)
        w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

        packed_list, scale_list, zero_packed_list, ref_list = [], [], [], []
        for e in range(E):
            q, s, z = _quantize_asym(w_fp[e], group_size)
            packed_list.append(gptq_pack(q, 4, K, N))
            scale_list.append(s)
            zero_packed_list.append(_pack_gptq_zeros(z, N))
            ref_list.append(_dequantize_ref(q, s, z, output_dtype=torch.float32))

        w_packed = torch.stack(packed_list).to(device)
        scale = torch.stack(scale_list).to(device)
        qzeros = torch.stack(zero_packed_list).to(device)
        ref = torch.stack(ref_list).to(device)

        out = _unpack_and_dequant_int4_gptq(
            w_packed, scale, qzeros, transpose_output=False, output_dtype=torch.float32
        )

        assert out.shape == (E, K, N)
        assert torch.allclose(out, ref, atol=0), (
            f"max diff: {(out - ref).abs().max().item()}"
        )

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    def test_transpose_output(self, E, K, N, group_size):
        """transpose_output=True should give [E, N, K]."""
        torch.manual_seed(2)
        w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

        packed_list, scale_list = [], []
        for e in range(E):
            q, s = _quantize_sym(w_fp[e], group_size)
            packed_list.append(gptq_pack(q, 4, K, N))
            scale_list.append(s)

        w_packed = torch.stack(packed_list).to(device)
        scale = torch.stack(scale_list).to(device)

        out_normal = _unpack_and_dequant_int4_gptq(
            w_packed, scale, None, transpose_output=False
        )
        out_transposed = _unpack_and_dequant_int4_gptq(
            w_packed, scale, None, transpose_output=True
        )

        assert out_transposed.shape == (E, N, K)
        assert torch.allclose(
            out_transposed, out_normal.permute(0, 2, 1).contiguous(), atol=0
        )


# ---------------------------------------------------------------------------
# Unit tests: _unpack_and_dequant_int4_awq
# ---------------------------------------------------------------------------


class TestUnpackDequantAWQ:
    """Test the AWQ unpacker against the same numpy reference."""

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    def test_symmetric(self, E, K, N, group_size):
        """AWQ symmetric: uint4b8 bias, packed along N."""
        torch.manual_seed(3)
        w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

        packed_list, scale_list, ref_list = [], [], []
        for e in range(E):
            q, s = _quantize_sym(w_fp[e], group_size)  # [K, N]
            packed_list.append(awq_pack(q, 4, K, N))  # [K, N//8]
            scale_list.append(s)  # [K//gs, N]
            ref_list.append(_dequantize_ref(q, s, output_dtype=torch.float32))

        w_packed = torch.stack(packed_list).to(device)  # [E, K, N//8]
        scale = torch.stack(scale_list).to(device)  # [E, K//gs, N]
        ref = torch.stack(ref_list).to(device)  # [E, K, N]

        out = _unpack_and_dequant_int4_awq(
            w_packed, scale, None, transpose_output=False, output_dtype=torch.float32
        )

        assert out.shape == (E, K, N)
        assert torch.allclose(out, ref, atol=0), (
            f"max diff: {(out - ref).abs().max().item()}"
        )

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    def test_asymmetric(self, E, K, N, group_size):
        """AWQ asymmetric: packed zero-points in [E, K//gs, N//8]."""
        torch.manual_seed(4)
        w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

        packed_list, scale_list, zero_packed_list, ref_list = [], [], [], []
        for e in range(E):
            q, s, z = _quantize_asym(w_fp[e], group_size)
            packed_list.append(awq_pack(q, 4, K, N))
            scale_list.append(s)
            zero_packed_list.append(_pack_awq_zeros(z, N))
            ref_list.append(_dequantize_ref(q, s, z, output_dtype=torch.float32))

        w_packed = torch.stack(packed_list).to(device)
        scale = torch.stack(scale_list).to(device)
        qzeros = torch.stack(zero_packed_list).to(device)
        ref = torch.stack(ref_list).to(device)

        out = _unpack_and_dequant_int4_awq(
            w_packed, scale, qzeros, transpose_output=False, output_dtype=torch.float32
        )

        assert out.shape == (E, K, N)
        assert torch.allclose(out, ref, atol=0), (
            f"max diff: {(out - ref).abs().max().item()}"
        )

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    def test_transpose_output(self, E, K, N, group_size):
        """transpose_output=True should give [E, N, K]."""
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
    def test_awq_gptq_agree_on_same_weights(self, E, K, N, group_size):
        """AWQ and GPTQ unpackers must produce identical float values
        when given the same original quantized weights packed in their
        respective formats."""
        torch.manual_seed(6)
        w_fp = torch.randn(E, K, N, dtype=torch.float16, device=device)

        gptq_packed_list, awq_packed_list, scale_list = [], [], []
        for e in range(E):
            q, s = _quantize_sym(w_fp[e], group_size)
            gptq_packed_list.append(gptq_pack(q, 4, K, N))
            awq_packed_list.append(awq_pack(q, 4, K, N))
            scale_list.append(s)

        gptq_w = torch.stack(gptq_packed_list).to(device)
        awq_w = torch.stack(awq_packed_list).to(device)
        scale = torch.stack(scale_list).to(device)

        out_gptq = _unpack_and_dequant_int4_gptq(
            gptq_w, scale, None, False, torch.float32
        )
        out_awq = _unpack_and_dequant_int4_awq(awq_w, scale, None, False, torch.float32)

        assert torch.allclose(out_gptq, out_awq, atol=0), (
            f"max diff: {(out_gptq - out_awq).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Unit tests: _process_weights_emulation_{gptq,awq}
# ---------------------------------------------------------------------------


def _make_gptq_moe_weights(E, K, N, group_size, asym=False):
    """Build w13/w2 in GPTQ MoE format and return the float references."""
    # w13: gate+up stacked → [E, K//8, 2N], scale [E, K//gs, 2N]
    # w2:  down projection → [E, N//8, K],  scale [E, N//gs, K]
    torch.manual_seed(7)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

    w13_list, w13_scale_list, w13_zero_list, w13_ref_list = [], [], [], []
    w2_list, w2_scale_list, w2_zero_list, w2_ref_list = [], [], [], []

    for e in range(E):
        if asym:
            q13, s13, z13 = _quantize_asym(w13_fp[e], group_size)
            w13_list.append(gptq_pack(q13, 4, K, 2 * N))
            w13_scale_list.append(s13)
            w13_zero_list.append(_pack_gptq_zeros(z13, 2 * N))
            w13_ref_list.append(_dequantize_ref(q13, s13, z13))

            q2, s2, z2 = _quantize_asym(w2_fp[e], group_size)
            w2_list.append(gptq_pack(q2, 4, N, K))
            w2_scale_list.append(s2)
            w2_zero_list.append(_pack_gptq_zeros(z2, K))
            w2_ref_list.append(_dequantize_ref(q2, s2, z2))
        else:
            q13, s13 = _quantize_sym(w13_fp[e], group_size)
            w13_list.append(gptq_pack(q13, 4, K, 2 * N))
            w13_scale_list.append(s13)
            w13_zero_list.append(None)
            w13_ref_list.append(_dequantize_ref(q13, s13))

            q2, s2 = _quantize_sym(w2_fp[e], group_size)
            w2_list.append(gptq_pack(q2, 4, N, K))
            w2_scale_list.append(s2)
            w2_zero_list.append(None)
            w2_ref_list.append(_dequantize_ref(q2, s2))

    w13 = torch.stack(w13_list)  # [E, K//8, 2N]
    w13_scale = torch.stack(w13_scale_list)
    w13_qzeros = torch.stack(w13_zero_list) if asym else None
    w13_ref = torch.stack(w13_ref_list)  # [E, K, 2N] float32

    w2 = torch.stack(w2_list)  # [E, N//8, K]
    w2_scale = torch.stack(w2_scale_list)
    w2_qzeros = torch.stack(w2_zero_list) if asym else None
    w2_ref = torch.stack(w2_ref_list)  # [E, N, K] float32

    return (
        w13,
        w13_scale,
        w13_qzeros,
        w13_ref,
        w2,
        w2_scale,
        w2_qzeros,
        w2_ref,
    )


def _make_awq_moe_weights(E, K, N, group_size, asym=False):
    """Build w13/w2 in AWQ MoE format and return the float references.

    AWQ shapes:
      w13: [E, K, 2N//8]    scale: [E, K//gs, 2N]  qzeros: [E, K//gs, 2N//8]
      w2:  [E, N, K//8]     scale: [E, N//gs, K]   qzeros: [E, N//gs, K//8]
    """
    torch.manual_seed(8)
    w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
    w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

    w13_list, w13_scale_list, w13_zero_list, w13_ref_list = [], [], [], []
    w2_list, w2_scale_list, w2_zero_list, w2_ref_list = [], [], [], []

    for e in range(E):
        if asym:
            q13, s13, z13 = _quantize_asym(w13_fp[e], group_size)
            w13_list.append(awq_pack(q13, 4, K, 2 * N))
            w13_scale_list.append(s13)
            w13_zero_list.append(_pack_awq_zeros(z13, 2 * N))
            w13_ref_list.append(_dequantize_ref(q13, s13, z13))

            q2, s2, z2 = _quantize_asym(w2_fp[e], group_size)
            w2_list.append(awq_pack(q2, 4, N, K))
            w2_scale_list.append(s2)
            w2_zero_list.append(_pack_awq_zeros(z2, K))
            w2_ref_list.append(_dequantize_ref(q2, s2, z2))
        else:
            q13, s13 = _quantize_sym(w13_fp[e], group_size)
            w13_list.append(awq_pack(q13, 4, K, 2 * N))
            w13_scale_list.append(s13)
            w13_zero_list.append(None)
            w13_ref_list.append(_dequantize_ref(q13, s13))

            q2, s2 = _quantize_sym(w2_fp[e], group_size)
            w2_list.append(awq_pack(q2, 4, N, K))
            w2_scale_list.append(s2)
            w2_zero_list.append(None)
            w2_ref_list.append(_dequantize_ref(q2, s2))

    w13 = torch.stack(w13_list)
    w13_scale = torch.stack(w13_scale_list)
    w13_qzeros = torch.stack(w13_zero_list) if asym else None
    w13_ref = torch.stack(w13_ref_list)  # [E, K, 2N]

    w2 = torch.stack(w2_list)
    w2_scale = torch.stack(w2_scale_list)
    w2_qzeros = torch.stack(w2_zero_list) if asym else None
    w2_ref = torch.stack(w2_ref_list)  # [E, N, K]

    return (
        w13,
        w13_scale,
        w13_qzeros,
        w13_ref,
        w2,
        w2_scale,
        w2_qzeros,
        w2_ref,
    )


class TestProcessWeightsEmulation:
    """Test _process_weights_emulation_{gptq,awq}: output shapes and values."""

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    @pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
    def test_gptq_output_shapes_and_values(self, E, K, N, group_size, asym):
        (
            w13,
            w13_scale,
            w13_qzeros,
            w13_ref,
            w2,
            w2_scale,
            w2_qzeros,
            w2_ref,
        ) = _make_gptq_moe_weights(E, K, N, group_size, asym)

        result = _process_weights_emulation_gptq(
            w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros
        )
        w13_out, w2_out = result[0], result[1]

        # Shape checks
        assert w13_out.shape == (E, 2 * N, K), f"w13 shape: {w13_out.shape}"
        assert w2_out.shape == (E, K, N), f"w2 shape: {w2_out.shape}"
        assert w13_out.dtype == torch.bfloat16
        assert w2_out.dtype == torch.bfloat16

        # Value checks: compare to per-expert reference.
        # _dequantize_ref now uses bfloat16 arithmetic to match the unpacker
        # exactly, so we compare in float32 with zero tolerance on the
        # bfloat16 grid (any diff is a real logic error, not rounding noise).
        # w13_ref: [E, K, 2N] → expected out: [E, 2N, K]
        expected_w13 = w13_ref.permute(0, 2, 1)
        # w2_ref: [E, N, K] → expected out: [E, K, N]
        expected_w2 = w2_ref.permute(0, 2, 1)

        assert torch.allclose(w13_out.float(), expected_w13.float(), atol=0), (
            "w13 max diff: "
            f"{(w13_out.float() - expected_w13.float()).abs().max().item()}"
        )
        assert torch.allclose(w2_out.float(), expected_w2.float(), atol=0), (
            f"w2 max diff: {(w2_out.float() - expected_w2.float()).abs().max().item()}"
        )

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    @pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
    def test_awq_output_shapes_and_values(self, E, K, N, group_size, asym):
        (
            w13,
            w13_scale,
            w13_qzeros,
            w13_ref,
            w2,
            w2_scale,
            w2_qzeros,
            w2_ref,
        ) = _make_awq_moe_weights(E, K, N, group_size, asym)

        result = _process_weights_emulation_awq(
            w13, w2, w13_scale, w2_scale, w13_qzeros, w2_qzeros
        )
        w13_out, w2_out = result[0], result[1]

        assert w13_out.shape == (E, 2 * N, K), f"w13 shape: {w13_out.shape}"
        assert w2_out.shape == (E, K, N), f"w2 shape: {w2_out.shape}"
        assert w13_out.dtype == torch.bfloat16
        assert w2_out.dtype == torch.bfloat16

        expected_w13 = w13_ref.permute(0, 2, 1)
        expected_w2 = w2_ref.permute(0, 2, 1)

        assert torch.allclose(w13_out.float(), expected_w13.float(), atol=0), (
            "w13 max diff: "
            f"{(w13_out.float() - expected_w13.float()).abs().max().item()}"
        )
        assert torch.allclose(w2_out.float(), expected_w2.float(), atol=0), (
            f"w2 max diff: {(w2_out.float() - expected_w2.float()).abs().max().item()}"
        )

    @pytest.mark.parametrize("E, K, N, group_size", SHAPES)
    @pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
    def test_awq_gptq_agree(self, E, K, N, group_size, asym):
        """AWQ and GPTQ emulation paths must agree when given the same weights."""
        torch.manual_seed(9)
        w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
        w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

        g13_list, g13s_list, g13z_list = [], [], []
        a13_list, a13s_list, a13z_list = [], [], []
        g2_list, g2s_list, g2z_list = [], [], []
        a2_list, a2s_list, a2z_list = [], [], []

        for e in range(E):
            if asym:
                q13, s13, z13 = _quantize_asym(w13_fp[e], group_size)
                q2, s2, z2 = _quantize_asym(w2_fp[e], group_size)
                g13z_list.append(_pack_gptq_zeros(z13, 2 * N))
                a13z_list.append(_pack_awq_zeros(z13, 2 * N))
                g2z_list.append(_pack_gptq_zeros(z2, K))
                a2z_list.append(_pack_awq_zeros(z2, K))
            else:
                q13, s13 = _quantize_sym(w13_fp[e], group_size)
                q2, s2 = _quantize_sym(w2_fp[e], group_size)
                g13z_list.append(None)
                a13z_list.append(None)
                g2z_list.append(None)
                a2z_list.append(None)

            g13_list.append(gptq_pack(q13, 4, K, 2 * N))
            a13_list.append(awq_pack(q13, 4, K, 2 * N))
            g13s_list.append(s13)
            a13s_list.append(s13)

            g2_list.append(gptq_pack(q2, 4, N, K))
            a2_list.append(awq_pack(q2, 4, N, K))
            g2s_list.append(s2)
            a2s_list.append(s2)

        gptq_res = _process_weights_emulation_gptq(
            torch.stack(g13_list),
            torch.stack(g2_list),
            torch.stack(g13s_list),
            torch.stack(g2s_list),
            torch.stack(g13z_list) if asym else None,
            torch.stack(g2z_list) if asym else None,
        )
        awq_res = _process_weights_emulation_awq(
            torch.stack(a13_list),
            torch.stack(a2_list),
            torch.stack(a13s_list),
            torch.stack(a2s_list),
            torch.stack(a13z_list) if asym else None,
            torch.stack(a2z_list) if asym else None,
        )

        assert torch.allclose(gptq_res[0].float(), awq_res[0].float(), atol=1e-3), (
            f"w13 max diff: {(gptq_res[0] - awq_res[0]).float().abs().max().item()}"
        )
        assert torch.allclose(gptq_res[1].float(), awq_res[1].float(), atol=1e-3), (
            f"w2 max diff: {(gptq_res[1] - awq_res[1]).float().abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# End-to-end MoE forward pass test
# ---------------------------------------------------------------------------

# (E, K, N, top_k, group_size, num_tokens)
E2E_CONFIGS = [
    pytest.param(4, 64, 32, 2, 32, 8, id="tiny"),
    pytest.param(8, 128, 64, 2, 64, 16, id="small"),
]


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


def _run_emulation_forward(
    experts,
    w13_bf16,
    w2_bf16,
    w13_scale,
    w2_scale,
    hidden_states,
    topk_weights,
    topk_ids,
    E,
    K,
    N,
):
    """Run Int4EmulationTritonExperts.apply and return output tensor."""
    ws13_size = hidden_states.shape[0] * topk_ids.shape[1] * max(N, K)
    ws2_size = hidden_states.shape[0] * topk_ids.shape[1] * max(2 * N, K)
    workspace13 = torch.zeros(ws13_size, dtype=torch.bfloat16, device=device)
    workspace2 = torch.zeros(ws2_size, dtype=torch.bfloat16, device=device)
    output = torch.zeros(hidden_states.shape[0], K, dtype=torch.bfloat16, device=device)

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


class TestInt4EmulationMoEForward:
    """End-to-end: GPTQ and AWQ paths must produce identical MoE outputs."""

    @pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS)
    def test_gptq_vs_awq_forward_agree(self, E, K, N, top_k, group_size, num_tokens):
        """GPTQ and AWQ emulation backends produce bit-identical forward outputs."""
        torch.manual_seed(42)

        moe_config = _make_moe_config(E, K, N)

        # Build float weights and quantize with both formats
        w13_fp = torch.randn(E, K, 2 * N, dtype=torch.float16, device=device)
        w2_fp = torch.randn(E, N, K, dtype=torch.float16, device=device)

        g13_list, g13s, a13_list, a13s = [], [], [], []
        g2_list, g2s, a2_list, a2s = [], [], [], []

        for e in range(E):
            q13, s13 = _quantize_sym(w13_fp[e], group_size)
            q2, s2 = _quantize_sym(w2_fp[e], group_size)
            g13_list.append(gptq_pack(q13, 4, K, 2 * N))
            a13_list.append(awq_pack(q13, 4, K, 2 * N))
            g13s.append(s13)
            a13s.append(s13.clone())
            g2_list.append(gptq_pack(q2, 4, N, K))
            a2_list.append(awq_pack(q2, 4, N, K))
            g2s.append(s2)
            a2s.append(s2.clone())

        gptq_w13 = torch.stack(g13_list)
        gptq_w13s = torch.stack(g13s)
        gptq_w2 = torch.stack(g2_list)
        gptq_w2s = torch.stack(g2s)

        awq_w13 = torch.stack(a13_list)
        awq_w13s = torch.stack(a13s)
        awq_w2 = torch.stack(a2_list)
        awq_w2s = torch.stack(a2s)

        # Dequantize to BF16 via each path
        gptq_res = _process_weights_emulation_gptq(
            gptq_w13, gptq_w2, gptq_w13s, gptq_w2s, None, None
        )
        awq_res = _process_weights_emulation_awq(
            awq_w13, awq_w2, awq_w13s, awq_w2s, None, None
        )

        w13_gptq_bf16, w2_gptq_bf16 = gptq_res[0], gptq_res[1]
        w13_awq_bf16, w2_awq_bf16 = awq_res[0], awq_res[1]

        # Build quant configs (scales nulled out by Int4EmulationTritonExperts.__init__)
        dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
        quant_cfg_gptq = int4_w4a16_moe_quant_config(dummy_scale, dummy_scale)
        quant_cfg_awq = int4_w4a16_moe_quant_config(dummy_scale, dummy_scale)

        experts_gptq = Int4EmulationTritonExperts(moe_config, quant_cfg_gptq)
        experts_awq = Int4EmulationTritonExperts(moe_config, quant_cfg_awq)

        hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
        )
        topk_ids = torch.stack(
            [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
        ).to(torch.int32)

        out_gptq = _run_emulation_forward(
            experts_gptq,
            w13_gptq_bf16,
            w2_gptq_bf16,
            gptq_w13s,
            gptq_w2s,
            hidden_states,
            topk_weights,
            topk_ids,
            E,
            K,
            N,
        )
        out_awq = _run_emulation_forward(
            experts_awq,
            w13_awq_bf16,
            w2_awq_bf16,
            awq_w13s,
            awq_w2s,
            hidden_states,
            topk_weights,
            topk_ids,
            E,
            K,
            N,
        )

        assert torch.allclose(out_gptq, out_awq, atol=0), (
            f"max diff: {(out_gptq - out_awq).abs().max().item()}"
        )

    @pytest.mark.parametrize("E, K, N, top_k, group_size, num_tokens", E2E_CONFIGS)
    @pytest.mark.parametrize("fmt", ["gptq", "awq"])
    def test_output_close_to_bf16_reference(
        self, E, K, N, top_k, group_size, num_tokens, fmt
    ):
        """Emulation output should be close to a direct BF16 MoE forward."""
        torch.manual_seed(11)

        moe_config = _make_moe_config(E, K, N)

        # Use wide-range weights so quantization error is small relative to
        # the dynamic range (group_size is small, so per-group scales are tight)
        w13_fp = torch.randn(E, K, 2 * N, dtype=torch.bfloat16, device=device) * 0.02
        w2_fp = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.02

        if fmt == "gptq":
            packed_list13, scale_list13 = [], []
            packed_list2, scale_list2 = [], []
            for e in range(E):
                q13, s13 = _quantize_sym(w13_fp[e].float(), group_size)
                q2, s2 = _quantize_sym(w2_fp[e].float(), group_size)
                packed_list13.append(gptq_pack(q13, 4, K, 2 * N))
                scale_list13.append(s13)
                packed_list2.append(gptq_pack(q2, 4, N, K))
                scale_list2.append(s2)
            res = _process_weights_emulation_gptq(
                torch.stack(packed_list13),
                torch.stack(packed_list2),
                torch.stack(scale_list13),
                torch.stack(scale_list2),
                None,
                None,
            )
        else:
            packed_list13, scale_list13 = [], []
            packed_list2, scale_list2 = [], []
            for e in range(E):
                q13, s13 = _quantize_sym(w13_fp[e].float(), group_size)
                q2, s2 = _quantize_sym(w2_fp[e].float(), group_size)
                packed_list13.append(awq_pack(q13, 4, K, 2 * N))
                scale_list13.append(s13)
                packed_list2.append(awq_pack(q2, 4, N, K))
                scale_list2.append(s2)
            res = _process_weights_emulation_awq(
                torch.stack(packed_list13),
                torch.stack(packed_list2),
                torch.stack(scale_list13),
                torch.stack(scale_list2),
                None,
                None,
            )

        w13_bf16, w2_bf16 = res[0], res[1]

        dummy_scale = torch.ones(1, dtype=torch.float16, device=device)
        quant_cfg = int4_w4a16_moe_quant_config(dummy_scale, dummy_scale)
        experts = Int4EmulationTritonExperts(moe_config, quant_cfg)

        hidden_states = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
        )
        topk_ids = torch.stack(
            [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
        ).to(torch.int32)

        out_emulation = _run_emulation_forward(
            experts,
            w13_bf16,
            w2_bf16,
            dummy_scale,
            dummy_scale,
            hidden_states,
            topk_weights,
            topk_ids,
            E,
            K,
            N,
        )

        # BF16 reference: use the dequantized BF16 weights directly in a
        # per-expert dense forward pass (matches what TritonExperts does)
        ref_output = torch.zeros(num_tokens, K, dtype=torch.bfloat16, device=device)
        import torch.nn.functional as F

        for m in range(num_tokens):
            acc = torch.zeros(K, dtype=torch.float32, device=device)
            for k in range(top_k):
                e = topk_ids[m, k].item()
                w = topk_weights[m, k].item()
                gate_up = hidden_states[m] @ w13_bf16[e].T  # [2N]
                gate, up = gate_up.chunk(2)
                act = F.silu(gate) * up
                down = act @ w2_bf16[e].T  # [K]
                acc += w * down.float()
            ref_output[m] = acc.bfloat16()

        # Quantization introduces ~1/15 relative error per group; with small
        # weights and small group sizes the absolute error is tight.
        rel_l2 = (
            torch.norm(out_emulation.float() - ref_output.float())
            / torch.norm(ref_output.float()).clamp(min=1e-6)
        ).item()
        assert rel_l2 < 0.15, f"[{fmt}] relative L2 = {rel_l2:.4f} (threshold 0.15)"
