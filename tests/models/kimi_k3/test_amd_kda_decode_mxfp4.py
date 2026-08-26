# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused KDA decode MXFP4 epilogue vs pin BF16 decode + reference quant.

Covers both scale layouts the o_proj GEMM consumes:

* ``plain`` — Triton ``dynamic_mxfp4_quant`` (column-major e8m0)
* ``shuffled`` — AITER ``per_1x32_f4_quant_hip(..., shuffle=True)``
"""

from __future__ import annotations

import pytest
import torch

from tests.models.kimi_k3.test_amd_kda_decode import (
    DTYPE,
    GATE_LOWER_BOUND,
    HEAD_DIM,
    NORM_EPS,
    KdaDecodeInputs,
    _requires_kernel,
    _run_fused,
)
from vllm.platforms import current_platform


def _on_supported_arch() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    return on_gfx950() or on_gfx942()


requires_kda_mxfp4 = pytest.mark.skipif(
    not _on_supported_arch(),
    reason="The fused KDA decode kernel is only built for gfx942 / gfx950",
)


def _pad8(n: int) -> int:
    return (n + 7) // 8 * 8


def _pad32(n: int) -> int:
    return (n + 31) // 32 * 32


def _pad256(n: int) -> int:
    return (n + 255) // 256 * 256


def _mx_scale_shuffle_idx(scale_n_pad: int, row: int, col: int) -> int:
    r0, r1, r2 = row // 32, (row % 32) // 16, row % 16
    c0, c1, c2 = col // 8, (col % 8) // 4, col % 4
    return ((((r0 * (scale_n_pad // 8) + c0) * 4 + c2) * 16 + r2) * 2 + c1) * 2 + r1


def _alloc(layout: str, num_tokens: int, dim: int, device: torch.device):
    n_groups = dim // 32
    if layout == "plain":
        data = torch.empty((num_tokens, dim // 2), dtype=torch.uint8, device=device)
        scale = torch.empty((n_groups, num_tokens), dtype=torch.uint8, device=device).T
        return data, scale
    data = torch.zeros((_pad32(num_tokens), dim // 2), dtype=torch.uint8, device=device)
    scale = torch.zeros(
        (_pad256(num_tokens), _pad8(n_groups)), dtype=torch.uint8, device=device
    )
    return data, scale


def _run_fused_mxfp4(inp: KdaDecodeInputs, layout: str):
    from vllm import _custom_ops as ops

    conv_state = inp.conv_state.clone()
    recurrent_state = inp.recurrent_state.clone()
    num_tokens = inp.mixed_qkv.shape[0]
    out = torch.empty(
        1,
        num_tokens,
        inp.num_heads,
        HEAD_DIM,
        device=inp.mixed_qkv.device,
        dtype=DTYPE,
    )
    mx_q, mx_s = _alloc(layout, num_tokens, inp.dim, inp.mixed_qkv.device)
    ops.fused_kda_decode(
        x=inp.mixed_qkv,
        weight=inp.decode_conv1d_weight,
        bias=None,
        conv_state=inp.conv_state_view(conv_state),
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        state=recurrent_state,
        out=out,
        lower_bound=GATE_LOWER_BOUND,
        output_gate=inp.g2,
        norm_weight=inp.decode_norm_weight,
        norm_eps=NORM_EPS,
        mxfp4_out=mx_q,
        mxfp4_scale=mx_s,
        mxfp4_layout=layout,
    )
    return mx_q, mx_s, conv_state, recurrent_state


@requires_kda_mxfp4
@torch.inference_mode()
@pytest.mark.parametrize("layout", ["plain", "shuffled"])
@pytest.mark.parametrize("num_heads", [12, 24])
@pytest.mark.parametrize("num_tokens", [1, 8, 32])
def test_fused_kda_decode_mxfp4_writes_layout(
    layout: str, num_heads: int, num_tokens: int
) -> None:
    """MXFP4 epilogue launches, writes live rows, and does not perturb state."""
    _requires_kernel()
    inp = KdaDecodeInputs(num_tokens, num_heads, num_slots=max(num_tokens, 4) + 3)
    mx_q, mx_s, conv_h, state_h = _run_fused_mxfp4(inp, layout)
    _, conv_bf16, state_bf16 = _run_fused(inp)

    torch.testing.assert_close(conv_h, conv_bf16, atol=0, rtol=0)
    torch.testing.assert_close(state_h, state_bf16, atol=2e-3, rtol=2e-3)
    assert mx_q[:num_tokens].any()
    assert mx_s.any()


@requires_kda_mxfp4
@torch.inference_mode()
def test_fused_kda_decode_mxfp4_null_block_writes_zero_pair() -> None:
    _requires_kernel()
    num_real, num_padded, num_heads = 3, 5, 12
    inp = KdaDecodeInputs(num_real + num_padded, num_heads, num_slots=12, seed=11)
    inp.state_indices[num_real:] = 0
    mx_q, mx_s, conv_h, state_h = _run_fused_mxfp4(inp, "plain")
    assert not mx_q[num_real:].any()
    assert not mx_s[num_real:].any()
    torch.testing.assert_close(conv_h[0], inp.conv_state[0], atol=0, rtol=0)
    torch.testing.assert_close(state_h[0], inp.recurrent_state[0], atol=0, rtol=0)


@requires_kda_mxfp4
@torch.inference_mode()
def test_fused_kda_decode_mxfp4_requires_onorm() -> None:
    _requires_kernel()
    from vllm import _custom_ops as ops

    inp = KdaDecodeInputs(4, 12, num_slots=8)
    mx_q, mx_s = _alloc("plain", 4, inp.dim, inp.mixed_qkv.device)
    out = torch.empty(1, 4, 12, HEAD_DIM, device="cuda", dtype=DTYPE)
    with pytest.raises(RuntimeError, match="MXFP4 output requires"):
        ops.fused_kda_decode(
            x=inp.mixed_qkv,
            weight=inp.decode_conv1d_weight,
            bias=None,
            conv_state=inp.conv_state_view(inp.conv_state.clone()),
            raw_g=inp.g1,
            raw_beta=inp.beta,
            A_log=inp.A_log,
            dt_bias=inp.dt_bias,
            state_indices=inp.state_indices,
            state=inp.recurrent_state.clone(),
            out=out,
            lower_bound=GATE_LOWER_BOUND,
            mxfp4_out=mx_q,
            mxfp4_scale=mx_s,
            mxfp4_layout="plain",
        )


@requires_kda_mxfp4
@torch.inference_mode()
def test_fused_kda_decode_mxfp4_rejects_unknown_layout() -> None:
    _requires_kernel()
    from vllm import _custom_ops as ops

    inp = KdaDecodeInputs(2, 12, num_slots=5)
    mx_q, mx_s = _alloc("plain", 2, inp.dim, inp.mixed_qkv.device)
    out = torch.empty(1, 2, 12, HEAD_DIM, device="cuda", dtype=DTYPE)
    with pytest.raises(RuntimeError, match="plain' or 'shuffled"):
        ops.fused_kda_decode(
            x=inp.mixed_qkv,
            weight=inp.decode_conv1d_weight,
            bias=None,
            conv_state=inp.conv_state_view(inp.conv_state.clone()),
            raw_g=inp.g1,
            raw_beta=inp.beta,
            A_log=inp.A_log,
            dt_bias=inp.dt_bias,
            state_indices=inp.state_indices,
            state=inp.recurrent_state.clone(),
            out=out,
            lower_bound=GATE_LOWER_BOUND,
            output_gate=inp.g2,
            norm_weight=inp.decode_norm_weight,
            norm_eps=NORM_EPS,
            mxfp4_out=mx_q,
            mxfp4_scale=mx_s,
            mxfp4_layout="swizzled",
        )


def test_mxfp4_layout_is_on_the_activation_not_the_key() -> None:
    """Scheme vs ABI: kMxfp4Dynamic is unchanged; shuffled is a layout bit."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    assert not hasattr(kMxfp4Dynamic, "layout")
    plain = QuantizedActivation(
        data=torch.zeros(1, 4, dtype=torch.uint8),
        scale=torch.zeros(1, 1, dtype=torch.uint8),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([1, 8]),
        quant_key=kMxfp4Dynamic,
        layout=None,
    )
    shuffled = QuantizedActivation(
        data=plain.data,
        scale=plain.scale,
        orig_dtype=plain.orig_dtype,
        orig_shape=plain.orig_shape,
        quant_key=kMxfp4Dynamic,
        layout="shuffled",
    )
    assert plain.quant_key == shuffled.quant_key == kMxfp4Dynamic
    assert plain.layout is None
    assert shuffled.layout == "shuffled"


def test_mxfp4_layout_for_oproj_reads_layer_layout() -> None:
    from vllm.models.kimi_k3.amd.ops.kda_decode import mxfp4_layout_for_oproj
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
        kFp8StaticTensorSym,
    )

    class _Layer:
        pass

    missing = _Layer()
    assert mxfp4_layout_for_oproj(missing) is None

    fp8 = _Layer()
    fp8.input_quant_key = kFp8StaticTensorSym
    assert mxfp4_layout_for_oproj(fp8) is None

    triton = _Layer()
    triton.input_quant_key = kMxfp4Dynamic
    assert mxfp4_layout_for_oproj(triton) == "plain"

    asm = _Layer()
    asm.input_quant_key = kMxfp4Dynamic
    asm.input_quant_layout = "shuffled"
    assert mxfp4_layout_for_oproj(asm) == "shuffled"


def test_mx_scale_shuffle_idx_matches_aiter_view() -> None:
    """Guard the device shuffle against aiter's permute."""
    sm, sn = 32, 48
    dummy = torch.arange(sm * sn, dtype=torch.int32).view(sm, sn)
    shuffled = (
        dummy.view(sm // 32, 2, 16, sn // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(sm, sn)
    )
    for row in range(8):
        for col in range(sn):
            assert shuffled.view(-1)[_mx_scale_shuffle_idx(sn, row, col)].item() == dummy[
                row, col
            ].item()
