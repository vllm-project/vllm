# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w13_for_flashinfer_cutedsl,
)


def test_reorder_w13_for_flashinfer_cutedsl_swigluoai_interleaved():
    gate = torch.tensor([[[1], [2], [3], [4]]])
    up = torch.tensor([[[10], [20], [30], [40]]])
    w13 = torch.empty(1, 8, 1, dtype=gate.dtype)
    w13[:, 0::2] = gate
    w13[:, 1::2] = up
    w13_scale = w13 + 100

    layer = SimpleNamespace(activation=MoEActivation.SWIGLUOAI)
    out, out_scale = reorder_w13_for_flashinfer_cutedsl(layer, w13, w13_scale)

    expected = torch.cat([up, gate], dim=1)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out_scale, expected + 100)


def test_reorder_w13_for_flashinfer_cutedsl_packed_layouts():
    gate = torch.tensor([[[1], [2], [3], [4]]])
    up = torch.tensor([[[10], [20], [30], [40]]])
    w13 = torch.cat([gate, up], dim=1)
    w13_scale = w13 + 100

    for activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE):
        layer = SimpleNamespace(activation=activation)
        out, out_scale = reorder_w13_for_flashinfer_cutedsl(layer, w13, w13_scale)

        expected = torch.cat([up, gate], dim=1)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(out_scale, expected + 100)


def test_reorder_w13_for_flashinfer_cutedsl_relu2_no_mul_noop():
    w13 = torch.tensor([[[1], [2], [3], [4]]])
    w13_scale = w13 + 100

    layer = SimpleNamespace(activation=MoEActivation.RELU2_NO_MUL)
    out, out_scale = reorder_w13_for_flashinfer_cutedsl(layer, w13, w13_scale)

    assert out is w13
    assert out_scale is w13_scale
    torch.testing.assert_close(out, w13)
    torch.testing.assert_close(out_scale, w13_scale)
