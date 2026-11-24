# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-layout normalization for the FlashInfer CuTeDSL MoE backends."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w13_to_w31_for_flashinfer_cutedsl,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    interleave_w13_up_gate_for_flashinfer_cutedsl_sm90,
)

_GATE = torch.tensor([[[1], [2], [3], [4]]])
_UP = torch.tensor([[[10], [20], [30], [40]]])
_EXPECTED = torch.cat([_UP, _GATE], dim=1)


def test_reorder_w13_swigluoai_interleaved():
    """gpt-oss w13 is [gate0, up0, gate1, ...] rather than packed [gate; up]."""
    w13 = torch.empty(1, 8, 1, dtype=_GATE.dtype)
    w13[:, 0::2] = _GATE
    w13[:, 1::2] = _UP

    out, out_scale = reorder_w13_to_w31_for_flashinfer_cutedsl(
        MoEActivation.SWIGLUOAI, w13, w13 + 100
    )

    torch.testing.assert_close(out, _EXPECTED)
    torch.testing.assert_close(out_scale, _EXPECTED + 100)


@pytest.mark.parametrize(
    "activation", [MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE]
)
def test_reorder_w13_packed_layouts(activation: MoEActivation):
    w13 = torch.cat([_GATE, _UP], dim=1)

    out, out_scale = reorder_w13_to_w31_for_flashinfer_cutedsl(
        activation, w13, w13 + 100
    )

    torch.testing.assert_close(out, _EXPECTED)
    torch.testing.assert_close(out_scale, _EXPECTED + 100)


def test_interleave_w13_up_gate_for_flashinfer_cutedsl_sm90():
    """`[gate; up]` rows are regrouped as 32-row up/gate blocks,
    `[up 0:32 | gate 0:32 | up 32:64 | gate 32:64]`; the intermediate size must
    be a multiple of 32."""
    inter = 64
    gate = torch.arange(inter).reshape(1, inter, 1)
    up = gate + 100
    out = interleave_w13_up_gate_for_flashinfer_cutedsl_sm90(
        torch.cat([gate, up], dim=1)
    )
    expected = torch.cat([up[:, :32], gate[:, :32], up[:, 32:], gate[:, 32:]], dim=1)
    torch.testing.assert_close(out, expected)
    with pytest.raises(ValueError, match="multiple of 32"):
        interleave_w13_up_gate_for_flashinfer_cutedsl_sm90(torch.zeros(1, 2 * 48, 1))
