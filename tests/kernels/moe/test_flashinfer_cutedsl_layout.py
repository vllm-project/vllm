# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-layout normalization for the FlashInfer CuTeDSL NVFP4 MoE backend."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w13_to_w31_for_flashinfer_cutedsl,
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
