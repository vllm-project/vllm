# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AITER MXFP4 W4A16 gate/up interleave dispatch (GFX950)."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    _aiter_mxfp4_w4a16_gu_interleaved,
)


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        (MoEActivation.SWIGLUOAI, True),
        (MoEActivation.SILU, False),
        (MoEActivation.GELU, False),
        (MoEActivation.SWIGLUOAI_UNINTERLEAVE, False),
        (None, False),
    ],
)
def test_aiter_mxfp4_w4a16_gu_interleaved(activation, expected):
    layer = SimpleNamespace(activation=activation)
    assert _aiter_mxfp4_w4a16_gu_interleaved(layer) is expected


def test_aiter_mxfp4_w4a16_gu_interleaved_missing_activation_attr():
    layer = SimpleNamespace()
    assert _aiter_mxfp4_w4a16_gu_interleaved(layer) is False
