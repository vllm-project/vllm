# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN readout buffer must be wide enough for an fp32 SSM state.

With fp16 activations and an fp32 state the unnormalized readout can exceed
fp16's 65504. Widening ``o`` inside the kernel wrapper is not enough on its own:
the value is then stored into ``core_attn_out``, so that buffer has to be wide
too, and the normalized result is narrowed back before ``out_proj``.
"""

import types

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

pytestmark = pytest.mark.cpu_test

LARGE = 1e6


def _normalize(x, z):
    return RMSNormGated.forward_static(
        x, z, weight=torch.ones(x.shape[-1]), epsilon=1e-5, orig_dtype=x.dtype
    )


def _bare_layer():
    return QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)


def test_fp16_buffer_turns_a_large_readout_into_nan():
    readout = torch.full((2, 128), LARGE, dtype=torch.float32)
    fp16_buffer = torch.zeros((2, 128), dtype=torch.float16)
    fp16_buffer[:] = readout

    assert torch.isinf(fp16_buffer).all()
    assert torch.isnan(_normalize(fp16_buffer, torch.zeros_like(fp16_buffer))).all()


def test_the_same_readout_survives_the_norm_in_fp32():
    readout = torch.full((2, 128), LARGE, dtype=torch.float32)

    assert torch.isfinite(_normalize(readout, torch.zeros_like(readout))).all()


@pytest.mark.parametrize(
    "activation_dtype,state_dtype,expected",
    [
        (torch.float16, torch.float32, torch.float32),
        (torch.float16, torch.float16, torch.float16),
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.float32, torch.float32, torch.float32),
    ],
)
def test_buffer_is_widened_only_for_fp16_with_fp32_state(
    activation_dtype, state_dtype, expected
):
    layer = _bare_layer()
    layer.kv_cache = [
        torch.zeros(1, dtype=torch.float16),
        torch.zeros(1, dtype=state_dtype),
    ]

    assert layer._core_attn_out_dtype(activation_dtype) is expected


def test_no_kv_cache_leaves_the_dtype_alone():
    assert _bare_layer()._core_attn_out_dtype(torch.float16) is torch.float16


def test_output_projection_narrows_a_widened_readout_to_the_activation_dtype():
    seen = {}

    def out_proj(x):
        seen["dtype"] = x.dtype
        return x, None

    layer = types.SimpleNamespace(norm=lambda x, z: _normalize(x, z.float()), out_proj=out_proj)
    layer._output_projection = types.MethodType(
        QwenGatedDeltaNetAttention._output_projection, layer
    )
    readout = torch.full((2, 4, 32), LARGE, dtype=torch.float32)
    gate = torch.zeros((2, 4, 32), dtype=torch.float16)

    out = layer._output_projection(readout, gate)

    assert seen["dtype"] is torch.float16
    assert torch.isfinite(out).all()
