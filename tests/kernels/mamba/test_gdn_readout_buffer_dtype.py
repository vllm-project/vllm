# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN readout buffer overflows before RMSNormGated can normalize it.

Upstream PR #54146 widens the readout inside the kernel wrapper, but in this
tree the value is copied straight into ``core_attn_out``, which is allocated in
the *activation* dtype, and the norm runs after that copy. With fp16
activations and an fp32 SSM state the ``inf`` is produced one line lower down,
so both changes are needed to close the path that ends in NaN logits.

CPU only: this is dtype arithmetic and a policy check, no GPU required.
"""

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

pytestmark = pytest.mark.cpu_test

# Magnitude the unnormalized GDN accumulator reaches over long context.
# fp16 tops out at 65504.
LARGE = 1e6


def _normalize(x, z):
    return RMSNormGated.forward_static(
        x, z, weight=torch.ones(x.shape[-1]), epsilon=1e-5, orig_dtype=x.dtype
    )


def test_fp16_buffer_turns_a_large_readout_into_nan():
    """Reproduces the defect: the store to fp16 kills the value before the norm."""
    readout = torch.full((2, 128), LARGE, dtype=torch.float32)
    gate = torch.zeros_like(readout)

    fp16_buffer = torch.zeros((2, 128), dtype=torch.float16)
    fp16_buffer[:] = readout  # exactly what `core_attn_out[...] = o` does

    assert torch.isinf(fp16_buffer).all()
    assert torch.isnan(_normalize(fp16_buffer, gate.half())).all()


def test_the_same_readout_survives_the_norm_in_fp32():
    """The value is not intrinsically bad; only the buffer width kills it."""
    readout = torch.full((2, 128), LARGE, dtype=torch.float32)
    gate = torch.zeros_like(readout)

    fp32_buffer = torch.zeros((2, 128), dtype=torch.float32)
    fp32_buffer[:] = readout

    assert torch.isfinite(_normalize(fp32_buffer, gate)).all()


def _bare_layer():
    return QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)


@pytest.mark.parametrize(
    "activation_dtype,state_dtype,expected",
    [
        # The combination that overflows today.
        (torch.float16, torch.float32, torch.float32),
        # Narrow state: nothing to widen for.
        (torch.float16, torch.float16, torch.float16),
        # bf16 shares fp32's exponent range, so it never overflows.
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.float32, torch.float32, torch.float32),
    ],
)
def test_buffer_is_widened_only_when_needed(activation_dtype, state_dtype, expected):
    layer = _bare_layer()
    layer.kv_cache = [
        torch.zeros(1, dtype=torch.float16),
        torch.zeros(1, dtype=state_dtype),
    ]

    assert layer._core_attn_out_dtype(activation_dtype) is expected


def test_no_kv_cache_leaves_the_dtype_alone():
    """The cache is not bound yet during construction and profiling."""
    assert _bare_layer()._core_attn_out_dtype(torch.float16) is torch.float16
