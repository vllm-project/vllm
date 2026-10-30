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
    layer.get_state_dtype = lambda: (torch.float16, state_dtype)

    assert layer._core_attn_out_dtype(activation_dtype) is expected


def test_widening_does_not_wait_for_the_kv_cache():
    """The buffer is allocated in the compiled graph, which is traced during
    profiling, before the cache is bound: the dtype must not depend on it."""
    layer = _bare_layer()
    layer.get_state_dtype = lambda: (torch.float16, torch.float32)

    assert getattr(layer, "kv_cache", None) is None
    assert layer._core_attn_out_dtype(torch.float16) is torch.float32


def _layer_with_state(state_dtype):
    layer = _bare_layer()
    layer.prefix = "layers.0.linear_attn"
    layer.kv_cache = [
        torch.zeros(1, dtype=torch.float16),
        torch.zeros(1, dtype=state_dtype),
    ]
    return layer


def test_a_narrow_buffer_over_an_fp32_state_fails_loudly():
    """The 8-sep regression: the graph froze an fp16 buffer while the cache
    held fp32 state, and the readout fell back to a rounding copy in silence."""
    layer = _layer_with_state(torch.float32)
    buffer = torch.zeros(1, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="need torch.float32"):
        layer._check_readout_buffer(buffer, torch.float16)


@pytest.mark.parametrize(
    "activation_dtype,state_dtype",
    [
        (torch.float16, torch.float32),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_the_buffer_the_config_picks_passes_the_runtime_check(
    activation_dtype, state_dtype
):
    layer = _layer_with_state(state_dtype)
    layer.get_state_dtype = lambda: (torch.float16, state_dtype)
    buffer = torch.zeros(1, dtype=layer._core_attn_out_dtype(activation_dtype))

    layer._check_readout_buffer(buffer, activation_dtype)
