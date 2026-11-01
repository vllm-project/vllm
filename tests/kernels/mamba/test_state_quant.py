# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the reference Mamba/SSM state int8 quantization
utilities in ``vllm.model_executor.layers.mamba.state_quant``.

These tests exercise only the pure-PyTorch quantize/dequantize helpers
in isolation; they intentionally do not exercise any real Mamba kernel
or model, since that wiring is out of scope for this change (see the
module docstring in ``state_quant.py`` for what is and is not covered).
"""

import torch

from vllm.model_executor.layers.mamba.state_quant import (
    QuantizedSSMState,
    dequantize_state_int8,
    quantize_state_int8,
    simulate_quantized_recurrence_step,
)


def test_round_trip_error_is_bounded():
    torch.manual_seed(0)
    state = torch.randn(2, 4, 8, 16, dtype=torch.float32) * 3.0

    qdata, scale = quantize_state_int8(state)
    recon = dequantize_state_int8(qdata, scale)

    assert qdata.dtype == torch.int8
    assert qdata.shape == state.shape
    assert scale.shape == (2, 4, 8, 1)

    max_abs_per_row = state.abs().amax(dim=-1, keepdim=True)
    tolerance = (max_abs_per_row / 127.0) + 1e-4
    assert torch.all((recon - state).abs() <= tolerance)


def test_qdata_values_within_int8_symmetric_range():
    torch.manual_seed(1)
    state = torch.randn(3, 5, 9, dtype=torch.float32) * 10.0
    qdata, _ = quantize_state_int8(state)
    assert qdata.min().item() >= -127
    assert qdata.max().item() <= 127


def test_all_zero_state_quantizes_without_nan_or_inf():
    state = torch.zeros(2, 3, 4, dtype=torch.float32)
    qdata, scale = quantize_state_int8(state)
    assert torch.all(qdata == 0)
    assert torch.isfinite(scale).all()
    recon = dequantize_state_int8(qdata, scale)
    assert torch.all(recon == 0)


def test_dequantize_rejects_non_int8_input():
    state = torch.randn(2, 3)
    raised = False
    try:
        dequantize_state_int8(state, torch.ones(2, 1))
    except ValueError:
        raised = True
    assert raised


def test_quantized_ssm_state_wrapper_round_trip():
    torch.manual_seed(2)
    state = torch.randn(4, 6, 12, dtype=torch.float32)
    qstate = QuantizedSSMState.from_float(state)
    recon = qstate.to_float()
    assert recon.shape == state.shape
    assert (recon - state).abs().max().item() < 0.2


def test_quantized_ssm_state_update_in_place():
    torch.manual_seed(4)
    state = torch.randn(2, 3, 5, dtype=torch.float32)
    qstate = QuantizedSSMState.from_float(state)
    new_state = torch.randn(2, 3, 5, dtype=torch.float32) * 5.0
    qstate.update_(new_state)
    recon = qstate.to_float()
    assert (recon - new_state).abs().max().item() < 1.0


def test_quantized_recurrence_drift_is_bounded_over_many_steps():
    torch.manual_seed(3)
    shape = (2, 4, 8)
    h0 = torch.zeros(*shape, dtype=torch.float32)
    a = torch.full(shape, 0.9, dtype=torch.float32)

    h_ref = h0.clone()
    state_q = QuantizedSSMState.from_float(h0)

    num_steps = 50
    for _ in range(num_steps):
        b = torch.rand(*shape, dtype=torch.float32) * 0.5 + 0.5
        x = torch.randn(*shape, dtype=torch.float32)

        h_ref = a * h_ref + b * x
        state_q = simulate_quantized_recurrence_step(state_q, a, b, x)

    h_quant = state_q.to_float()
    rel_error = (h_quant - h_ref).abs() / (h_ref.abs() + 1e-3)
    assert rel_error.mean().item() < 0.25
