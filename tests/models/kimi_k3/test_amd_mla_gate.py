# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import torch

from vllm.model_executor.layers.mla import _apply_mla_output_gate
from vllm.models.kimi_k3.amd.ops.mla_gate import kimi_k3_mla_output_gate


class _GateProjection(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, hidden_states: torch.Tensor):
        return torch.nn.functional.linear(hidden_states, self.weight), None


def test_default_mla_output_gate_is_unchanged():
    hidden_states = torch.tensor([[1.0, -2.0]])
    attention_output = torch.tensor([[0.5, -0.25]])
    gate_projection = _GateProjection(torch.tensor([[0.2, 0.3], [-0.4, 0.1]]))

    actual = _apply_mla_output_gate(
        hidden_states,
        attention_output,
        gate_projection,
        output_gate=None,
    )
    expected = attention_output * gate_projection(hidden_states)[0].sigmoid()

    torch.testing.assert_close(actual, expected)


def test_amd_mla_output_gate_uses_aiter_specialization_in_place():
    hidden_states = torch.tensor([[1.0, -2.0]])
    attention_output = torch.tensor([[0.5, -0.25]])
    gate_projection = _GateProjection(torch.ones((2, 2)))
    expected = torch.tensor([[3.0, 4.0]])

    def fused_gate(hidden, weight, attention, *, out):
        assert hidden is hidden_states
        assert weight is gate_projection.weight
        assert attention is attention_output
        assert out is attention_output
        return out.copy_(expected)

    with patch(
        "vllm.models.kimi_k3.amd.ops.mla_gate._get_aiter_mla_gate",
        return_value=(fused_gate, lambda *_: True),
    ):
        actual = kimi_k3_mla_output_gate(
            hidden_states,
            attention_output,
            gate_projection,
        )

    assert actual is attention_output
    torch.testing.assert_close(actual, expected)


def test_amd_mla_output_gate_falls_back_when_shape_is_unsupported():
    hidden_states = torch.tensor([[1.0, -2.0]])
    attention_output = torch.tensor([[0.5, -0.25]])
    gate_projection = _GateProjection(torch.tensor([[0.2, 0.3], [-0.4, 0.1]]))

    with patch(
        "vllm.models.kimi_k3.amd.ops.mla_gate._get_aiter_mla_gate",
        return_value=(None, lambda *_: False),
    ):
        actual = kimi_k3_mla_output_gate(
            hidden_states,
            attention_output,
            gate_projection,
        )

    expected = attention_output * gate_projection(hidden_states)[0].sigmoid()
    torch.testing.assert_close(actual, expected)
