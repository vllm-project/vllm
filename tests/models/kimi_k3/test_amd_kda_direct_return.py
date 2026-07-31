# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


class _ReturningAttention(nn.Module):
    def __init__(self, result: torch.Tensor) -> None:
        super().__init__()
        self.result = result

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.result


class _WritingAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self.output = output
        output.copy_(hidden_states + 1)


def _make_layer(self_attn: nn.Module, writes_output: bool) -> KimiDecoderLayer:
    layer = object.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = self_attn
    layer._self_attn_writes_output = writes_output
    return layer


def test_kda_self_attention_returns_projection_storage_directly() -> None:
    hidden_states = torch.randn(4, 8)
    projected = torch.randn_like(hidden_states)
    layer = _make_layer(_ReturningAttention(projected), writes_output=False)

    output = layer._run_self_attn(torch.arange(4), hidden_states)

    assert output is projected
    assert output.data_ptr() == projected.data_ptr()


def test_mla_self_attention_keeps_explicit_output_buffer() -> None:
    hidden_states = torch.randn(4, 8)
    attention = _WritingAttention()
    layer = _make_layer(attention, writes_output=True)

    output = layer._run_self_attn(torch.arange(4), hidden_states)

    torch.testing.assert_close(output, hidden_states + 1)
    assert attention.output is output
    assert output.data_ptr() != hidden_states.data_ptr()
