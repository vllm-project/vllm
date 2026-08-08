# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer, KimiMLAAttention


class _DirectMLA(KimiMLAAttention):
    def __init__(self, result: torch.Tensor):
        nn.Module.__init__(self)
        self.result = result

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.result


class _BufferedAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        output.copy_(hidden_states + positions[:, None])


def _make_layer(self_attn: nn.Module) -> KimiDecoderLayer:
    layer = KimiDecoderLayer.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = self_attn
    return layer


def test_mla_returns_projection_output_without_copy():
    hidden_states = torch.randn(4, 8)
    positions = torch.arange(4)
    projected = torch.randn_like(hidden_states)
    layer = _make_layer(_DirectMLA(projected))

    output = layer._run_self_attn(positions, hidden_states)

    assert output is projected


def test_kda_keeps_caller_owned_output():
    hidden_states = torch.randn(4, 8)
    positions = torch.arange(4)
    layer = _make_layer(_BufferedAttention())

    output = layer._run_self_attn(positions, hidden_states)

    torch.testing.assert_close(output, hidden_states + positions[:, None])
