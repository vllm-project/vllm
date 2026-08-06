# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


class _ReturningAttention(nn.Module):
    def __init__(self, result: torch.Tensor) -> None:
        super().__init__()
        self.result = result

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.result


class _TupleProjection(nn.Module):
    def __init__(self, result: torch.Tensor) -> None:
        super().__init__()
        self.result = result

    def forward(self, _: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.result, None


def _make_layer(self_attn: nn.Module) -> KimiDecoderLayer:
    layer = object.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = self_attn
    return layer


def test_kda_self_attention_returns_projection_storage_directly() -> None:
    hidden_states = torch.randn(4, 8)
    projected = torch.randn_like(hidden_states)
    layer = _make_layer(_ReturningAttention(projected))

    output = layer._run_self_attn(torch.arange(4), hidden_states)

    assert output is projected
    assert output.data_ptr() == projected.data_ptr()


def test_k3_kda_forward_returns_output_projection_storage() -> None:
    hidden_states = torch.randn(4, 8)
    projected = torch.randn_like(hidden_states)

    def fill_core_output(**kwargs: torch.Tensor) -> None:
        kwargs["core_attn_out"].fill_(1)

    attention = SimpleNamespace(
        local_projection_size=4,
        head_dim=2,
        local_num_heads=2,
        in_proj_padding=0,
        in_proj_qkvgfab=_TupleProjection(torch.randn(4, 20)),
        f_b_proj=_TupleProjection(torch.randn(4, 4)),
        o_proj=_TupleProjection(projected),
        _forward=fill_core_output,
    )

    output = KimiK3DeltaAttention.forward(attention, hidden_states, torch.arange(4))

    assert output is projected
    assert output.data_ptr() == projected.data_ptr()
