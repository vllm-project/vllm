# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.models import qwen3_next


class _IdentityNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _SequenceParallelAttention(nn.Module):
    def __init__(self, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.input_num_tokens: list[int] = []
        self.num_tokens = 0
        self.full_num_tokens = 4

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.input_num_tokens.append(hidden_states.shape[0])
        if hidden_states.shape[0] != self.full_num_tokens:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
        self.num_tokens = hidden_states.shape[0]
        assert self.num_tokens % 2 == 0
        return hidden_states.chunk(2, dim=0)[0]


class _RecordingMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        assert already_sequence_parallel
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_qwen_decoder_uses_linear_sequence_parallel_boundary(
    monkeypatch,
    layer_type: str,
):
    layer = object.__new__(qwen3_next.Qwen3NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = layer_type
    layer.layer_scale = False
    layer.use_attn_reduce_scatter_for_moe = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    attention = _SequenceParallelAttention(layer_type)
    if layer_type == "full_attention":
        layer.self_attn = attention
    else:
        layer.linear_attn = attention
    layer.mlp = _RecordingMoe()

    shard = Mock(side_effect=lambda hidden_states: hidden_states[:2].clone())
    monkeypatch.setattr(qwen3_next, "sequence_parallel_chunk", shard)

    positions = torch.arange(4)
    full_hidden_states = torch.arange(8, dtype=torch.float32).view(4, 2)
    hidden_states, residual = layer(
        positions=positions,
        hidden_states=full_hidden_states,
        residual=None,
    )

    assert hidden_states.shape == residual.shape == (2, 2)
    assert attention.num_tokens == 4
    assert attention.input_num_tokens == [4]
    assert layer.mlp.num_tokens == 2
    shard.assert_called_once_with(full_hidden_states)
    sequence_shard = hidden_states

    hidden_states, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    assert hidden_states.shape == residual.shape == (2, 2)
    assert attention.num_tokens == 4
    assert attention.input_num_tokens == [4, 2]
    assert layer.mlp.num_tokens == 2
    torch.testing.assert_close(sequence_shard, hidden_states)
    shard.assert_called_once()
