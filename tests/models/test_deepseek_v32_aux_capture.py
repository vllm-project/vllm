# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture the reduced residual, not a TP rank's partial MoE output."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import model as implementation


class Norm(nn.Module):
    def forward(self, x, residual=None):
        if residual is None:
            return x * 0.25
        # Deliberately mutate like fused inference RMSNorm implementations.
        residual.add_(x)
        return residual * 0.25, residual


class Attention(nn.Module):
    def forward(self, *, positions, hidden_states):
        return hidden_states * 0.5


def tiny_model(taps, sp):
    model = implementation.DeepseekV32Model.__new__(implementation.DeepseekV32Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(rms_norm_eps=1e-5)
    model.replicated_embed = False
    model.use_sequence_parallel = sp
    model.start_layer, model.end_layer = 0, 2
    model.aux_hidden_state_layers = taps
    model.layers = nn.ModuleList()
    for _ in range(2):
        layer = implementation.DeepseekV32DecoderLayer.__new__(
            implementation.DeepseekV32DecoderLayer
        )
        nn.Module.__init__(layer)
        layer.use_sequence_parallel = sp
        layer.input_layernorm = Norm()
        layer.post_attention_layernorm = Norm()
        layer.self_attn = Attention()
        layer.mlp = nn.Identity()
        model.layers.append(layer)
    model.norm = Norm()
    return model


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("taps", [(0, 1), (2,), (0, 1, 2)])
def test_aux_states_are_complete_immutable_block_outputs(sp, taps):
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Simulate TP=2 collectives, using identical partial contributions. SP
    # helpers keep token layout small but still reduce attention output.
    reductions = []

    def allreduce_norm(hidden, residual, norm):
        reductions.append(1)
        return norm(hidden * 2, residual)

    replacements: dict[str, Any] = {
        "get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        "fused_allreduce_rms_norm": allreduce_norm,
        "sp_shard": lambda x: x,
        "sp_all_gather": lambda x: x,
        "sp_reduce_scatter": lambda x: x * 2,
    }
    with patch.multiple(implementation, **replacements):
        baseline = tiny_model((), sp)(
            None, torch.arange(2), inputs_embeds=inputs.clone()
        )
        baseline_reductions = len(reductions)
        reductions.clear()
        output, states = tiny_model(taps, sp)(
            None, torch.arange(2), inputs_embeds=inputs.clone()
        )
    assert len(states) == len(taps)
    # Each attention adds 2 * (x/4)/2 = x/4, then MLP is identity
    # on norm(x*1.25). Non-SP MLP output is TP-partial (factor 2).
    factor = 1.25 * (1 + (1 if sp else 2) / 4)
    for i, state in zip(taps, states, strict=True):
        torch.testing.assert_close(state, inputs * factor**i)
    torch.testing.assert_close(output, baseline)
    assert len(reductions) == baseline_reductions
