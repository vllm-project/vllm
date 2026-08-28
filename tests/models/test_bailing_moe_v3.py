# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models import bailing_moe_v3
from vllm.model_executor.models.bailing_moe_v3 import (
    BailingMoeV3ForCausalLM,
    BailingMoeV3Model,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec


class _AddOneLayer(nn.Module):
    def forward(self, hidden_states, positions, residual):
        del positions
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states + 1, residual


class _FinalNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states + residual, None


def test_bailing_v3_returns_requested_aux_hidden_states(monkeypatch):
    pp_group = type(
        "PPGroup",
        (),
        {"is_first_rank": True, "is_last_rank": True},
    )()
    monkeypatch.setattr(bailing_moe_v3, "get_pp_group", lambda: pp_group)

    inner_model = BailingMoeV3Model.__new__(BailingMoeV3Model)
    nn.Module.__init__(inner_model)
    inner_model.start_layer = 0
    inner_model.end_layer = 3
    inner_model.layers = nn.ModuleList([_AddOneLayer() for _ in range(3)])
    inner_model.word_embeddings = nn.Embedding(1, 2)
    nn.init.zeros_(inner_model.word_embeddings.weight)
    inner_model.norm = _FinalNorm()
    inner_model.do_not_compile = True

    model = BailingMoeV3ForCausalLM.__new__(BailingMoeV3ForCausalLM)
    nn.Module.__init__(model)
    model.model = inner_model
    model.set_aux_hidden_state_layers((1, 3))

    hidden_states, aux_hidden_states = model.forward(
        input_ids=torch.tensor([0]), positions=torch.tensor([0])
    )

    torch.testing.assert_close(hidden_states, torch.full((1, 2), 3.0))
    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(aux_hidden_states[0], torch.full((1, 2), 1.0))
    torch.testing.assert_close(aux_hidden_states[1], torch.full((1, 2), 3.0))
    assert model.supports_eagle3


def test_ling_dspark_attention_uses_independent_kv_block_size():
    attention = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        sliding_window=None,
        kv_cache_dtype="auto",
        kv_cache_torch_dtype=torch.bfloat16,
        num_kv_heads=16,
        head_size=128,
        head_size_v=128,
        kv_cache_block_size=128,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=1152),
    )

    spec = Attention.get_kv_cache_spec(attention, vllm_config)

    assert isinstance(spec, FullAttentionSpec)
    assert spec.block_size == 128

    attention.kv_cache_block_size = None
    spec = Attention.get_kv_cache_spec(attention, vllm_config)
    assert spec.block_size == 1152
