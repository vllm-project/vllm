# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the DeepGrove Maple checkpoint contract."""

import pytest

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.maple import (
    MapleForCausalLM,
    sliding_window_for_layer,
)
from vllm.transformers_utils.config import patch_rope_parameters
from vllm.transformers_utils.configs.maple import MapleConfig


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        # Maple names the embedding `word_embeddings`, unlike every other
        # decoder-only checkpoint vLLM loads.
        ("model.word_embeddings.weight", "model.embed_tokens.weight", None),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "v",
        ),
        # Expert weights must reach the MoE loader untouched: rewriting
        # `gate_proj`/`up_proj` here would shadow the per-expert stacking.
        (
            "model.layers.0.mlp.experts.7.gate_proj.weight",
            "model.layers.0.mlp.experts.7.gate_proj.weight",
            None,
        ),
        ("model.layers.0.mlp.gate.weight", "model.layers.0.mlp.gate.weight", None),
    ],
)
def test_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert MapleForCausalLM.hf_to_vllm_mapper._map_name_with_shard(checkpoint_name) == (
        runtime_name,
        shard_id,
    )


@pytest.mark.cpu_test
def test_rope_only_covers_half_of_each_head(default_vllm_config):
    """`partial_rotary_factor` must survive into `rope_parameters`.

    Losing it rotates all 128 head dims instead of 64, which loads fine and
    produces silently wrong logits.
    """
    config = MapleConfig()
    patch_rope_parameters(config)

    rotary_emb = get_rope(
        config.head_dim,
        max_position=config.max_position_embeddings,
        rope_parameters=config.rope_parameters,
    )
    assert rotary_emb.rotary_dim == config.head_dim // 2


@pytest.mark.cpu_test
def test_default_layer_types_interleave_three_sliding_per_global():
    config = MapleConfig()
    assert len(config.layer_types) == config.num_hidden_layers
    assert config.layer_types[:4] == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize(("layer_idx", "expected"), [(0, 513), (3, None)])
def test_sliding_window_boundary_is_inclusive(layer_idx, expected):
    """The checkpoint runs FlashAttention with `window_size=(sliding_window, 0)`.

    That boundary is inclusive, so a sliding layer attends `sliding_window + 1`
    positions. Forwarding `config.sliding_window` unchanged drops the oldest
    token of every local window, which costs ~2.6 points of top-1 agreement
    against the reference implementation in the windowed region.
    """
    config = MapleConfig()
    assert config.sliding_window == 512
    assert sliding_window_for_layer(config, layer_idx) == expected
