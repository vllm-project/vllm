# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PretrainedConfig

from vllm.model_executor.models.deepseek_mtp import (
    _get_mtp_block_config,
    _map_fast_mtp_weight_name,
)

pytestmark = pytest.mark.skip_global_cleanup


def _fast_mtp_config(**overrides):
    values = {
        "num_hidden_layers": 12,
        "num_nextn_predict_layers": 1,
        "mtp_num_heads": 1,
        "mtp_moe": False,
        "mtp_share_embedding_weights": True,
        "mtp_share_lm_head": True,
        "mtp_share_norm": True,
        "n_routed_experts": 64,
    }
    values.update(overrides)
    return PretrainedConfig(**values)


def test_fast_mtp_uses_dense_decoder_config():
    config = _fast_mtp_config()

    mtp_config = _get_mtp_block_config(config)

    assert mtp_config is not config
    assert mtp_config.n_routed_experts is None
    assert config.n_routed_experts == 64


def test_fast_mtp_weight_name_mapping():
    config = _fast_mtp_config()

    assert (
        _map_fast_mtp_weight_name(
            config, "mtp_module.heads.0.mtp_block.self_attn.q_proj.weight"
        )
        == "model.layers.12.self_attn.q_proj.weight"
    )
    assert (
        _map_fast_mtp_weight_name(config, "mtp_module.heads.0.eh_proj.weight")
        == "model.layers.12.eh_proj.weight"
    )
    assert (
        _map_fast_mtp_weight_name(config, "lm_head.weight")
        == "model.layers.12.shared_head.head.weight"
    )
    assert (
        _map_fast_mtp_weight_name(config, "model.norm.weight")
        == "model.layers.12.shared_head.norm.weight"
    )
    assert (
        _map_fast_mtp_weight_name(config, "model.embed_tokens.weight")
        == "model.layers.12.embed_tokens.weight"
    )
    assert (
        _map_fast_mtp_weight_name(config, "mtp_module.shared_head.norm.weight") is None
    )


def test_fast_mtp_maps_unshared_embedding():
    config = _fast_mtp_config(mtp_share_embedding_weights=False)

    assert (
        _map_fast_mtp_weight_name(config, "mtp_embed_tokens.weight")
        == "model.layers.12.embed_tokens.weight"
    )
