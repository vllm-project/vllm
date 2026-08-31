# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig


def _make_dflash2_config(**overrides) -> dict:
    config = {
        "speculators_model_type": "dflash2",
        "aux_hidden_state_layer_ids": [2, 5],
        "mask_token_id": 31,
        "conv_kernel_size": 2,
        "conv_group_size": 4,
        "selector_rank": 8,
        "selector_top_k": 4,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
    }
    config.update(overrides)
    return config


def test_dflash2_moe_geometry_reaches_draft_model_config() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dflash2_config(
            draft_ffn_type="moe",
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=12,
            shared_expert_intermediate_size=10,
            norm_topk_prob=False,
        )
    )

    assert config["draft_ffn_type"] == "moe"
    assert config["num_experts"] == 8
    assert config["num_experts_per_tok"] == 2
    assert config["moe_intermediate_size"] == 12
    assert config["shared_expert_intermediate_size"] == 10
    assert config["norm_topk_prob"] is False
    assert "draft_ffn_type" not in config["dflash_config"]


def test_dflash2_ffn_defaults_to_dense() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dflash2_config()
    )

    assert config["draft_ffn_type"] == "dense"


def test_dflash2_moe_requires_complete_geometry() -> None:
    with pytest.raises(ValueError, match="missing.*num_experts_per_tok"):
        SpeculatorsConfig.extract_transformers_pre_trained_config(
            _make_dflash2_config(
                draft_ffn_type="moe",
                num_experts=8,
                moe_intermediate_size=12,
                shared_expert_intermediate_size=10,
            )
        )
