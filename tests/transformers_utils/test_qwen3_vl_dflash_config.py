# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.transformers_utils.config import (
    get_config,
    get_hf_text_config,
    uses_mrope,
)
from vllm.transformers_utils.configs.qwen3_vl_dflash import (
    Qwen3VLDFlashConfig,
)

pytestmark = pytest.mark.skip_global_cleanup


def _qwen3_vl_dflash_config() -> dict:
    return {
        "architectures": ["Qwen3VLForConditionalGenerationDFlash"],
        "auto_map": {"AutoConfig": "qwen3_vl_dflash.Qwen3VLDFlashConfig"},
        "dflash_config": {
            "block_size": 16,
            "dtype": "bfloat16",
            "enable_confidence_head": True,
            "layer_types": ["full_attention", "full_attention"],
            "markov_head_type": "vanilla",
            "markov_rank": 256,
            "mask_token_id": 151669,
            "model_type": "dflash",
            "num_hidden_layers": 2,
            "num_target_feature_layers": 5,
            "num_target_layers": 28,
            "sliding_window": None,
            "target_layer_ids": [1, 7, 13, 19, 25],
        },
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "model_type": "qwen3_vl_dflash",
        "pad_token_id": 151643,
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
            "rope_theta": 5000000,
            "tie_word_embeddings": True,
            "use_cache": True,
            "vocab_size": 151936,
        },
        "tie_word_embeddings": True,
        "vision_config": {
            "hidden_size": 1024,
            "model_type": "qwen3_vl",
            "out_hidden_size": 2048,
        },
    }


def test_qwen3_vl_dflash_config_is_normalized_without_remote_code(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_qwen3_vl_dflash_config()), encoding="utf-8"
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, Qwen3VLDFlashConfig)
    assert config.architectures == ["Qwen3VLDSparkModel"]
    assert config.source_architectures == ["Qwen3VLForConditionalGenerationDFlash"]
    assert config.model_type == "qwen3_vl_dflash"
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 2
    assert config.layer_types == ["full_attention", "full_attention"]
    assert config.block_size == config.dspark_block_size == config.n_predict == 16
    assert config.markov_rank == 256
    assert config.target_layer_ids == [1, 7, 13, 19, 25]
    assert config.eagle_aux_hidden_state_layer_ids == [2, 8, 14, 20, 26]
    assert "mrope_section" not in config.rope_parameters
    assert "mrope_interleaved" not in config.rope_parameters
    assert config.rope_parameters["rope_theta"] == 5000000
    assert config.source_text_config["rope_scaling"]["mrope_section"] == [24, 20, 20]
    assert config.get_text_config() is config
    assert get_hf_text_config(config) is config
    assert not uses_mrope(config)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("num_target_layers", 48, "num_target_layers"),
        ("num_target_feature_layers", 4, "num_target_feature_layers"),
        ("num_hidden_layers", 3, "layer_types"),
    ],
)
def test_qwen3_vl_dflash_config_rejects_inconsistent_geometry(field, value, error):
    raw_config = _qwen3_vl_dflash_config()
    raw_config["dflash_config"][field] = value

    with pytest.raises(ValueError, match=error):
        Qwen3VLDFlashConfig(**raw_config)
