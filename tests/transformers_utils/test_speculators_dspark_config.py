# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_dspark_config(architecture: str) -> dict:
    return {
        "speculators_model_type": "dspark",
        "architectures": [architecture],
        "sample_from_anchor": True,
        "use_aux_hidden_state": True,
        "draft_vocab_size": 32,
        "target_hidden_size": 16,
        "mask_token_id": 31,
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "block_size": 3,
        "aux_hidden_state_layer_ids": [2, 5],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
    }


def test_dspark_updater_preserves_qwen3_omni_contract() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("Qwen3OmniDSparkModel")
    )

    assert config["architectures"] == ["Qwen3OmniDSparkModel"]
    assert config["sample_from_anchor"] is True
    assert config["dspark_bonus_anchor"] is False
    assert config["use_aux_hidden_state"] is True
    assert config["target_layer_ids"] == [1, 4]


def test_dspark_updater_keeps_legacy_qwen3_checkpoint_loadable() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("DSparkSpeculator")
    )

    assert config["architectures"] == ["Qwen3DSparkModel"]


def test_eagle3_extract_keeps_outer_quantization_config() -> None:
    """Quant metadata lives on the speculators config, not the layer config."""
    outer = {
        "speculators_model_type": "eagle3",
        "draft_vocab_size": 32000,
        "transformer_layer_config": {
            "model_type": "llama",
            "hidden_size": 5376,
            "vocab_size": 262144,
        },
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "exclude_modules": ["lm_head", "embed_tokens"],
        },
    }

    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer)

    assert config["architectures"] == ["Eagle3LlamaForCausalLM"]
    assert config["quantization_config"]["exclude_modules"] == [
        "lm_head",
        "embed_tokens",
    ]
    assert "quantization_config" not in outer["transformer_layer_config"]
    assert config["quantization_config"] is not outer["quantization_config"]


def test_dspark_updater_maps_bonus_anchor_semantics() -> None:
    outer_config = _make_dspark_config("Qwen3DSparkModel")
    outer_config["sample_from_anchor"] = False

    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer_config)

    assert config["sample_from_anchor"] is False
    assert config["dspark_bonus_anchor"] is True
