# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import (
    get_safetensors_params_metadata,
    try_get_generation_config,
)
from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)


def test_glm5_next_accepts_deepseek_sparse_attention_layers():
    layer_types = ["linear_attention", "deepseek_sparse_attention"]

    config = Glm5NextTextConfig(
        num_hidden_layers=len(layer_types), layer_types=layer_types
    )

    assert config.layer_types == layer_types
    assert config.layers_block_type == ["linear_attention", "attention"]


def test_glm5_next_accepts_prebuilt_subconfigs():
    text_config = Glm5NextTextConfig(hidden_size=1024)
    vision_config = Glm5NextVisionConfig(hidden_size=768)

    config = Glm5NextConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    assert config.text_config is text_config
    assert config.vision_config is vision_config


@pytest.mark.parametrize(
    ("kwargs", "option"),
    [
        (
            {"index_topk": 2048, "index_dsa_use_layernorm": False},
            "index_dsa_use_layernorm",
        ),
        (
            {"index_topk": 2048, "index_kpool_compress": False},
            "index_kpool_compress",
        ),
        (
            {"index_topk": 2048, "index_kpool_always_select_tail": False},
            "index_kpool_always_select_tail",
        ),
        ({"hres_vwnstyle": False}, "hres_vwnstyle"),
        ({"mhc_no_norm_weight": True}, "mhc_no_norm_weight"),
    ],
)
def test_glm5_next_rejects_unimplemented_config_options(kwargs, option):
    with pytest.raises(NotImplementedError, match=option):
        Glm5NextTextConfig(**kwargs)


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_model_config_generation_fallback_forwards_code_revision():
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            generation_config="auto",
            hf_config_path=None,
            model="org/model",
            trust_remote_code=True,
            revision="model-pin",
            code_revision="code-pin",
            config_format="auto",
            hf_token=None,
        ),
    )

    with (
        patch.object(
            config_module.GenerationConfig,
            "from_pretrained",
            side_effect=OSError,
        ),
        patch.object(
            config_module,
            "get_config",
            return_value=PretrainedConfig(),
        ) as get_config,
    ):
        ModelConfig.try_get_generation_config(model_config)

    get_config.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
        revision="model-pin",
        code_revision="code-pin",
        config_format="auto",
        token=None,
    )


def test_safetensors_metadata_of_repo_without_safetensors():
    """A repo storing its weights in another format is an answer, not a failure,
    so it must not be retried."""
    from huggingface_hub.errors import LocalEntryNotFoundError, NotASafetensorsRepoError

    get_safetensors_metadata = MagicMock(
        side_effect=NotASafetensorsRepoError("not a safetensors repo")
    )
    api = SimpleNamespace(
        get_safetensors_metadata=get_safetensors_metadata,
        snapshot_download=MagicMock(side_effect=LocalEntryNotFoundError("no cache")),
    )

    with patch.object(config_module, "hf_api", lambda: api):
        assert get_safetensors_params_metadata("some/pytorch-only-model") == {}

    get_safetensors_metadata.assert_called_once()


def _write_granite_config(tmp_path, quantization_config=None):
    payload = {
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
    }
    if quantization_config is not None:
        payload["quantization_config"] = quantization_config
    (tmp_path / "config.json").write_text(json.dumps(payload))


def test_recipe_yaml_fp8_dynamic_maps_to_compressed_tensors():
    recipe = {
        "default_stage": {
            "default_modifiers": {
                "QuantizationModifier": {
                    "targets": ["Linear"],
                    "ignore": ["lm_head"],
                    "scheme": "FP8_DYNAMIC",
                }
            }
        }
    }
    mapped = config_module._recipe_to_quantization_config(recipe)
    assert mapped is not None
    assert mapped["quant_method"] == "compressed-tensors"
    assert mapped["format"] == "naive-quantized"
    assert mapped["ignore"] == ["lm_head"]
    group = mapped["config_groups"]["group_0"]
    assert group["targets"] == ["Linear"]
    assert group["weights"]["strategy"] == "channel"
    assert group["weights"]["type"] == "float"
    assert group["weights"]["num_bits"] == 8
    assert group["input_activations"]["strategy"] == "token"
    assert group["input_activations"]["dynamic"] is True


def test_recipe_yaml_unknown_or_shapeless_input_returns_none():
    assert config_module._recipe_to_quantization_config(None) is None
    assert config_module._recipe_to_quantization_config({}) is None
    assert (
        config_module._recipe_to_quantization_config(
            {
                "default_stage": {
                    "default_modifiers": {
                        "QuantizationModifier": {
                            "scheme": "W8A8",
                            "targets": ["Linear"],
                        }
                    }
                }
            }
        )
        is None
    )
    assert (
        config_module._recipe_to_quantization_config(
            {
                "default_stage": {
                    "default_modifiers": {
                        "QuantizationModifier": {"scheme": "FP8_DYNAMIC"}
                    }
                }
            }
        )
        is None
    )


def test_config_loads_quantization_from_recipe_yaml(tmp_path):
    _write_granite_config(tmp_path)
    (tmp_path / "recipe.yaml").write_text(
        "default_stage:\n"
        "  default_modifiers:\n"
        "    QuantizationModifier:\n"
        "      targets: [Linear]\n"
        "      ignore: [lm_head]\n"
        "      scheme: FP8_DYNAMIC\n"
    )

    config = config_module.get_config(tmp_path, trust_remote_code=False)
    assert config.quantization_config is not None
    assert config.quantization_config["quant_method"] == "compressed-tensors"


def test_config_json_quantization_wins_over_recipe_yaml(tmp_path):
    _write_granite_config(tmp_path, quantization_config={"quant_method": "awq"})
    (tmp_path / "recipe.yaml").write_text(
        "default_stage:\n"
        "  default_modifiers:\n"
        "    QuantizationModifier:\n"
        "      targets: [Linear]\n"
        "      scheme: FP8_DYNAMIC\n"
    )

    config = config_module.get_config(tmp_path, trust_remote_code=False)
    assert config.quantization_config == {"quant_method": "awq"}
