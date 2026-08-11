# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from transformers import GlmMoeDsaConfig, PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.configs.glm5v import Glm5vConfig
from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig


def test_glm5v_config_uses_glm_hidden_size_for_projector():
    quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
    }
    text_config = GlmMoeDsaConfig(
        hidden_size=6144,
        quantization_config=quantization_config,
    )
    vision_config = KimiK25VisionConfig(hidden_size=1152, mm_hidden_size=1152)

    config = Glm5vConfig(
        text_config=text_config,
        vision_config=vision_config,
        media_placeholder_token_id=154854,
    )

    assert config.hidden_size == 6144
    assert config.vision_config.mm_hidden_size == 6144
    assert config.media_placeholder_token_id == 154854
    assert config.quantization_config == quantization_config


def test_glm5v_config_preserves_nested_glm_dsa_dimensions():
    config = Glm5vConfig(
        text_config={
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "index_topk_freq": 4,
        }
    )

    assert config.text_config.qk_rope_head_dim == 64
    assert config.text_config.qk_head_dim == 256
    assert config.text_config.index_topk_freq == 4


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
