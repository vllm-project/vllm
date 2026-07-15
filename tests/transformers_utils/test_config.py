# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from unittest.mock import MagicMock, call, patch

from transformers import GenerationConfig, PretrainedConfig

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import (
    _generation_config_from_hf_configs,
    try_get_generation_config,
)


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


def test_try_get_generation_config_uses_loaded_hf_config():
    hf_config = MagicMock(spec=PretrainedConfig)
    hf_text_config = MagicMock(spec=PretrainedConfig)
    expected = GenerationConfig(eos_token_id=42)

    with patch(
        "vllm.transformers_utils.config.GenerationConfig.from_pretrained",
        side_effect=OSError("missing generation_config.json"),
    ), patch(
        "vllm.transformers_utils.config._generation_config_from_hf_configs",
        return_value=expected,
    ) as from_hf_configs, patch(
        "vllm.transformers_utils.config.get_config",
    ) as get_config:
        result = try_get_generation_config(
            "deepseek-ai/deepseek-vl2-tiny",
            trust_remote_code=False,
            hf_config=hf_config,
            hf_text_config=hf_text_config,
        )

    assert result is expected
    get_config.assert_not_called()
    from_hf_configs.assert_called_once_with(hf_config, hf_text_config)


def test_generation_config_from_hf_configs_falls_back_to_text_config():
    hf_config = MagicMock(spec=PretrainedConfig)
    hf_text_config = MagicMock(spec=PretrainedConfig)
    expected = GenerationConfig(eos_token_id=7)

    with patch(
        "vllm.transformers_utils.config.GenerationConfig.from_model_config",
        side_effect=[ValueError("invalid full config"), expected],
    ) as from_model_config:
        result = _generation_config_from_hf_configs(hf_config, hf_text_config)

    assert result is expected
    assert from_model_config.call_args_list == [
        call(hf_config),
        call(hf_text_config),
    ]
