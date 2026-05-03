# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from unittest.mock import patch

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import (
    maybe_override_with_speculators,
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


def test_remote_gguf_speculator_check_uses_repo_id_string():
    model = (
        "hesamation/"
        "Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:Q4_K_M"
    )

    with patch(
        "vllm.transformers_utils.config.PretrainedConfig.get_config_dict",
        return_value=({}, {}),
    ) as get_config_dict:
        resolved_model, tokenizer, speculative_config = maybe_override_with_speculators(
            model=model,
            tokenizer=None,
            trust_remote_code=False,
        )

    config_model = get_config_dict.call_args.args[0]
    assert config_model == (
        "hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
    )
    assert isinstance(config_model, str)
    assert resolved_model == model
    assert tokenizer is None
    assert speculative_config is None
