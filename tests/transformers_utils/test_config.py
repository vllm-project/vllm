# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import (
    patch_rope_parameters,
    try_get_generation_config,
)
from vllm.transformers_utils.configs.olmo_hybrid import OlmoHybridConfig


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


def test_olmo_hybrid_null_rope_theta_disables_rope():
    config = OlmoHybridConfig(rope_parameters={"rope_theta": None})

    assert config.rope_parameters is None
    patch_rope_parameters(config)
    assert config.rope_parameters is None


def test_olmo_hybrid_rope_parameters_default_rope_type():
    config = OlmoHybridConfig(rope_parameters={"rope_theta": 500000.0})

    assert config.rope_parameters == {
        "rope_theta": 500000.0,
        "rope_type": "default",
    }
    patch_rope_parameters(config)
    assert config.rope_parameters["rope_type"] == "default"
