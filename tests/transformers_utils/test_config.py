# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig


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


def test_qwen3_5_config_json_serializable():
    """Ensure Qwen3.5 configs can be serialized to JSON.

    Regression test: ignore_keys_at_rope_validation was a set literal,
    which causes TypeError in json.dumps when huggingface_hub >= 1.7.2
    triggers config validation via to_json_string().
    """
    config = Qwen3_5TextConfig()
    json.loads(config.to_json_string())


def test_qwen3_5_moe_config_json_serializable():
    """Ensure Qwen3.5-MoE configs can be serialized to JSON.

    Regression test: same set-vs-list issue as the dense config.
    """
    config = Qwen3_5MoeTextConfig()
    json.loads(config.to_json_string())
