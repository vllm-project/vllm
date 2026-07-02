# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from unittest.mock import patch

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import get_pooling_config, try_get_generation_config


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


@patch("vllm.transformers_utils.config.file_or_path_exists", return_value=True)
@patch(
    "vllm.transformers_utils.config.get_hf_file_to_dict",
    side_effect=lambda name, *_args, **_kwargs: {
        "modules.json": [
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
        ],
        "1_Pooling/config.json": {
            "embedding_dimension": 384,
            "pooling_mode": "mean",
            "include_prompt": True,
        },
    }.get(name),
)
def test_get_pooling_config_compact_schema(_mock_hf, _mock_exists):
    get_pooling_config.cache_clear()
    config = get_pooling_config("dummy-model")
    assert config is not None
    assert config["seq_pooling_type"] == "MEAN"
