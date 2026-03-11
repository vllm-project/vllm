# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from pathlib import Path
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


@patch("vllm.transformers_utils.config.PretrainedConfig.get_config_dict")
@patch("vllm.transformers_utils.config.check_gguf_file", return_value=True)
def test_maybe_override_with_speculators_uses_hf_config_path_for_local_gguf(
    mock_check_gguf_file,
    mock_get_config_dict,
):
    mock_get_config_dict.return_value = ({}, {})
    model = "/tmp/model.gguf"
    tokenizer = "/tmp/tokenizer"

    resolved = maybe_override_with_speculators(
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=False,
        hf_config_path="/tmp/hf-config",
    )

    assert resolved == (model, tokenizer, None)
    mock_check_gguf_file.assert_not_called()
    mock_get_config_dict.assert_called_once()
    args, kwargs = mock_get_config_dict.call_args
    assert args[0] == "/tmp/hf-config"
    assert "gguf_file" not in kwargs


@patch("vllm.transformers_utils.config.PretrainedConfig.get_config_dict")
@patch("vllm.transformers_utils.config.check_gguf_file", return_value=True)
def test_maybe_override_with_speculators_keeps_gguf_file_without_hf_config_path(
    mock_check_gguf_file,
    mock_get_config_dict,
):
    mock_get_config_dict.return_value = ({}, {})

    maybe_override_with_speculators(
        model="/tmp/model.gguf",
        tokenizer="/tmp/tokenizer",
        trust_remote_code=False,
    )

    mock_check_gguf_file.assert_called_once_with("/tmp/model.gguf")
    mock_get_config_dict.assert_called_once()
    args, kwargs = mock_get_config_dict.call_args
    assert args[0] == Path("/tmp")
    assert kwargs["gguf_file"] == "model.gguf"
