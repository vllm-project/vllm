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

from transformers import PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import (
    get_config,
    get_safetensors_params_metadata,
    try_get_generation_config,
)


def test_longcat_config_without_model_type_uses_builtin_transformers(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["LongcatFlashForCausalLM"],
                "expert_ffn_hidden_size": 1024,
                "num_hidden_layers": 56,
                "num_layers": 14,
                "auto_map": {
                    "AutoConfig": "configuration_longcat_flash.LongcatFlashConfig",
                    "AutoModel": "modeling_longcat_flash.LongcatFlashModel",
                },
            }
        ),
        encoding="utf-8",
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert type(config).__name__ == "LongcatFlashConfig"
    assert config.model_type == "longcat_flash"
    assert config.auto_map == {}
    assert config.num_hidden_layers == 28
    assert config.moe_intermediate_size == 1024


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
