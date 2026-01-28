# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: SIM117
from typing import Any

import pytest

from ...utils import EmbedModelInfo

MODELS = [
    EmbedModelInfo("nomic-ai/nomic-embed-text-v1"),
    # EmbedModelInfo("nomic-ai/nomic-embed-text-v1.5"),
    # EmbedModelInfo("nomic-ai/CodeRankEmbed"),
    EmbedModelInfo("nomic-ai/nomic-embed-text-v2-moe"),
    # EmbedModelInfo("Snowflake/snowflake-arctic-embed-m-long"),
]

rope_theta = 1000
factor = 4.0
original_max_position_embeddings = 2048
max_model_len = int(original_max_position_embeddings * factor)


@pytest.mark.parametrize("model_info", MODELS)
def test_default(model_info, vllm_runner):
    with vllm_runner(
        model_info.name, runner="pooling", max_model_len=None
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        if model_info.name == "nomic-ai/nomic-embed-text-v2-moe":
            # For nomic-embed-text-v2-moe the length is set to 512
            # by sentence_bert_config.json.
            assert model_config.max_model_len == 512
        else:
            assert model_config.max_model_len == original_max_position_embeddings


@pytest.mark.parametrize("model_info", MODELS)
def test_set_max_model_len_legal(model_info, vllm_runner):
    # set max_model_len <= 512
    with vllm_runner(
        model_info.name, runner="pooling", max_model_len=256
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.max_model_len == 256

    # set 512 < max_model_len <= 2048
    if model_info.name == "nomic-ai/nomic-embed-text-v2-moe":
        # For nomic-embed-text-v2-moe the length is set to 512
        # by sentence_bert_config.json.
        with pytest.raises(ValueError):
            with vllm_runner(model_info.name, runner="pooling", max_model_len=1024):
                pass
    else:
        with vllm_runner(
            model_info.name, runner="pooling", max_model_len=1024
        ) as vllm_model:
            model_config = vllm_model.llm.llm_engine.model_config
            assert model_config.max_model_len == 1024


@pytest.mark.parametrize("model_info", MODELS)
def test_set_max_model_len_illegal(model_info, vllm_runner):
    # set max_model_len > 2048
    with pytest.raises(ValueError):
        with vllm_runner(model_info.name, runner="pooling", max_model_len=4096):
            pass

    # set max_model_len > 2048 by hf_overrides
    hf_overrides = {"max_model_len": 4096}
    with pytest.raises(ValueError):
        with vllm_runner(
            model_info.name,
            runner="pooling",
            max_model_len=None,
            hf_overrides=hf_overrides,
        ):
            pass


@pytest.mark.parametrize("model_info", MODELS)
def test_use_rope_scaling_legal(model_info, vllm_runner):
    hf_overrides = {
        "rope_parameters": {
            "rope_theta": rope_theta,
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original_max_position_embeddings,
        },
        "max_model_len": max_model_len,
    }

    with vllm_runner(
        model_info.name, runner="pooling", max_model_len=None, hf_overrides=hf_overrides
    ):
        pass


@pytest.mark.parametrize("model_info", MODELS)
def test_use_rope_scaling_illegal(model_info, vllm_runner):
    hf_overrides: dict[str, Any] = {
        "rope_parameters": {
            "rope_theta": rope_theta,
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original_max_position_embeddings,
        },
    }
    # illegal max_model_len
    with pytest.raises(ValueError):
        with vllm_runner(
            model_info.name,
            runner="pooling",
            max_model_len=max_model_len + 1,
            hf_overrides=hf_overrides,
        ):
            pass

    hf_overrides = {
        "rope_parameters": {
            "rope_theta": rope_theta,
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original_max_position_embeddings,
        },
        "max_model_len": max_model_len + 1,
    }
    # illegal max_model_len by hf_overrides
    with pytest.raises(ValueError):
        with vllm_runner(
            model_info.name,
            runner="pooling",
            max_model_len=None,
            hf_overrides=hf_overrides,
        ):
            pass
