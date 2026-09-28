# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: SIM117

import pytest

from ...utils import EmbedModelInfo

MODELS = [
    EmbedModelInfo(
        "nomic-ai/nomic-embed-text-v1",
        # Fixme:
        #  Update nomic-embed code to support the latest
        #  HF version and remove revision set.
        revision="720244025c1a7e15661a174c63cce63c8218e52b",
    ),
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
        model_info.name,
        revision=model_info.revision,
        runner="pooling",
        max_model_len=None,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        if model_info.name == "nomic-ai/nomic-embed-text-v2-moe":
            # For nomic-embed-text-v2-moe the length is set to 512
            # by sentence_bert_config.json.
            assert model_config.max_model_len == 512
        if model_info.name == "nomic-ai/nomic-embed-text-v1":
            assert model_config.max_model_len == 8192


@pytest.mark.parametrize("model_info", MODELS)
def test_set_max_model_len_legal(model_info, vllm_runner):
    # set max_model_len <= 512
    with vllm_runner(
        model_info.name,
        revision=model_info.revision,
        runner="pooling",
        max_model_len=256,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.max_model_len == 256

    # For nomic-embed-text-v2-moe the length is set to 512
    # by sentence_bert_config.json.
    if model_info.name == "nomic-ai/nomic-embed-text-v2-moe":
        with pytest.raises(ValueError):
            with vllm_runner(
                model_info.name,
                revision=model_info.revision,
                runner="pooling",
                max_model_len=1024,
            ):
                pass
        return

    # set 512 < max_model_len <= 2048
    with vllm_runner(
        model_info.name,
        revision=model_info.revision,
        runner="pooling",
        max_model_len=1024,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.max_model_len == 1024

    # set max_model_len > 2048
    with vllm_runner(
        model_info.name,
        revision=model_info.revision,
        runner="pooling",
        max_model_len=4096,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.max_model_len == 4096


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
        model_info.name,
        revision=model_info.revision,
        runner="pooling",
        max_model_len=None,
        hf_overrides=hf_overrides,
    ):
        pass
