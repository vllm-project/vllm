# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.model_executor.layers.pooler import CLSPool, DispatchPooler, MeanPool
from vllm.model_executor.models.bert import BertEmbeddingModel
from vllm.model_executor.models.roberta import RobertaEmbeddingModel
from vllm.platforms import current_platform

MAX_MODEL_LEN = 128
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-base-en-v1.5")
REVISION = os.environ.get("REVISION", "main")

MODEL_NAME_ROBERTA = os.environ.get("MODEL_NAME", "intfloat/multilingual-e5-base")
REVISION_ROBERTA = os.environ.get("REVISION", "main")


@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Xformers backend is not supported on ROCm."
)
def test_model_loading_with_params(vllm_runner, monkeypatch):
    """
    Test parameter weight loading with tp>1.
    """
    # to use apply_model
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        model_name=MODEL_NAME,
        revision=REVISION,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
    ) as vllm_model:
        output = vllm_model.embed(
            "Write a short story about a robot that dreams for the first time.\n"
        )

        model_config = vllm_model.llm.llm_engine.model_config
        model_tokenizer = vllm_model.llm.llm_engine.tokenizer

        # asserts on the bert model config file
        assert model_config.encoder_config["max_seq_length"] == 512
        assert model_config.encoder_config["do_lower_case"]

        # asserts on the pooling config files
        assert model_config.pooler_config.pooling_type == "CLS"
        assert model_config.pooler_config.normalize

        # asserts on the tokenizer loaded
        assert model_config.tokenizer == "BAAI/bge-base-en-v1.5"
        assert model_tokenizer.model_max_length == 512

        def check_model(model):
            assert isinstance(model, BertEmbeddingModel)
            assert isinstance(pooler := model.pooler, DispatchPooler)
            assert isinstance(pooler.poolers_by_task["embed"].pooling, CLSPool)

        vllm_model.apply_model(check_model)

        assert output


@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Xformers backend is not supported on ROCm."
)
def test_roberta_model_loading_with_params(vllm_runner, monkeypatch):
    """
    Test parameter weight loading with tp>1.
    """
    # to use apply_model
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        model_name=MODEL_NAME_ROBERTA,
        revision=REVISION_ROBERTA,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
    ) as vllm_model:
        output = vllm_model.embed(
            "Write a short story about a robot that dreams for the first time.\n"
        )

        model_config = vllm_model.llm.llm_engine.model_config
        model_tokenizer = vllm_model.llm.llm_engine.tokenizer

        # asserts on the bert model config file
        assert model_config.encoder_config["max_seq_length"] == 512
        assert not model_config.encoder_config["do_lower_case"]

        # asserts on the pooling config files
        assert model_config.pooler_config.pooling_type == "MEAN"
        assert model_config.pooler_config.normalize

        # asserts on the tokenizer loaded
        assert model_config.tokenizer == "intfloat/multilingual-e5-base"
        assert model_tokenizer.model_max_length == 512

        def check_model(model):
            assert isinstance(model, RobertaEmbeddingModel)
            assert isinstance(pooler := model.pooler, DispatchPooler)
            assert isinstance(pooler.poolers_by_task["embed"].pooling, MeanPool)

        vllm_model.apply_model(check_model)

        assert output


@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Xformers backend is not supported on ROCm."
)
def test_facebook_roberta_model_loading_with_params(vllm_runner, monkeypatch):
    """
    Test loading roberta-base model with no lm_head.
    """
    # to use apply_model
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model_name = "FacebookAI/roberta-base"
    with vllm_runner(
        model_name=model_name, dtype="float16", max_model_len=MAX_MODEL_LEN
    ) as vllm_model:
        output = vllm_model.embed(
            "Write a short story about a robot that dreams for the first time.\n"
        )

        assert vllm_model.llm.llm_engine.model_config.tokenizer == model_name

        def check_model(model):
            assert isinstance(model, RobertaEmbeddingModel)
            assert not hasattr(model, "lm_head")
            assert isinstance(pooler := model.pooler, DispatchPooler)
            assert isinstance(pooler.poolers_by_task["embed"].pooling, CLSPool)

        vllm_model.apply_model(check_model)

        assert output
