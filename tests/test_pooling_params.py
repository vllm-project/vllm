# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from tests.models.utils import EmbedModelInfo
from vllm import PoolingParams
from vllm.config import ModelConfig, PoolerConfig
from vllm.entrypoints.pooling.classify.protocol import ClassificationRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from vllm.entrypoints.pooling.pooling.protocol import PoolingRequest
from vllm.exceptions import VLLMValidationError

EMBEDDING_MODELS = [
    EmbedModelInfo("intfloat/multilingual-e5-small", is_matryoshka=False),
    EmbedModelInfo(
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        is_matryoshka=True,
        matryoshka_dimensions=[256],
    ),
]

classify_parameters = ["use_activation"]
embed_parameters = ["dimensions", "use_activation"]
step_pooling_parameters = ["step_tag_id", "returned_token_ids"]


@dataclass()
class MockModelConfig:
    pooler_config: PoolerConfig


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        (
            "normalize",
            False,
            "Parameter `normalize` was removed; use `use_activation` instead.",
        ),
        ("task", "score", "`score` task was removed; use `classify` instead."),
        (
            "task",
            "encode",
            "`encode` task was removed; use `token_embed` or `token_classify` instead.",
        ),
    ],
)
def test_removed_pooling_parameters(parameter: str, value: Any, message: str):
    data = {"input": "hello", parameter: value}
    for request_type in (EmbeddingRequest, ClassificationRequest, PoolingRequest):
        with pytest.raises(ValidationError, match=message) as exc_info:
            TypeAdapter(request_type).validate_python(data)
        assert len(exc_info.value.errors()) == 1

    with pytest.raises(ValidationError, match=message) as exc_info:
        TypeAdapter(PoolerConfig).validate_python({parameter: value})
    assert len(exc_info.value.errors()) == 1

    if parameter == "task":
        with pytest.raises(VLLMValidationError, match=message):
            PoolingParams(task=value)


def test_embed():
    task = "embed"
    model_config = MockModelConfig(pooler_config=PoolerConfig(seq_pooling_type="CLS"))

    pooling_params = PoolingParams(task=task, use_activation=None)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=True)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=False)
    pooling_params.verify(model_config)

    invalid_parameters = classify_parameters + step_pooling_parameters
    for p in set(invalid_parameters) - set(embed_parameters):
        with pytest.raises(ValueError):
            pooling_params = PoolingParams(task=task, **{p: True})
            pooling_params.verify(model_config)


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_dimensions(model_info: EmbedModelInfo):
    task = "embed"
    model_config = ModelConfig(
        model_info.name,
        tokenizer=model_info.name,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
    )

    pooling_params = PoolingParams(task=task, dimensions=None)
    pooling_params.verify(model_config)

    with pytest.raises(ValueError):
        pooling_params = PoolingParams(task=task, dimensions=1)
        pooling_params.verify(model_config)

    if model_info.is_matryoshka:
        assert model_info.matryoshka_dimensions is not None
        pooling_params = PoolingParams(
            task=task, dimensions=model_info.matryoshka_dimensions[0]
        )
        pooling_params.verify(model_config)


@dataclass()
class MockMatryoshkaModelConfig:
    pooler_config: PoolerConfig
    is_matryoshka: bool = True
    matryoshka_dimensions: list[int] | None = None
    served_model_name: str = "mock-matryoshka-model"
    embedding_size: int = 32


def test_embed_dimensions_matryoshka_without_list_upper_bound():
    task = "embed"
    model_config = MockMatryoshkaModelConfig(
        pooler_config=PoolerConfig(seq_pooling_type="CLS"),
        matryoshka_dimensions=None,
        embedding_size=32,
    )

    PoolingParams(task=task, dimensions=16).verify(model_config)

    with pytest.raises(ValueError):
        PoolingParams(task=task, dimensions=64).verify(model_config)


@pytest.mark.parametrize("task", ["classify"])
def test_classify(task):
    model_config = MockModelConfig(pooler_config=PoolerConfig(seq_pooling_type="CLS"))

    pooling_params = PoolingParams(task=task, use_activation=None)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=True)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=False)
    pooling_params.verify(model_config)

    invalid_parameters = embed_parameters + step_pooling_parameters
    for p in set(invalid_parameters) - set(classify_parameters):
        with pytest.raises(ValueError):
            pooling_params = PoolingParams(task=task, **{p: True})
            pooling_params.verify(model_config)


@pytest.mark.parametrize("pooling_type", ["ALL", "STEP"])
def test_token_embed(pooling_type: str):
    task = "token_embed"
    model_config = MockModelConfig(
        pooler_config=PoolerConfig(tok_pooling_type=pooling_type)
    )

    pooling_params = PoolingParams(task=task, use_activation=None)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=True)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=False)
    pooling_params.verify(model_config)

    invalid_parameters = classify_parameters
    if pooling_type != "STEP":
        invalid_parameters = classify_parameters + step_pooling_parameters

    for p in set(invalid_parameters) - set(embed_parameters):
        with pytest.raises(ValueError):
            pooling_params = PoolingParams(task=task, **{p: True})
            pooling_params.verify(model_config)


@pytest.mark.parametrize("pooling_type", ["ALL", "STEP"])
def test_token_classify(pooling_type: str):
    task = "token_classify"
    model_config = MockModelConfig(
        pooler_config=PoolerConfig(tok_pooling_type=pooling_type)
    )

    pooling_params = PoolingParams(task=task, use_activation=None)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=True)
    pooling_params.verify(model_config)

    pooling_params = PoolingParams(task=task, use_activation=False)
    pooling_params.verify(model_config)

    invalid_parameters = embed_parameters
    if pooling_type != "STEP":
        invalid_parameters = embed_parameters + step_pooling_parameters

    for p in set(invalid_parameters) - set(classify_parameters):
        with pytest.raises(ValueError):
            pooling_params = PoolingParams(task=task, **{p: True})
            pooling_params.verify(model_config)
