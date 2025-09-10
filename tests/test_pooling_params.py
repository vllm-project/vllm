# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import EmbedModelInfo
from vllm import PoolingParams
from vllm.config import ModelConfig

EMBEDDING_MODELS = [
    EmbedModelInfo("intfloat/multilingual-e5-small", is_matryoshka=False),
    EmbedModelInfo("Snowflake/snowflake-arctic-embed-m-v1.5",
                   is_matryoshka=True,
                   matryoshka_dimensions=[256]),
]


def test_task():
    pooling_params = PoolingParams()
    pooling_params.verify(task="score")

    pooling_params = PoolingParams(task="score")
    pooling_params.verify(task="score")

    with pytest.raises(ValueError):
        pooling_params.verify(task="encode")


def test_embed():
    task = "embed"
    pooling_params = PoolingParams(normalize=None)
    pooling_params.verify(task=task)

    pooling_params = PoolingParams(normalize=True)
    pooling_params.verify(task=task)

    pooling_params = PoolingParams(normalize=False)
    pooling_params.verify(task=task)

    invalid_parameters = ["activation", "softmax"]
    for p in invalid_parameters:
        with pytest.raises(ValueError):
            pooling_params = PoolingParams(**{p: True})
            pooling_params.verify(task=task)


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_dimensions(model_info: EmbedModelInfo):
    task = "embed"
    model_config = ModelConfig(
        model_info.name,
        task="auto",
        tokenizer=model_info.name,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
    )

    pooling_params = PoolingParams(dimensions=None)
    pooling_params.verify(task=task, model_config=model_config)

    with pytest.raises(ValueError):
        pooling_params = PoolingParams(dimensions=1)
        pooling_params.verify(task=task, model_config=model_config)

    if model_info.is_matryoshka:
        assert model_info.matryoshka_dimensions is not None
        pooling_params = PoolingParams(
            dimensions=model_info.matryoshka_dimensions[0])
        pooling_params.verify(task=task, model_config=model_config)


@pytest.mark.parametrize("task", ["score", "classify"])
def test_classify(task):
    pooling_params = PoolingParams(activation=None)
    pooling_params.verify(task=task)

    pooling_params = PoolingParams(activation=True)
    pooling_params.verify(task=task)

    pooling_params = PoolingParams(activation=False)
    pooling_params.verify(task=task)

    invalid_parameters = ["dimensions", "normalize", "softmax"]
    for p in invalid_parameters:
        with pytest.raises(ValueError):
            pooling_params = PoolingParams(**{p: True})
            pooling_params.verify(task=task)


def test_encode():
    task = "encode"
    pooling_params = PoolingParams(softmax=None)
    pooling_params.verify(task=task)

    pooling_params = PoolingParams(softmax=True)
    pooling_params.verify(task=task)

    pooling_params = PoolingParams(softmax=False)
    pooling_params.verify(task=task)

    invalid_parameters = ["dimensions", "normalize", "activation"]
    for p in invalid_parameters:
        with pytest.raises(ValueError):
            pooling_params = PoolingParams(**{p: True})
            pooling_params.verify(task=task)
