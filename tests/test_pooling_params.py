# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from pydantic import TypeAdapter, ValidationError

from tests.models.utils import EmbedModelInfo
from vllm import PoolingParams
from vllm.config import ModelConfig, PoolerConfig
from vllm.entrypoints.pooling.classify.protocol import ClassificationRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from vllm.entrypoints.pooling.pooling.protocol import PoolingRequest
from vllm.exceptions import VLLMValidationError
from vllm.pooling_params import LateChunkingParams

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
        with pytest.raises(VLLMValidationError, match=message):
            TypeAdapter(request_type).validate_python(data)

    # PoolerConfig still raises bare ValueError for `normalize`
    # (wrapped to ValidationError by Pydantic), but `check_removed_pooling_task`
    # raises VLLMValidationError for removed tasks.
    if parameter == "normalize":
        with pytest.raises(ValidationError, match=message) as exc_info:
            TypeAdapter(PoolerConfig).validate_python({parameter: value})
        assert len(exc_info.value.errors()) == 1
    else:
        with pytest.raises(VLLMValidationError, match=message):
            TypeAdapter(PoolerConfig).validate_python({parameter: value})

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
        with pytest.raises(VLLMValidationError):
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

    with pytest.raises(VLLMValidationError):
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

    with pytest.raises(VLLMValidationError):
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
        with pytest.raises(VLLMValidationError):
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
        with pytest.raises(VLLMValidationError):
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
        with pytest.raises(VLLMValidationError):
            pooling_params = PoolingParams(task=task, **{p: True})
            pooling_params.verify(model_config)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
def test_late_chunk_size_rejects_non_positive_integers(value):
    with pytest.raises(VLLMValidationError, match="positive integer"):
        PoolingParams(late_chunking_params=LateChunkingParams(chunk_size=value))


def _late_chunking_model_config():
    return SimpleNamespace(
        architecture="NomicBertModel",
        model_impl="auto",
        is_matryoshka=False,
        hf_config=SimpleNamespace(),
        pooler_config=PoolerConfig(seq_pooling_type="MEAN", tok_pooling_type="ALL"),
    )


def test_late_chunking_params_preserve_defaults_clone_and_wire_format():
    model_config = _late_chunking_model_config()
    params = PoolingParams(
        task="token_embed", late_chunking_params=LateChunkingParams(chunk_size=3)
    )
    params.verify(model_config)
    assert params.skip_reading_prefix_cache is True
    assert params.use_activation is True
    # With caching disabled by the frontend, this override is harmless.
    explicit_cache_flag = PoolingParams(
        task="token_embed",
        late_chunking_params=LateChunkingParams(chunk_size=3),
        skip_reading_prefix_cache=False,
    )
    explicit_cache_flag.verify(model_config)
    assert explicit_cache_flag.skip_reading_prefix_cache is False
    clone = params.clone()
    assert clone.late_chunking_params is not None
    clone.late_chunking_params.chunk_size = 7
    assert params.late_chunking_params is not None
    assert params.late_chunking_params.chunk_size == 3
    assert (
        msgspec.msgpack.decode(msgspec.msgpack.encode(params), type=PoolingParams)
        == params
    )
    # A message written before the appended field still uses the old default.
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(PoolingParams()))
    assert msgspec.convert(wire[:-1], type=PoolingParams).late_chunking_params is None


@pytest.mark.parametrize(
    "task", ["embed", "classify", "token_classify", "plugin", None]
)
def test_late_chunking_rejects_other_tasks(task):
    with pytest.raises(VLLMValidationError, match="requires token_embed"):
        PoolingParams(
            task=task, late_chunking_params=LateChunkingParams(chunk_size=2)
        ).verify(_late_chunking_model_config())


@pytest.mark.parametrize(
    "override",
    [
        {"architecture": "BertModel"},
        {"model_impl": "transformers"},
        {"is_matryoshka": True},
        {"seq_pooling_type": "CLS"},
        {"tok_pooling_type": "STEP"},
        {"enable_chunked_processing": True},
        {"num_experts": 8},
    ],
)
def test_late_chunking_rejects_unverified_model_contracts(override):
    model_config = _late_chunking_model_config()
    for key, value in override.items():
        target = (
            model_config.hf_config
            if key == "num_experts"
            else model_config.pooler_config
            if hasattr(model_config.pooler_config, key)
            else model_config
        )
        setattr(target, key, value)
    with pytest.raises(VLLMValidationError, match="dense NomicBertModel"):
        PoolingParams(
            task="token_embed", late_chunking_params=LateChunkingParams(chunk_size=2)
        ).verify(model_config)


def test_late_chunking_rejects_dimension_reduction():
    with pytest.raises(VLLMValidationError, match="does not support"):
        PoolingParams(
            task="token_embed",
            late_chunking_params=LateChunkingParams(chunk_size=2),
            dimensions=4,
        ).verify(_late_chunking_model_config())
