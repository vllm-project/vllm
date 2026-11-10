# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vllm import PoolingParams
from vllm.entrypoints.pooling import factories
from vllm.entrypoints.pooling.factories import init_pooling_io_processors
from vllm.entrypoints.pooling.pooling.io_processor import (
    PluginWithIOProcessorPlugins,
    UnsupportedCombinedTaskIOProcessor,
)
from vllm.entrypoints.pooling.token_embed.serving import ServingTokenEmbedding
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.exceptions import VLLMValidationError
from vllm.pooling_params import LateChunkingParams


def _bge_m3_config(io_processor_plugin=None):
    model_config = MagicMock()
    model_config.get_pooling_task.return_value = "embed&token_classify"
    model_config.io_processor_plugin = io_processor_plugin
    model_config.hf_config.to_dict.return_value = {}
    model_config.architecture = "BgeM3EmbeddingModel"

    vllm_config = MagicMock(model_config=model_config)
    renderer = MagicMock()
    renderer._executor = MagicMock()
    chat_template_config = MagicMock(
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
    )
    return vllm_config, renderer, chat_template_config


def test_combined_task_without_plugin_uses_rejection_processor():
    vllm_config, renderer, chat_template_config = _bge_m3_config()

    processors = init_pooling_io_processors(
        supported_tasks=("embed", "embed&token_classify"),
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    assert processors.keys() == {"embed&token_classify"}
    assert isinstance(
        processors["embed&token_classify"], UnsupportedCombinedTaskIOProcessor
    )


def test_combined_task_with_plugin_uses_plugin_processor(monkeypatch):
    vllm_config, renderer, chat_template_config = _bge_m3_config("bge_m3_sparse_plugin")
    monkeypatch.setattr(factories, "has_io_processor", lambda *_: True)
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.pooling.io_processor.get_io_processor",
        lambda *_: MagicMock(),
    )

    processors = init_pooling_io_processors(
        supported_tasks=("embed", "embed&token_classify"),
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )

    assert processors.keys() == {"embed&token_classify", "plugin"}
    assert isinstance(processors["plugin"], PluginWithIOProcessorPlugins)


def test_combined_task_plain_pooling_request_has_actionable_error(monkeypatch):
    from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest
    from vllm.entrypoints.pooling.pooling.serving import ServingPooling

    vllm_config, renderer, chat_template_config = _bge_m3_config("bge_m3_sparse_plugin")
    monkeypatch.setattr(factories, "has_io_processor", lambda *_: True)
    monkeypatch.setattr(
        "vllm.entrypoints.pooling.pooling.io_processor.get_io_processor",
        lambda *_: MagicMock(),
    )

    engine_client = MagicMock(renderer=renderer, vllm_config=vllm_config)
    models = MagicMock(model_config=vllm_config.model_config)
    serving = ServingPooling(
        engine_client,
        models,
        supported_tasks=("embed", "embed&token_classify"),
        request_logger=None,
        chat_template_config=chat_template_config,
    )
    request = PoolingCompletionRequest(model="BAAI/bge-m3", input=["hola"])

    assert serving.io_processors.keys() == {"embed&token_classify", "plugin"}
    io_processor = serving.get_io_processor(request)
    with pytest.raises(ValueError, match="plugin request with a 'data' field"):
        io_processor.create_pooling_params(request)


@pytest.mark.parametrize("task", ["embed", "token_embed"])
def test_embedding_route_validates_the_configured_task(task):
    """Invalid chunk requests must never fall back to the ordinary schema."""
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    init_exception_handler(app)
    model_config = MagicMock()
    model_config.get_pooling_task.return_value = task
    model_config.is_cross_encoder = False
    factories.register_pooling_api_routers(app, ("embed", "token_embed"), model_config)
    received = []

    async def handle(request, raw_request):
        received.append(request)
        return JSONResponse({"task": task})

    if task == "token_embed":
        app.state.serving_token_embedding = handle
    else:
        app.state.serving_embedding = handle

    with TestClient(app) as client:
        if task == "embed":
            assert client.post("/v1/embeddings", json={"input": "text"}).json() == {
                "task": "embed"
            }
            assert client.post("/v2/embed", json={"texts": ["text"]}).status_code == 200
            return

        assert client.post("/v2/embed", json={"texts": ["text"]}).status_code == 404
        bad_requests: list[dict[str, object]] = [
            {"late_chunking_params": {}},
            {"late_chunking_params": {"chunk_size": 2, "metadata": {}}},
            {"encoding_format": "base64"},
            {"dimensions": 128},
        ]
        bad_requests.extend(
            {"late_chunking_params": {"chunk_size": value}}
            for value in (0, -1, True, 1.5, "2", 2**63, 2**64)
        )
        for overrides in bad_requests:
            body: dict[str, object] = {"input": "text", **overrides}
            if "late_chunking_params" not in overrides and overrides:
                body["late_chunking_params"] = {"chunk_size": 2}
            response = client.post("/v1/embeddings", json=body)
            assert response.status_code == 400, response.text
        assert not received

        for body in (
            {"input": "text"},
            {"input": ["text", "another text"], "late_chunking_params": None},
            {"input": [1, 2]},
        ):
            response = client.post("/v1/embeddings", json=body)
            assert response.json() == {"task": "token_embed"}
            params = received[-1].to_pooling_params()
            assert params.task == "token_embed"
            assert params.late_chunking_params is None

        for chunk_size in (1, 1024, 2**63 - 1):
            response = client.post(
                "/v1/embeddings",
                json={
                    "input": "text",
                    "late_chunking_params": {"chunk_size": chunk_size},
                },
            )
            assert response.json() == {"task": "token_embed"}
            params = received[-1].to_pooling_params()
            assert params.task == "token_embed"
            assert params.late_chunking_params.chunk_size == chunk_size


def test_token_embedding_keeps_model_specific_chunking_rejection():
    vllm_config, renderer, chat_template_config = _bge_m3_config()
    vllm_config.model_config.get_pooling_task.return_value = "token_embed"
    vllm_config.model_config.architecture = "JinaForRanking"
    serving = object.__new__(ServingTokenEmbedding)
    processor = serving.init_io_processor(
        vllm_config=vllm_config,
        renderer=renderer,
        chat_template_config=chat_template_config,
    )
    with pytest.raises(VLLMValidationError, match="JinaForRanking.*late chunking"):
        processor.render(
            {"params": PoolingParams(late_chunking_params=LateChunkingParams(2))}
        )
    renderer.render_cmpl.assert_not_called()
