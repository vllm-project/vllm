# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

import openai  # use the official async_client for correctness check
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...utils import RemoteOpenAIServer
from .conftest import MODEL_NAME_SMOLLM


class _FakeLoRAModelHandler:
    def __init__(self):
        self.load_requests = []

    async def load_lora_adapter(self, request):
        from vllm.entrypoints.utils import create_error_response

        self.load_requests.append(request)
        return create_error_response(
            "LoRA adapter nonexistent-adapter does not exist",
            "NotFoundError",
            HTTPStatus.NOT_FOUND,
        )

    async def unload_lora_adapter(self, request):
        raise AssertionError("unexpected unload request")


def _build_sagemaker_lora_test_client(
    monkeypatch, model_handler: _FakeLoRAModelHandler
) -> TestClient:
    from model_hosting_container_standards.common.handler import handler_registry
    from model_hosting_container_standards.sagemaker.sagemaker_loader import (
        SageMakerFunctionLoader,
    )

    from vllm.entrypoints.sagemaker.api_router import sagemaker_standards_bootstrap
    from vllm.entrypoints.serve.lora.api_router import attach_router

    handler_registry.clear()
    SageMakerFunctionLoader._default_function_loader = None
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True")

    app = FastAPI()
    app.state.openai_serving_models = model_handler
    attach_router(app)
    return TestClient(sagemaker_standards_bootstrap(app))


@pytest.mark.asyncio
async def test_sagemaker_load_adapter_happy_path(
    async_client: openai.AsyncOpenAI,
    basic_server_with_lora: RemoteOpenAIServer,
    smollm2_lora_files,
):
    # The SageMaker standards library creates a POST /adapters endpoint
    # that maps to the load_lora_adapter handler with request shape:
    # {"lora_name": "body.name", "lora_path": "body.src"}
    load_response = requests.post(
        basic_server_with_lora.url_for("adapters"),
        json={"name": "smollm2-lora-sagemaker", "src": smollm2_lora_files},
    )
    load_response.raise_for_status()

    models = await async_client.models.list()
    models = models.data
    dynamic_lora_model = models[-1]
    assert dynamic_lora_model.root == smollm2_lora_files
    assert dynamic_lora_model.parent == MODEL_NAME_SMOLLM
    assert dynamic_lora_model.id == "smollm2-lora-sagemaker"


@pytest.mark.asyncio
async def test_sagemaker_unload_adapter_happy_path(
    async_client: openai.AsyncOpenAI,
    basic_server_with_lora: RemoteOpenAIServer,
    smollm2_lora_files,
):
    # First, load an adapter
    adapter_name = "smollm2-lora-sagemaker-unload"
    load_response = requests.post(
        basic_server_with_lora.url_for("adapters"),
        json={"name": adapter_name, "src": smollm2_lora_files},
    )
    load_response.raise_for_status()

    # Verify it's in the models list
    models = await async_client.models.list()
    adapter_ids = [model.id for model in models.data]
    assert adapter_name in adapter_ids

    # Now unload it using DELETE /adapters/{adapter_name}
    # The SageMaker standards maps this to unload_lora_adapter with:
    # {"lora_name": "path_params.adapter_name"}
    unload_response = requests.delete(
        basic_server_with_lora.url_for("adapters", adapter_name),
    )
    unload_response.raise_for_status()

    # Verify it's no longer in the models list
    models = await async_client.models.list()
    adapter_ids = [model.id for model in models.data]
    assert adapter_name not in adapter_ids


@pytest.mark.skip_global_cleanup
def test_sagemaker_load_adapter_not_found(monkeypatch):
    model_handler = _FakeLoRAModelHandler()

    with _build_sagemaker_lora_test_client(monkeypatch, model_handler) as client:
        load_response = client.post(
            "/adapters",
            json={"name": "nonexistent-adapter", "src": "/path/does/not/exist"},
        )

    assert load_response.status_code == 404
    assert len(model_handler.load_requests) == 1
    load_request = model_handler.load_requests[0]
    assert load_request.lora_name == "nonexistent-adapter"
    assert load_request.lora_path == "/path/does/not/exist"
    assert load_request.load_inplace is False
    assert load_request.is_3d_lora_weight is False


@pytest.mark.asyncio
async def test_sagemaker_load_adapter_invalid_files(
    basic_server_with_lora: RemoteOpenAIServer,
    tmp_path,
):
    invalid_files = tmp_path / "invalid_adapter"
    invalid_files.mkdir()
    (invalid_files / "adapter_config.json").write_text("not valid json")

    load_response = requests.post(
        basic_server_with_lora.url_for("adapters"),
        json={"name": "invalid-adapter", "src": str(invalid_files)},
    )
    assert load_response.status_code == 500


@pytest.mark.asyncio
async def test_sagemaker_unload_nonexistent_adapter(
    basic_server_with_lora: RemoteOpenAIServer,
):
    # Attempt to unload an adapter that doesn't exist
    unload_response = requests.delete(
        basic_server_with_lora.url_for("adapters", "nonexistent-adapter-name"),
    )
    assert unload_response.status_code in (400, 404)


@pytest.mark.asyncio
async def test_sagemaker_invocations_with_adapter(
    basic_server_with_lora: RemoteOpenAIServer,
    smollm2_lora_files,
):
    # First, load an adapter via SageMaker endpoint
    adapter_name = "smollm2-lora-invoke-test"
    load_response = requests.post(
        basic_server_with_lora.url_for("adapters"),
        json={"name": adapter_name, "src": smollm2_lora_files},
    )
    load_response.raise_for_status()

    # Now test the /invocations endpoint with the adapter
    invocation_response = requests.post(
        basic_server_with_lora.url_for("invocations"),
        headers={
            "X-Amzn-SageMaker-Adapter-Identifier": adapter_name,
        },
        json={
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
        },
    )
    invocation_response.raise_for_status()
    invocation_output = invocation_response.json()

    # Verify we got a valid completion response
    assert "choices" in invocation_output
    assert len(invocation_output["choices"]) > 0
    assert "text" in invocation_output["choices"][0]


@pytest.mark.asyncio
async def test_sagemaker_multiple_adapters_load_unload(
    async_client: openai.AsyncOpenAI,
    basic_server_with_lora: RemoteOpenAIServer,
    smollm2_lora_files,
):
    adapter_names = [f"sagemaker-adapter-{i}" for i in range(5)]

    # Load all adapters
    for adapter_name in adapter_names:
        load_response = requests.post(
            basic_server_with_lora.url_for("adapters"),
            json={"name": adapter_name, "src": smollm2_lora_files},
        )
        load_response.raise_for_status()

    # Verify all are in the models list
    models = await async_client.models.list()
    adapter_ids = [model.id for model in models.data]
    for adapter_name in adapter_names:
        assert adapter_name in adapter_ids

    # Unload all adapters
    for adapter_name in adapter_names:
        unload_response = requests.delete(
            basic_server_with_lora.url_for("adapters", adapter_name),
        )
        unload_response.raise_for_status()

    # Verify all are removed from models list
    models = await async_client.models.list()
    adapter_ids = [model.id for model in models.data]
    for adapter_name in adapter_names:
        assert adapter_name not in adapter_ids
