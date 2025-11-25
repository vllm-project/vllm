# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from unittest.mock import MagicMock

import pytest

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.lora.request import LoRARequest

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]
LORA_LOADING_SUCCESS_MESSAGE = "Success: LoRA adapter '{lora_name}' added successfully."
LORA_UNLOADING_SUCCESS_MESSAGE = (
    "Success: LoRA adapter '{lora_name}' removed successfully."
)


async def _async_serving_models_init() -> OpenAIServingModels:
    mock_engine_client = MagicMock(spec=EngineClient)
    # Set the max_model_len attribute to avoid missing attribute
    mock_model_config = MagicMock(spec=ModelConfig)
    mock_model_config.max_model_len = 2048
    mock_engine_client.model_config = mock_model_config
    mock_engine_client.processor = MagicMock()
    mock_engine_client.io_processor = MagicMock()

    serving_models = OpenAIServingModels(
        engine_client=mock_engine_client,
        base_model_paths=BASE_MODEL_PATHS,
        lora_modules=None,
    )
    await serving_models.init_static_loras()

    return serving_models


@pytest.mark.asyncio
async def test_serving_model_name():
    serving_models = await _async_serving_models_init()
    assert serving_models.model_name(None) == MODEL_NAME
    request = LoRARequest(
        lora_name="adapter", lora_path="/path/to/adapter2", lora_int_id=1
    )
    assert serving_models.model_name(request) == request.lora_name


@pytest.mark.asyncio
async def test_load_lora_adapter_success():
    serving_models = await _async_serving_models_init()
    request = LoadLoRAAdapterRequest(lora_name="adapter", lora_path="/path/to/adapter2")
    response = await serving_models.load_lora_adapter(request)
    assert response == LORA_LOADING_SUCCESS_MESSAGE.format(lora_name="adapter")
    assert len(serving_models.lora_requests) == 1
    assert "adapter" in serving_models.lora_requests
    assert serving_models.lora_requests["adapter"].lora_name == "adapter"


@pytest.mark.asyncio
async def test_load_lora_adapter_missing_fields():
    serving_models = await _async_serving_models_init()
    request = LoadLoRAAdapterRequest(lora_name="", lora_path="")
    response = await serving_models.load_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_load_lora_adapter_duplicate():
    serving_models = await _async_serving_models_init()
    request = LoadLoRAAdapterRequest(
        lora_name="adapter1", lora_path="/path/to/adapter1"
    )
    response = await serving_models.load_lora_adapter(request)
    assert response == LORA_LOADING_SUCCESS_MESSAGE.format(lora_name="adapter1")
    assert len(serving_models.lora_requests) == 1

    request = LoadLoRAAdapterRequest(
        lora_name="adapter1", lora_path="/path/to/adapter1"
    )
    response = await serving_models.load_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST
    assert len(serving_models.lora_requests) == 1


@pytest.mark.asyncio
async def test_unload_lora_adapter_success():
    serving_models = await _async_serving_models_init()
    request = LoadLoRAAdapterRequest(
        lora_name="adapter1", lora_path="/path/to/adapter1"
    )
    response = await serving_models.load_lora_adapter(request)
    assert len(serving_models.lora_requests) == 1

    request = UnloadLoRAAdapterRequest(lora_name="adapter1")
    response = await serving_models.unload_lora_adapter(request)
    assert response == LORA_UNLOADING_SUCCESS_MESSAGE.format(lora_name="adapter1")
    assert len(serving_models.lora_requests) == 0


@pytest.mark.asyncio
async def test_unload_lora_adapter_missing_fields():
    serving_models = await _async_serving_models_init()
    request = UnloadLoRAAdapterRequest(lora_name="", lora_int_id=None)
    response = await serving_models.unload_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_unload_lora_adapter_not_found():
    serving_models = await _async_serving_models_init()
    request = UnloadLoRAAdapterRequest(lora_name="nonexistent_adapter")
    response = await serving_models.unload_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "NotFoundError"
    assert response.error.code == HTTPStatus.NOT_FOUND
