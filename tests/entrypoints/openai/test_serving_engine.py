import asyncio
from http import HTTPStatus
from unittest.mock import MagicMock

import pytest

from vllm.config import ModelConfig
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              UnloadLoraAdapterRequest)
from vllm.entrypoints.openai.serving_engine import OpenAIServing

MODEL_NAME = "meta-llama/Llama-2-7b"


async def _async_serving_engine_init():
    mock_engine_client = MagicMock(spec=AsyncEngineClient)
    mock_model_config = MagicMock(spec=ModelConfig)

    serving_engine = OpenAIServing(mock_engine_client,
                                   mock_model_config,
                                   served_model_names=[MODEL_NAME])
    return serving_engine


@pytest.mark.asyncio
async def test_load_lora_adapter_success():
    serving_engine = asyncio.run(_async_serving_engine_init())
    request = LoadLoraAdapterRequest(lora_name="adapter",
                                     lora_path="/path/to/adapter2")
    response = await serving_engine.load_lora_adapter(request)
    assert response.status_code == HTTPStatus.OK
    assert len(serving_engine.lora_requests) == 1
    assert serving_engine.lora_requests[0].lora_name == "adapter"


@pytest.mark.asyncio
async def test_load_lora_adapter_missing_fields():
    serving_engine = asyncio.run(_async_serving_engine_init())
    request = LoadLoraAdapterRequest(lora_name="", lora_path="")
    response = await serving_engine.load_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.err_type == "InvalidUserInput"
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert len(serving_engine.lora_requests) == 1


@pytest.mark.asyncio
async def test_load_lora_adapter_duplicate():
    serving_engine = asyncio.run(_async_serving_engine_init())
    request = LoadLoraAdapterRequest(lora_name="adapter1",
                                     lora_path="/path/to/adapter1")
    response = await serving_engine.load_lora_adapter(request)
    assert response.status_code == HTTPStatus.OK
    assert len(serving_engine.lora_requests) == 1

    request = LoadLoraAdapterRequest(lora_name="adapter1",
                                     lora_path="/path/to/adapter1")
    response = await serving_engine.load_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.err_type == "InvalidUserInput"
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert len(serving_engine.lora_requests) == 1


@pytest.mark.asyncio
async def test_unload_lora_adapter_success():
    serving_engine = asyncio.run(_async_serving_engine_init())
    request = LoadLoraAdapterRequest(lora_name="adapter1",
                                     lora_path="/path/to/adapter1")
    response = await serving_engine.load_lora_adapter(request)
    assert response.status_code == HTTPStatus.OK
    assert len(serving_engine.lora_requests) == 1

    request = UnloadLoraAdapterRequest(lora_name="adapter1")
    response = await serving_engine.unload_lora_adapter(request)
    assert response == "Success: LoRA adapter 'adapter1' removed successfully."
    assert len(serving_engine.lora_requests) == 0


@pytest.mark.asyncio
async def test_unload_lora_adapter_missing_fields(serving_engine):
    request = UnloadLoraAdapterRequest(lora_name="", lora_int_id=None)
    response = await serving_engine.unload_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.err_type == "InvalidUserInput"
    assert response.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_unload_lora_adapter_not_found(serving_engine):
    request = UnloadLoraAdapterRequest(lora_name="nonexistent_adapter")
    response = await serving_engine.unload_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.err_type == "InvalidUserInput"
    assert response.status_code == HTTPStatus.BAD_REQUEST
