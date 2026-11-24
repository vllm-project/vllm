# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from vllm import PoolingParams, envs
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.base.serving import PoolingBaseServing
from vllm.entrypoints.pooling.types import PoolingServeContext
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.lora import api_router as lora_api_router
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    PinLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
    UnpinLoRAAdapterRequest,
)
from vllm.exceptions import VLLMNotFoundError
from vllm.lora.request import LoRARequest

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]
LORA_LOADING_SUCCESS_MESSAGE = "Success: LoRA adapter '{lora_name}' added successfully."
LORA_UNLOADING_SUCCESS_MESSAGE = (
    "Success: LoRA adapter '{lora_name}' removed successfully."
)
LORA_PINNING_SUCCESS_MESSAGE = (
    "Success: LoRA adapter '{lora_name}' pinned successfully."
)
LORA_UNPINNING_SUCCESS_MESSAGE = (
    "Success: LoRA adapter '{lora_name}' unpinned successfully."
)


async def _async_serving_models_init_with_mock() -> tuple[
    OpenAIServingModels, MagicMock
]:
    mock_engine_client = MagicMock(spec=EngineClient)
    # Set the max_model_len attribute to avoid missing attribute
    mock_model_config = MagicMock(spec=ModelConfig)
    mock_model_config.max_model_len = 2048
    mock_engine_client.model_config = mock_model_config
    mock_engine_client.input_processor = MagicMock()
    mock_engine_client.renderer = MagicMock()

    serving_models = OpenAIServingModels(
        engine_client=mock_engine_client,
        base_model_paths=BASE_MODEL_PATHS,
        lora_modules=None,
    )
    await serving_models.init_static_loras()

    # The mock is returned separately: `engine_client` is typed as the protocol,
    # so assertion helpers are not visible through it.
    return serving_models, mock_engine_client


async def _async_serving_models_init() -> OpenAIServingModels:
    serving_models, _ = await _async_serving_models_init_with_mock()
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
@pytest.mark.parametrize("load_inplace", [False, True])
async def test_load_lora_adapter_base_model_name(load_inplace: bool):
    """A LoRA named after a served model would shadow it for all requests."""
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    request = LoadLoRAAdapterRequest(
        lora_name=MODEL_NAME, lora_path="/path/to/adapter", load_inplace=load_inplace
    )
    response = await serving_models.load_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST
    assert len(serving_models.lora_requests) == 0
    mock_engine_client.add_lora.assert_not_called()


@pytest.mark.asyncio
async def test_unload_lora_adapter_success():
    serving_models = await _async_serving_models_init()
    request = LoadLoRAAdapterRequest(
        lora_name="adapter1", lora_path="/path/to/adapter1"
    )
    response = await serving_models.load_lora_adapter(request)
    assert len(serving_models.lora_requests) == 1

    unload_request = UnloadLoRAAdapterRequest(lora_name="adapter1")
    response = await serving_models.unload_lora_adapter(unload_request)
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


@pytest.mark.asyncio
async def test_load_lora_adapter_pinned_success():
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    request = LoadLoRAAdapterRequest(
        lora_name="adapter_pinned", lora_path="/path/to/adapter", pinned=True
    )
    response = await serving_models.load_lora_adapter(request)
    assert response == LORA_LOADING_SUCCESS_MESSAGE.format(lora_name="adapter_pinned")
    assert "adapter_pinned" in serving_models.lora_requests
    lora_int_id = serving_models.lora_requests["adapter_pinned"].lora_int_id
    mock_engine_client.add_lora.assert_awaited_once()
    mock_engine_client.pin_lora.assert_awaited_once_with(lora_int_id)


@pytest.mark.asyncio
async def test_load_lora_adapter_pinned_failure_rolls_back():
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    mock_engine_client.pin_lora.side_effect = Exception(
        "Call to pin_lora method failed: "
        "All items are pinned, cannot remove oldest from the cache."
    )
    request = LoadLoRAAdapterRequest(
        lora_name="adapter_fail_pin", lora_path="/path/to/adapter", pinned=True
    )
    response = await serving_models.load_lora_adapter(request)
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST
    assert "adapter_fail_pin" not in serving_models.lora_requests
    mock_engine_client.remove_lora.assert_awaited_once()


def _attach_fake_worker_lora_cache(
    mock_engine_client: MagicMock,
    *,
    failing_paths: set[str] | None = None,
) -> tuple[dict[int, str], set[int]]:
    """Simulate LRUCacheWorkerLoRAManager cache semantics on the mock client."""
    worker_cache: dict[int, str] = {}
    worker_pinned: set[int] = set()
    bad_paths = failing_paths if failing_paths is not None else set()

    async def _add_lora(lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in worker_cache or lora_request.load_inplace:
            if lora_request.lora_path in bad_paths:
                raise RuntimeError(f"Failed to load {lora_request.lora_path}")
            # Mirrors LRUCacheWorkerLoRAManager.add_adapter: remove_adapter
            # unpins the old entry before inserting the replacement.
            worker_pinned.discard(lora_request.lora_int_id)
            worker_cache[lora_request.lora_int_id] = lora_request.lora_path
            return True
        return False

    async def _remove_lora(lora_id: int) -> bool:
        worker_pinned.discard(lora_id)
        return worker_cache.pop(lora_id, None) is not None

    async def _pin_lora(lora_id: int) -> bool:
        if lora_id not in worker_cache:
            raise ValueError(f"Pinning failed. LoRA {lora_id} is not registered.")
        worker_pinned.add(lora_id)
        return True

    async def _unpin_lora(lora_id: int) -> bool:
        if lora_id not in worker_cache:
            raise ValueError(f"Unpinning failed. LoRA {lora_id} is not registered.")
        worker_pinned.discard(lora_id)
        return True

    mock_engine_client.add_lora.side_effect = _add_lora
    mock_engine_client.remove_lora.side_effect = _remove_lora
    mock_engine_client.pin_lora.side_effect = _pin_lora
    mock_engine_client.unpin_lora.side_effect = _unpin_lora
    return worker_cache, worker_pinned


async def _next_request_served_version(
    serving_models: OpenAIServingModels,
    worker_cache: dict[int, str],
    lora_name: str,
) -> tuple[str, str]:
    """Return (frontend_path, worker_path) seen by the next inference request."""
    resolved = await serving_models.resolve_lora(lora_name)
    assert isinstance(resolved, LoRARequest)
    assert resolved.load_inplace is False
    # On inference, LRUCacheWorkerLoRAManager reuses the cached adapter for
    # lora_int_id when load_inplace is False.
    worker_path = worker_cache.get(resolved.lora_int_id, resolved.lora_path)
    return resolved.lora_path, worker_path


@pytest.mark.asyncio
async def test_load_lora_adapter_inplace_pinned_failure_rolls_back_to_previous():
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    worker_cache, worker_pinned = _attach_fake_worker_lora_cache(mock_engine_client)

    await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(lora_name="adapter1", lora_path="/path/to/v1")
    )
    lora_int_id = serving_models.lora_requests["adapter1"].lora_int_id
    assert worker_cache[lora_int_id] == "/path/to/v1"

    mock_engine_client.pin_lora.side_effect = NotImplementedError(
        "Pinning is not supported in LoRAModelManager."
    )
    response = await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name="adapter1",
            lora_path="/path/to/v2",
            load_inplace=True,
            pinned=True,
        )
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST
    mock_engine_client.remove_lora.assert_not_called()

    frontend_path, worker_path = await _next_request_served_version(
        serving_models, worker_cache, "adapter1"
    )
    assert frontend_path == "/path/to/v1"
    assert worker_path == "/path/to/v1"
    assert lora_int_id not in worker_pinned


@pytest.mark.asyncio
async def test_load_lora_adapter_inplace_pinned_failure_restores_previous_pin():
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    worker_cache, worker_pinned = _attach_fake_worker_lora_cache(mock_engine_client)

    await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name="adapter1", lora_path="/path/to/v1", pinned=True
        )
    )
    lora_int_id = serving_models.lora_requests["adapter1"].lora_int_id
    assert lora_int_id in worker_pinned
    assert lora_int_id in serving_models._pinned_lora_ids

    # Simulate a transient pin failure on v2, followed by a successful re-pin
    # when rolling back to v1.
    real_pin = mock_engine_client.pin_lora.side_effect
    attempts = 0

    async def _fail_once_then_pin(lora_id: int) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise Exception(
                "Call to pin_lora method failed: "
                "All items are pinned, cannot remove oldest from the cache."
            )
        return await real_pin(lora_id)

    mock_engine_client.pin_lora.side_effect = _fail_once_then_pin
    response = await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name="adapter1",
            lora_path="/path/to/v2",
            load_inplace=True,
            pinned=True,
        )
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST

    frontend_path, worker_path = await _next_request_served_version(
        serving_models, worker_cache, "adapter1"
    )
    assert frontend_path == "/path/to/v1"
    assert worker_path == "/path/to/v1"
    assert lora_int_id in worker_pinned
    assert lora_int_id in serving_models._pinned_lora_ids


@pytest.mark.asyncio
async def test_load_lora_adapter_inplace_pinned_failure_rollback_error_stays_in_sync():
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    failing_paths: set[str] = set()
    worker_cache, worker_pinned = _attach_fake_worker_lora_cache(
        mock_engine_client, failing_paths=failing_paths
    )

    await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name="adapter1", lora_path="/path/to/v1", pinned=True
        )
    )
    lora_int_id = serving_models.lora_requests["adapter1"].lora_int_id

    # v1 can no longer be reloaded during rollback, while pinning v2 fails.
    failing_paths.add("/path/to/v1")
    mock_engine_client.pin_lora.side_effect = Exception(
        "Call to pin_lora method failed: "
        "All items are pinned, cannot remove oldest from the cache."
    )
    response = await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name="adapter1",
            lora_path="/path/to/v2",
            load_inplace=True,
            pinned=True,
        )
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InvalidUserInput"
    assert response.error.code == HTTPStatus.BAD_REQUEST

    # Since worker failed to reload v1 and kept v2, frontend must also point
    # to v2 (unpinned) so frontend and worker do not diverge.
    frontend_path, worker_path = await _next_request_served_version(
        serving_models, worker_cache, "adapter1"
    )
    assert frontend_path == "/path/to/v2"
    assert worker_path == "/path/to/v2"
    assert lora_int_id not in worker_pinned
    assert lora_int_id not in serving_models._pinned_lora_ids


@pytest.mark.asyncio
async def test_pin_and_unpin_lora_adapter_success():
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(lora_name="adapter1", lora_path="/path/to/adapter1")
    )
    lora_int_id = serving_models.lora_requests["adapter1"].lora_int_id

    pin_resp = await serving_models.pin_lora_adapter(
        PinLoRAAdapterRequest(lora_name="adapter1")
    )
    assert pin_resp == LORA_PINNING_SUCCESS_MESSAGE.format(lora_name="adapter1")
    mock_engine_client.pin_lora.assert_awaited_once_with(lora_int_id)

    unpin_resp = await serving_models.unpin_lora_adapter(
        UnpinLoRAAdapterRequest(lora_name="adapter1")
    )
    assert unpin_resp == LORA_UNPINNING_SUCCESS_MESSAGE.format(lora_name="adapter1")
    mock_engine_client.unpin_lora.assert_awaited_once_with(lora_int_id)


@pytest.mark.asyncio
async def test_unload_pinned_lora_adapter_unpins_first():
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name="pinned_adapter", lora_path="/path/to/pinned", pinned=True
        )
    )
    lora_int_id = serving_models.lora_requests["pinned_adapter"].lora_int_id

    response = await serving_models.unload_lora_adapter(
        UnloadLoRAAdapterRequest(lora_name="pinned_adapter")
    )
    assert response == LORA_UNLOADING_SUCCESS_MESSAGE.format(lora_name="pinned_adapter")
    mock_engine_client.unpin_lora.assert_awaited_once_with(lora_int_id)
    assert "pinned_adapter" not in serving_models.lora_requests

    # Unloading an unpinned adapter should not issue an extra unpin_lora RPC.
    mock_engine_client.unpin_lora.reset_mock()
    await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(lora_name="normal_adapter", lora_path="/path/to/normal")
    )
    await serving_models.unload_lora_adapter(
        UnloadLoRAAdapterRequest(lora_name="normal_adapter")
    )
    mock_engine_client.unpin_lora.assert_not_called()


@pytest.mark.asyncio
async def test_pin_and_unpin_lora_adapter_validation_errors():
    serving_models = await _async_serving_models_init()

    for call in (
        serving_models.pin_lora_adapter(PinLoRAAdapterRequest(lora_name="")),
        serving_models.unpin_lora_adapter(UnpinLoRAAdapterRequest(lora_name="")),
    ):
        resp = await call
        assert isinstance(resp, ErrorResponse)
        assert resp.error.type == "InvalidUserInput"
        assert resp.error.code == HTTPStatus.BAD_REQUEST

    for call in (
        serving_models.pin_lora_adapter(PinLoRAAdapterRequest(lora_name="missing")),
        serving_models.unpin_lora_adapter(UnpinLoRAAdapterRequest(lora_name="missing")),
    ):
        resp = await call
        assert isinstance(resp, ErrorResponse)
        assert resp.error.type == "NotFoundError"
        assert resp.error.code == HTTPStatus.NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_type"),
    [
        (
            NotImplementedError("Pinning is not supported in LoRAModelManager."),
            HTTPStatus.BAD_REQUEST,
            "InvalidUserInput",
        ),
        (
            Exception(
                "Call to pin_lora method failed: "
                "Pinning is not supported in LoRAModelManager."
            ),
            HTTPStatus.BAD_REQUEST,
            "InvalidUserInput",
        ),
        (
            Exception(
                "Call to pin_lora method failed: "
                "All items are pinned, cannot remove oldest from the cache."
            ),
            HTTPStatus.BAD_REQUEST,
            "InvalidUserInput",
        ),
        (
            Exception(
                "Call to unpin_lora method failed: "
                "Unpinning failed. LoRA 1 is not registered."
            ),
            HTTPStatus.BAD_REQUEST,
            "InvalidUserInput",
        ),
        (
            RuntimeError("unexpected engine failure"),
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "InternalServerError",
        ),
    ],
)
async def test_pin_and_unpin_lora_adapter_engine_errors(
    exc: Exception, expected_status: HTTPStatus, expected_type: str
):
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()
    await serving_models.load_lora_adapter(
        LoadLoRAAdapterRequest(lora_name="adapter1", lora_path="/path/to/adapter1")
    )

    mock_engine_client.pin_lora.side_effect = exc
    pin_resp = await serving_models.pin_lora_adapter(
        PinLoRAAdapterRequest(lora_name="adapter1")
    )
    assert isinstance(pin_resp, ErrorResponse)
    assert pin_resp.error.code == expected_status
    assert pin_resp.error.type == expected_type

    mock_engine_client.unpin_lora.side_effect = exc
    unpin_resp = await serving_models.unpin_lora_adapter(
        UnpinLoRAAdapterRequest(lora_name="adapter1")
    )
    assert isinstance(unpin_resp, ErrorResponse)
    assert unpin_resp.error.code == expected_status
    assert unpin_resp.error.type == expected_type


@pytest.mark.asyncio
async def test_lora_router_pin_and_unpin_endpoints(monkeypatch: pytest.MonkeyPatch):
    serving_models, mock_engine_client = await _async_serving_models_init_with_mock()

    # Disabled by default when VLLM_ALLOW_RUNTIME_LORA_UPDATING is False.
    monkeypatch.setattr(envs, "VLLM_ALLOW_RUNTIME_LORA_UPDATING", False)
    monkeypatch.setattr(lora_api_router, "router", APIRouter())
    disabled_app = FastAPI()
    disabled_app.state.openai_serving_models = serving_models
    lora_api_router.attach_router(disabled_app)
    with TestClient(disabled_app) as client:
        assert (
            client.post("/v1/pin_lora_adapter", json={"lora_name": "a"}).status_code
            == HTTPStatus.NOT_FOUND
        )
        assert (
            client.post("/v1/unpin_lora_adapter", json={"lora_name": "a"}).status_code
            == HTTPStatus.NOT_FOUND
        )

    # Enabled when VLLM_ALLOW_RUNTIME_LORA_UPDATING is True.
    monkeypatch.setattr(envs, "VLLM_ALLOW_RUNTIME_LORA_UPDATING", True)
    monkeypatch.setattr(lora_api_router, "router", APIRouter())
    app = FastAPI()
    app.state.openai_serving_models = serving_models
    lora_api_router.attach_router(app)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/load_lora_adapter",
            json={
                "lora_name": "adapter1",
                "lora_path": "/path/to/adapter1",
                "pinned": True,
            },
        )
        assert resp.status_code == HTTPStatus.OK
        assert resp.text == LORA_LOADING_SUCCESS_MESSAGE.format(lora_name="adapter1")
        lora_int_id = serving_models.lora_requests["adapter1"].lora_int_id
        mock_engine_client.pin_lora.assert_awaited_once_with(lora_int_id)

        resp = client.post("/v1/pin_lora_adapter", json={"lora_name": "adapter1"})
        assert resp.status_code == HTTPStatus.OK
        assert resp.text == LORA_PINNING_SUCCESS_MESSAGE.format(lora_name="adapter1")

        resp = client.post("/v1/unpin_lora_adapter", json={"lora_name": "adapter1"})
        assert resp.status_code == HTTPStatus.OK
        assert resp.text == LORA_UNPINNING_SUCCESS_MESSAGE.format(lora_name="adapter1")
        mock_engine_client.unpin_lora.assert_awaited_once_with(lora_int_id)

        resp = client.post("/v1/pin_lora_adapter", json={"lora_name": "missing"})
        assert resp.status_code == HTTPStatus.NOT_FOUND
        assert resp.json()["error"]["type"] == "NotFoundError"

        mock_engine_client.pin_lora.side_effect = Exception(
            "Call to pin_lora method failed: "
            "All items are pinned, cannot remove oldest from the cache."
        )
        resp = client.post("/v1/pin_lora_adapter", json={"lora_name": "adapter1"})
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert resp.json()["error"]["type"] == "InvalidUserInput"


class _ConcretePoolingServing(PoolingBaseServing):
    """Minimal concrete subclass used only in these unit tests."""

    request_id_prefix = "test"

    def get_io_processor(self, request):
        raise NotImplementedError

    def _build_response(self, ctx):
        raise NotImplementedError


def _make_pooling_serving(lora_name: str) -> _ConcretePoolingServing:
    lora_request = LoRARequest(
        lora_name=lora_name, lora_int_id=1, lora_path="/path/to/lora"
    )
    mock_models = MagicMock()
    mock_models.lora_requests = {lora_name: lora_request}
    mock_models.is_base_model.side_effect = lambda name: name == MODEL_NAME

    serving = object.__new__(_ConcretePoolingServing)
    serving.models = mock_models
    return serving


def _make_pooling_ctx(
    model_name: str, serving: PoolingBaseServing
) -> PoolingServeContext:
    mock_request = MagicMock()
    mock_request.model = model_name
    return PoolingServeContext(
        request=mock_request,
        model_name=MODEL_NAME,
        request_id="test-id",
        pooling_params=PoolingParams(),
        lora_request=serving._maybe_get_adapters(mock_request),
        priorities=0,
        prompt_extras=None,
    )


def test_pooling_maybe_get_adapters_lora_name_sets_lora_request():
    """LoRA adapter name must populate ctx.lora_request without raising."""
    lora_name = "bot-embed-lora"
    serving = _make_pooling_serving(lora_name)
    ctx = _make_pooling_ctx(lora_name, serving)

    assert ctx.lora_request is not None
    assert ctx.lora_request.lora_name == lora_name


def test_pooling_maybe_get_adapters_unknown_model_raises():
    """An unrecognised model name must still raise VLLMNotFoundError."""
    serving = _make_pooling_serving("some-lora")

    with pytest.raises(VLLMNotFoundError):
        _make_pooling_ctx("unknown-model", serving)


@pytest.mark.parametrize(
    ("content_type", "part"),
    [
        ("audio_url", {"audio_url": {"url": "http://example.com/a.wav"}}),
        ("input_audio", {"input_audio": {"data": "abc", "format": "wav"}}),
    ],
)
def test_get_message_types_maps_audio_parts_to_audio(content_type, part):
    """Both "audio_url" and OpenAI-style "input_audio" parts must resolve to the
    "audio" modality so a default multimodal LoRA keyed on "audio" can match."""
    serving = object.__new__(BaseServing)
    request = SimpleNamespace(
        messages=[{"role": "user", "content": [{"type": content_type, **part}]}]
    )

    assert serving._get_message_types(request) == {"audio"}
