import asyncio
import gc
import hashlib
import importlib
import inspect
import json
import multiprocessing
import multiprocessing.forkserver as forkserver
import os
import secrets
import signal
import socket
import tempfile
import uuid
from argparse import Namespace
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Annotated, Any, Literal

import prometheus_client
import pydantic
import regex as re
import uvloop
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, Headers, MutableHeaders, State
from starlette.routing import Mount
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import Device, EngineClient
from vllm.entrypoints.anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from vllm.entrypoints.anthropic.serving_messages import AnthropicServingMessages
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ClassificationRequest,
    ClassificationResponse,
    CompletionRequest,
    CompletionResponse,
    DetokenizeRequest,
    DetokenizeResponse,
    EmbeddingBytesResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorInfo,
    ErrorResponse,
    IOProcessorResponse,
    LoadLoRAAdapterRequest,
    PoolingBytesResponse,
    PoolingRequest,
    PoolingResponse,
    RerankRequest,
    RerankResponse,
    ResponsesRequest,
    ResponsesResponse,
    ScoreRequest,
    ScoreResponse,
    StreamingResponsesResponse,
    TokenizeRequest,
    TokenizeResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
    UnloadLoRAAdapterRequest,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_classification import ServingClassification
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.entrypoints.openai.serving_responses import OpenAIServingResponses
from vllm.entrypoints.openai.serving_score import ServingScores
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.serve.utils import validate_json_request
from vllm.entrypoints.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.utils import (
    cli_env_setup,
    load_aware_call,
    log_non_default_args,
    process_chat_template,
    process_lora_modules,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tasks import POOLING_TASKS
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import decorate_logs, set_ulimit
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.prometheus import get_prometheus_registry
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter(tags=["vLLM APIs"])

logger = init_logger("vllm.entrypoints.jinja.api_server")


if envs.VLLM_SERVER_DEV_MODE:
    logger.warning(
        "SECURITY WARNING: Development endpoints are enabled! "
        "This should NOT be used in production!"
    )

    PydanticVllmConfig = pydantic.TypeAdapter(VllmConfig)

    @router.get("/server_info")
    async def show_server_info(
        raw_request: Request,
        config_format: Annotated[Literal["text", "json"], Query()] = "text",
    ):
        vllm_config: VllmConfig = raw_request.app.state.vllm_config
        server_info = {
            "vllm_config": str(vllm_config)
            if config_format == "text"
            else PydanticVllmConfig.dump_python(vllm_config, mode="json", fallback=str)
            # fallback=str is needed to handle e.g. torch.dtype
        }
        return JSONResponse(content=server_info)

    @router.post("/reset_prefix_cache")
    async def reset_prefix_cache(raw_request: Request):
        """
        Reset the prefix cache. Note that we currently do not check if the
        prefix cache is successfully reset in the API server.
        """
        device = None
        device_str = raw_request.query_params.get("device")
        if device_str is not None:
            device = Device[device_str.upper()]
        logger.info("Resetting prefix cache with specific %s...", str(device))
        await engine_client(raw_request).reset_prefix_cache(device)
        return Response(status_code=200)

    @router.post("/reset_mm_cache")
    async def reset_mm_cache(raw_request: Request):
        """
        Reset the multi-modal cache. Note that we currently do not check if the
        multi-modal cache is successfully reset in the API server.
        """
        logger.info("Resetting multi-modal cache...")
        await engine_client(raw_request).reset_mm_cache()
        return Response(status_code=200)

    @router.post("/sleep")
    async def sleep(raw_request: Request):
        # get POST params
        level = raw_request.query_params.get("level", "1")
        await engine_client(raw_request).sleep(int(level))
        # FIXME: in v0 with frontend multiprocessing, the sleep command
        # is sent but does not finish yet when we return a response.
        return Response(status_code=200)

    @router.post("/wake_up")
    async def wake_up(raw_request: Request):
        tags = raw_request.query_params.getlist("tags")
        if tags == []:
            # set to None to wake up all tags if no tags are provided
            tags = None
        logger.info("wake up the engine with tags: %s", tags)
        await engine_client(raw_request).wake_up(tags)
        # FIXME: in v0 with frontend multiprocessing, the wake-up command
        # is sent but does not finish yet when we return a response.
        return Response(status_code=200)

    @router.get("/is_sleeping")
    async def is_sleeping(raw_request: Request):
        logger.info("check whether the engine is sleeping")
        is_sleeping = await engine_client(raw_request).is_sleeping()
        return JSONResponse(content={"is_sleeping": is_sleeping})

    @router.post("/collective_rpc")
    async def collective_rpc(raw_request: Request):
        try:
            body = await raw_request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"JSON decode error: {e}",
            ) from e
        method = body.get("method")
        if method is None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Missing 'method' in request body",
            )
        # For security reason, only serialized string args/kwargs are passed.
        # User-defined `method` is responsible for deserialization if needed.
        args: list[str] = body.get("args", [])
        kwargs: dict[str, str] = body.get("kwargs", {})
        timeout: float | None = body.get("timeout")
        results = await engine_client(raw_request).collective_rpc(
            method=method, timeout=timeout, args=tuple(args), kwargs=kwargs
        )
        if results is None:
            return Response(status_code=200)
        response: list[Any] = []
        for result in results:
            if result is None or isinstance(result, (dict, list)):
                response.append(result)
            else:
                response.append(str(result))
        return JSONResponse(content={"results": response})


@router.post(
    "/scale_elastic_ep",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def scale_elastic_ep(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e  # noqa: B904

    new_data_parallel_size = body.get("new_data_parallel_size")
    drain_timeout = body.get("drain_timeout", 120)  # Default 2 minutes

    if new_data_parallel_size is None:
        raise HTTPException(
            status_code=400, detail="new_data_parallel_size is required"
        )

    if not isinstance(new_data_parallel_size, int) or new_data_parallel_size <= 0:
        raise HTTPException(
            status_code=400, detail="new_data_parallel_size must be a positive integer"
        )

    if not isinstance(drain_timeout, int) or drain_timeout <= 0:
        raise HTTPException(
            status_code=400, detail="drain_timeout must be a positive integer"
        )

    # Set scaling flag to prevent new requests
    global _scaling_elastic_ep
    _scaling_elastic_ep = True
    client = engine_client(raw_request)
    try:
        await client.scale_elastic_ep(new_data_parallel_size, drain_timeout)
        return JSONResponse(
            {
                "message": f"Scaled to {new_data_parallel_size} data parallel engines",
            }
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail="Scale failed due to request drain timeout "
            f"after {drain_timeout} seconds",
        ) from e
    except Exception as e:
        logger.error("Scale failed: %s", e)
        raise HTTPException(status_code=500, detail="Scale failed") from e
    finally:
        _scaling_elastic_ep = False


@router.post("/is_scaling_elastic_ep")
async def is_scaling_elastic_ep(raw_request: Request):
    return JSONResponse({"is_scaling_elastic_ep": _scaling_elastic_ep})


# TODO: RequestType = TypeForm[BaseModel] when recognized by type checkers
# (requires typing_extensions >= 4.13)
RequestType = Any
GetHandlerFn = Callable[[Request], OpenAIServing | None]
EndpointFn = Callable[[RequestType, Request], Awaitable[Any]]

# NOTE: Items defined earlier take higher priority
INVOCATION_TYPES: list[tuple[RequestType, tuple[GetHandlerFn, EndpointFn]]] = [
    (ChatCompletionRequest, (chat, create_chat_completion)),
    (CompletionRequest, (completion, create_completion)),
    (EmbeddingRequest, (embedding, create_embedding)),
    (ClassificationRequest, (classify, create_classify)),
    (ScoreRequest, (score, create_score)),
    (RerankRequest, (rerank, do_rerank)),
    (PoolingRequest, (pooling, create_pooling)),
]

# NOTE: Construct the TypeAdapters only once
INVOCATION_VALIDATORS = [
    (pydantic.TypeAdapter(request_type), (get_handler, endpoint))
    for request_type, (get_handler, endpoint) in INVOCATION_TYPES
]


@router.post(
    "/invocations",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def invocations(raw_request: Request):
    """For SageMaker, routes requests based on the request type."""
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, detail=f"JSON decode error: {e}"
        ) from e

    valid_endpoints = [
        (validator, endpoint)
        for validator, (get_handler, endpoint) in INVOCATION_VALIDATORS
        if get_handler(raw_request) is not None
    ]

    for request_validator, endpoint in valid_endpoints:
        try:
            request = request_validator.validate_python(body)
        except pydantic.ValidationError:
            continue

        return await endpoint(request, raw_request)

    type_names = [
        t.__name__ if isinstance(t := validator._type, type) else str(t)
        for validator, _ in valid_endpoints
    ]
    msg = f"Cannot find suitable handler for request. Expected one of: {type_names}"
    res = base(raw_request).create_error_response(message=msg)
    return JSONResponse(content=res.model_dump(), status_code=res.error.code)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!"
    )

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
    logger.warning(
        "LoRA dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!"
    )

    @router.post("/v1/load_lora_adapter", dependencies=[Depends(validate_json_request)])
    async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
        handler = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

        return Response(status_code=200, content=response)

    @router.post(
        "/v1/unload_lora_adapter", dependencies=[Depends(validate_json_request)]
    )
    async def unload_lora_adapter(
        request: UnloadLoRAAdapterRequest, raw_request: Request
    ):
        handler = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

        return Response(status_code=200, content=response)
