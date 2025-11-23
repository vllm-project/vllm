# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

logger = init_logger("vllm.entrypoints.serve")
_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        gc.collect()
        gc.freeze()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


@asynccontextmanager
async def build_async_engine_client(
    args: Namespace,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)
    if client_config:
        engine_args._api_process_count = client_config.get("client_count", 1)
        engine_args._api_process_rank = client_config.get("client_index", 0)

    if disable_frontend_multiprocessing is None:
        disable_frontend_multiprocessing = bool(args.disable_frontend_multiprocessing)

    async with build_async_engine_client_from_engine_args(
        engine_args,
        usage_context=usage_context,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
        client_config=client_config,
    ) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool = False,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig (determines if we can use V1).
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    # V1 AsyncLLM.
    assert envs.VLLM_USE_V1

    if disable_frontend_multiprocessing:
        logger.warning(
            "V1 is enabled, but got --disable-frontend-multiprocessing. "
            "To disable frontend multiprocessing, set VLLM_USE_V1=0."
        )

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None

    # Don't mutate the input client_config
    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown()


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=["Unsupported Media Type: Only 'application/json' is allowed"]
        )


