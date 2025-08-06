# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://github.com/vllm/vllm/entrypoints/openai/api_server.py

import asyncio
import signal
import tempfile
from argparse import Namespace
from http import HTTPStatus
from typing import Optional

import uvloop
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import State

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.protocol import AnthropicErrorResponse, AnthropicMessagesRequest, \
    AnthropicMessagesResponse
from vllm.entrypoints.anthropic.serving_messages import AnthropicServingMessages
from vllm.entrypoints.chat_utils import (load_chat_template,
                                         resolve_hf_chat_template,
                                         resolve_mistral_chat_template)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import validate_api_server_args, create_server_socket, load_log_config, \
    lifespan, build_async_engine_client, validate_json_request
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath, LoRAModulePath
#
# yapf: enable
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.utils import (cli_env_setup, load_aware_call,
                                    with_cancellation)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import (FlexibleArgumentParser,
                        is_valid_ipv6_address,
                        set_ulimit)
from vllm.version import __version__ as VLLM_VERSION

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.anthropic.api_server')

_running_tasks: set[asyncio.Task] = set()

router = APIRouter()


def messages(request: Request) -> Optional[AnthropicServingMessages]:
    return request.app.state.anthropic_serving_messages


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.get("/ping", response_class=Response)
@router.post("/ping", response_class=Response)
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return await health(raw_request)


@router.post("/v1/messages",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.OK.value: {
                     "content": {
                         "text/event-stream": {}
                     }
                 },
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": AnthropicErrorResponse
                 },
                 HTTPStatus.NOT_FOUND.value: {
                     "model": AnthropicErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": AnthropicErrorResponse
                 }
             })
@with_cancellation
@load_aware_call
async def create_messages(request: AnthropicMessagesRequest,
                          raw_request: Request):
    handler = messages(raw_request)
    if handler is None:
        return messages(raw_request).create_error_response(
            message="The model does not support Chat Completions API")

    generator = await handler.create_messages(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, AnthropicMessagesResponse):
        return JSONResponse(content=generator.model_dump(exclude_none=True, exclude_unset=True))

    return StreamingResponse(content=generator, media_type="text/event-stream")


async def init_app_state(
        engine_client: EngineClient,
        vllm_config: VllmConfig,
        state: State,
        args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    model_config = vllm_config.model_config

    default_mm_loras = (vllm_config.lora_config.default_mm_loras
                        if vllm_config.lora_config is not None else {})
    lora_modules = args.lora_modules
    if default_mm_loras:
        default_mm_lora_paths = [
            LoRAModulePath(
                name=modality,
                path=lora_path,
            ) for modality, lora_path in default_mm_loras.items()
        ]
        if args.lora_modules is None:
            lora_modules = default_mm_lora_paths
        else:
            lora_modules += default_mm_lora_paths

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = await engine_client.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(
                chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer=tokenizer,
                chat_template=None,
                tools=None,
                model_config=vllm_config.model_config,
            )

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: %s\n"
                    "It is different from official chat template '%s'. "
                    "This discrepancy may lead to performance degradation.",
                    resolved_chat_template, args.model)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()
    state.anthropic_serving_messages = AnthropicServingMessages(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
    )


def setup_server(args):
    """Validate API server args, set up signal handler, create socket
    ready to serve."""

    logger.info("vLLM API server version %s", VLLM_VERSION)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    addr, port = sock_addr
    is_ssl = args.ssl_keyfile and args.ssl_certfile
    host_part = f"[{addr}]" if is_valid_ipv6_address(
        addr) else addr or "0.0.0.0"
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"

    return listen_address, sock


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    return app


async def run_server_worker(listen_address,
                            sock,
                            args,
                            client_config=None,
                            **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs['log_config'] = log_config

    async with build_async_engine_client(
            args,
            client_config=client_config,
    ) as engine_client:
        app = build_app(args)

        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index,
                    listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM Anthropic-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
