import asyncio
import atexit
import importlib
import inspect
import multiprocessing
import os
import re
import signal
import socket
import tempfile
import uuid
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from typing import AsyncIterator, Optional, Set, Tuple

import uvloop
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import State
from starlette.routing import Mount
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              ScoreRequest, ScoreResponse,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              UnloadLoraAdapterRequest)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import BaseModelPath, OpenAIServing
from vllm.entrypoints.openai.serving_score import OpenAIServingScores
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, get_open_zmq_ipc_path,
                        is_valid_ipv6_address)
from vllm.version import __version__ as VLLM_VERSION

if envs.VLLM_USE_V1:
    from vllm.v1.engine.async_llm import AsyncLLMEngine  # type: ignore
else:
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore

TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None
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
        args: Namespace) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)

    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Fall back
    # TODO: fill out feature matrix.
    if (MQLLMEngineClient.is_unsupported_config(engine_args)
            or envs.VLLM_USE_V1 or disable_frontend_multiprocessing):
        engine_config = engine_args.create_engine_config(
            UsageContext.OPENAI_API_SERVER)
        uses_ray = getattr(AsyncLLMEngine._get_executor_cls(engine_config),
                           "uses_ray", False)

        build_engine = partial(AsyncLLMEngine.from_engine_args,
                               engine_args=engine_args,
                               engine_config=engine_config,
                               usage_context=UsageContext.OPENAI_API_SERVER)
        if uses_ray:
            # Must run in main thread with ray for its signal handlers to work
            engine_client = build_engine()
        else:
            engine_client = await asyncio.get_running_loop().run_in_executor(
                None, build_engine)

        yield engine_client
        if hasattr(engine_client, "shutdown"):
            engine_client.shutdown()
        return

    # Otherwise, use the multiprocessing AsyncLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ[
                "PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup.")

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.debug("Multiprocessing frontend to use %s for IPC Path.",
                     ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value('b', True, lock=False)
        engine_process = context.Process(target=run_mp_engine,
                                         args=(engine_args,
                                               UsageContext.OPENAI_API_SERVER,
                                               ipc_path, engine_alive))
        engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID %d", engine_pid)

        def _cleanup_ipc_path():
            socket_path = ipc_path.replace("ipc://", "")
            if os.path.exists(socket_path):
                os.remove(socket_path)

        # Ensure we clean up the local IPC socket file on exit.
        atexit.register(_cleanup_ipc_path)

        # Build RPCClient, which conforms to EngineClient Protocol.
        engine_config = engine_args.create_engine_config()
        build_client = partial(MQLLMEngineClient, ipc_path, engine_config,
                               engine_pid)
        mq_engine_client = await asyncio.get_running_loop().run_in_executor(
            None, build_client)
        try:
            while True:
                try:
                    await mq_engine_client.setup()
                    break
                except TimeoutError:
                    if (not engine_process.is_alive()
                            or not engine_alive.value):
                        raise RuntimeError(
                            "Engine process failed to start. See stack "
                            "trace for the root cause.") from None

            yield mq_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mq_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(engine_process.pid)


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import (CollectorRegistry, make_asgi_app,
                                   multiprocess)

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.debug("vLLM to use %s as PROMETHEUS_MULTIPROC_DIR",
                     prometheus_multiproc_dir_path)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.openai_serving_completion


def embedding(request: Request) -> Optional[OpenAIServingEmbedding]:
    return request.app.state.openai_serving_embedding


def score(request: Request) -> Optional[OpenAIServingScores]:
    return request.app.state.openai_serving_scores


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_tokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_detokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = base(raw_request)

    models = await handler.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API")

    generator = await handler.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    handler = embedding(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Embeddings API")

    generator = await handler.create_embedding(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/score")
async def create_score(request: ScoreRequest, raw_request: Request):
    handler = score(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Score API")

    generator = await handler.create_score(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, ScoreResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/score")
async def create_score_v1(request: ScoreRequest, raw_request: Request):
    logger.warning(
        "To indicate that Score API is not part of standard OpenAI API, we "
        "have moved it to `/score`. Please update your client accordingly.")

    return await create_score(request, raw_request)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!")

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
        "Lora dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!")

    @router.post("/v1/load_lora_adapter")
    async def load_lora_adapter(request: LoadLoraAdapterRequest,
                                raw_request: Request):
        for route in [chat, completion, embedding]:
            handler = route(raw_request)
            if handler is not None:
                response = await handler.load_lora_adapter(request)
                if isinstance(response, ErrorResponse):
                    return JSONResponse(content=response.model_dump(),
                                        status_code=response.code)

        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(request: UnloadLoraAdapterRequest,
                                  raw_request: Request):
        for route in [chat, completion, embedding]:
            handler = route(raw_request)
            if handler is not None:
                response = await handler.unload_lora_adapter(request)
                if isinstance(response, ErrorResponse):
                    return JSONResponse(content=response.model_dump(),
                                        status_code=response.code)

        return Response(status_code=200, content=response)


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = ErrorResponse(message=str(exc),
                            type="BadRequestError",
                            code=HTTPStatus.BAD_REQUEST)
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path
            if app.root_path and url_path.startswith(app.root_path):
                url_path = url_path[len(app.root_path):]
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
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

    resolved_chat_template = load_chat_template(args.chat_template)
    logger.info("Using supplied chat template:\n%s", resolved_chat_template)

    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        base_model_paths,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        base_model_paths,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    ) if model_config.runner_type == "pooling" else None
    state.openai_serving_scores = OpenAIServingScores(
        engine_client,
        model_config,
        base_model_paths,
        request_logger=request_logger
    ) if (model_config.runner_type == "pooling" \
          and model_config.is_cross_encoder) else None
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    )


def create_server_socket(addr: Tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)

    return sock


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valide_tool_parses)} }})")

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
