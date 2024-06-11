"""
How to run:
from root directory run command
python3 <path_to_file>/standalone_api_server --host 0.0.0.0 --port 8000 --log-level info --workers 4
"""

import asyncio
import importlib
import inspect
import os

from contextlib import asynccontextmanager
from http import HTTPStatus

import fastapi
import uvicorn
from fastapi import Request, FastAPI, APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.middleware.cors import CORSMiddleware

import vllm
from vllm import AsyncLLMEngine, AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
engine: AsyncLLMEngine = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    # if not engine_args.disable_log_stats:
    #     asyncio.create_task(_force_log())

    yield


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


router = APIRouter()
app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return JSONResponse({"status": "ok"}, status_code=200)


@router.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


def get_app(*args) -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    metrics_app = make_asgi_app()

    app.mount("/metrics", metrics_app)
    app.include_router(router=router)

    return app


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

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

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    openai_serving_chat = OpenAIServingChat(engine, served_model_names,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names, args.lora_modules)

    app.root_path = args.root_path
    if args.workers > 1:
        uvicorn.run("vllm.entrypoints.openai.standalone_api_server:get_app",
                    host=args.host,
                    port=args.port,
                    log_level=args.uvicorn_log_level,
                    ssl_keyfile=args.ssl_keyfile,
                    ssl_certfile=args.ssl_certfile,
                    ssl_ca_certs=args.ssl_ca_certs,
                    ssl_cert_reqs=args.ssl_cert_reqs,
                    workers=args.workers)
    else:
        uvicorn.run("vllm.entrypoints.openai.api_server:app",
                    host=args.host,
                    port=args.port,
                    log_level=args.uvicorn_log_level,
                    ssl_keyfile=args.ssl_keyfile,
                    ssl_certfile=args.ssl_certfile,
                    ssl_ca_certs=args.ssl_ca_certs,
                    ssl_cert_reqs=args.ssl_cert_reqs)
