"""
How to run:
from root directory run command
uvicorn vllm.entrypoints.openai.standalone_api_server:get_app --host 0.0.0.0 --port 8000 --log-level info
"""

import asyncio

from contextlib import asynccontextmanager
from http import HTTPStatus

import fastapi
from fastapi import Request, FastAPI, APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app

import vllm
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    yield


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


router = APIRouter()
app = FastAPI(lifespan=lifespan)


# Add prometheus asgi middleware to route /metrics requests


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


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    metrics_app = make_asgi_app()

    app.mount("/metrics", metrics_app)
    app.include_router(router=router)
    # TODO: add engine

    return app
