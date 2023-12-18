# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import json
from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.exceptions import RequestValidationError
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import add_global_metrics_labels
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving import OpenAIServing
from vllm.logger import init_logger

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving: OpenAIServing = None
logger = init_logger(__name__)
app = fastapi.FastAPI()

def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--chat-template",
                        type=str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


app.add_middleware(MetricsMiddleware)  # Trace HTTP server metrics
app.add_route("/metrics", metrics)  # Exposes HTTP metrics


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    return openai_serving.create_error_response(message=str(exc))

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving.show_available_models()
    return JSONResponse(content=models.dict())
    pass

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    generator = await openai_serving.create_chat_completion(request, raw_request)
    if request.stream and not isinstance(generator, ErrorResponse):
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.dict())

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving.create_completion(request, raw_request)
    if request.stream and not isinstance(generator, ErrorResponse):
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.dict())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    openai_serving = OpenAIServing(engine, served_model, args.response_role, args.chat_template)

    # Register labels for metrics
    add_global_metrics_labels(model_name=engine_args.model)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
