import asyncio
import importlib
import inspect
import json
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, List, Tuple

import fastapi
import uvicorn
from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from prometheus_client import make_asgi_app

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.kobold.protocol import KAIGenerationInputSchema
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import _SAMPLING_EPS, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
kai_api = APIRouter()
extra_api = APIRouter()
kobold_lite_ui = ""
gen_cache: dict = {}
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@app.post("/v1/chat/completions")
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


@app.post("/v1/completions")
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


# ============ KoboldAI API ============ #


def prepare_engine_payload(
        kai_payload: KAIGenerationInputSchema
) -> Tuple[SamplingParams, List[int]]:
    """Create SamplingParams and truncated input tokens for AsyncEngine"""

    if not kai_payload.genkey:
        kai_payload.genkey = f"kai-{random_uuid()}"

    kai_payload.top_k = kai_payload.top_k if kai_payload.top_k != 0.0 else -1
    if kai_payload.temperature < _SAMPLING_EPS:
        kai_payload.n = 1
        kai_payload.top_p = 1.0
        kai_payload.top_k = -1

    sampling_params = SamplingParams(
        n=kai_payload.n,
        best_of=kai_payload.n,
        repetition_penalty=kai_payload.rep_pen,
        temperature=kai_payload.temperature,
        top_p=kai_payload.top_p,
        top_k=kai_payload.top_k,
        min_p=kai_payload.min_p,
        stop=kai_payload.stop_sequence,
        include_stop_str_in_output=kai_payload.include_stop_str_in_output,
        max_tokens=kai_payload.max_length,
        seed=kai_payload.sampler_seed,
    )

    max_input_tokens = max(
        1, kai_payload.max_context_length - kai_payload.max_length)
    input_tokens = tokenizer(kai_payload.prompt).input_ids[-max_input_tokens:]

    return sampling_params, input_tokens


@kai_api.post("/generate")
async def generate(kai_payload: KAIGenerationInputSchema) -> JSONResponse:
    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    result_generator = engine.generate(None, sampling_params,
                                       kai_payload.genkey, input_tokens)

    final_res: RequestOutput = None
    previous_output = ""
    async for res in result_generator:
        final_res = res
        new_chunk = res.outputs[0].text[len(previous_output):]
        previous_output += new_chunk
        gen_cache[kai_payload.genkey] = previous_output

    assert final_res is not None
    del gen_cache[kai_payload.genkey]

    return JSONResponse(
        {"results": [{
            "text": output.text
        } for output in final_res.outputs]})


@extra_api.post("/generate/stream")
async def generate_stream(
        kai_payload: KAIGenerationInputSchema) -> StreamingResponse:

    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    results_generator = engine.generate(None, sampling_params,
                                        kai_payload.genkey, input_tokens)

    async def stream_kobold() -> AsyncGenerator[bytes, None]:
        previous_output = ""
        async for res in results_generator:
            new_chunk = res.outputs[0].text[len(previous_output):]
            previous_output += new_chunk
            yield b"event: message\n"
            yield f"data: {json.dumps({'token': new_chunk})}\n\n".encode()

    return StreamingResponse(stream_kobold(),
                             headers={
                                 "Cache-Control": "no-cache",
                                 "Connection": "keep-alive",
                             },
                             media_type="text/event-stream")


@extra_api.post("/generate/check")
@extra_api.get("/generate/check")
async def check_generation(request: Request):
    text = ""
    try:
        request_dict = await request.json()
        if "genkey" in request_dict and request_dict["genkey"] in gen_cache:
            text = gen_cache[request_dict["genkey"]]
    except json.JSONDecodeError:
        pass

    return JSONResponse({"results": [{"text": text}]})


@extra_api.post("/abort")
async def abort_generation(request: Request):
    try:
        request_dict = await request.json()
        if "genkey" in request_dict:
            await engine.abort(request_dict["genkey"])
    except json.JSONDecodeError:
        pass

    return JSONResponse({})


@extra_api.post("/tokencount")
async def count_tokens(request: Request):
    """Tokenize string and return token count"""

    request_dict = await request.json()
    tokenizer_result = await openai_serving_chat.tokenize(
        request_dict["prompt"])
    return JSONResponse({"value": len(tokenizer_result)})


@kai_api.get("/info/version")
async def get_version():
    """Impersonate KAI"""
    return JSONResponse({"result": "1.2.4"})


@kai_api.get("/model")
async def get_model():
    return JSONResponse({"result": f"vllm/{served_model}"})


@kai_api.get("/config/soft_prompts_list")
async def get_available_softprompts():
    """Stub for compatibility"""
    return JSONResponse({"values": []})


@kai_api.get("/config/soft_prompt")
async def get_current_softprompt():
    """Stub for compatibility"""
    return JSONResponse({"value": ""})


@kai_api.put("/config/soft_prompt")
async def set_current_softprompt():
    """Stub for compatibility"""
    return JSONResponse({})


@kai_api.get("/config/max_length")
async def get_max_length() -> JSONResponse:
    max_length = args.max_length
    return JSONResponse({"value": max_length})


@kai_api.get("/config/max_context_length")
@extra_api.get("/true_max_context_length")
async def get_max_context_length() -> JSONResponse:
    max_context_length = engine_args.max_model_len
    return JSONResponse({"value": max_context_length})


@extra_api.get("/preloadstory")
async def get_preloaded_story() -> JSONResponse:
    """Stub for compatibility"""
    return JSONResponse({})


@extra_api.get("/version")
async def get_extra_version():
    """Impersonate KoboldCpp"""
    return JSONResponse({"result": "KoboldCpp", "version": "1.60.1"})


@app.get("/")
async def get_kobold_lite_ui():
    """Serves a cached copy of the Kobold Lite UI, loading it from disk
    on demand if needed."""
    global kobold_lite_ui
    if kobold_lite_ui == "":
        scriptpath = os.path.dirname(os.path.abspath(__file__))
        klitepath = os.path.join(scriptpath, "klite.embd")
        klitepath = os.path.normpath(klitepath)  # Normalize the path
        if os.path.exists(klitepath):
            with open(klitepath, "r") as f:
                kobold_lite_ui = f.read()
        else:
            logger.error("Kobold Lite UI not found at " + klitepath)
    return HTMLResponse(content=kobold_lite_ui)


if __name__ == "__main__":
    args = parse_args()

    logger.warning("Launching Kobold API server in addition to OpenAI. "
                   "Keep in mind that the Kobold API routes are NOT "
                   "protected via the API key.")
    app.include_router(kai_api, prefix="/api/v1")
    app.include_router(kai_api, prefix="/api/latest", include_in_schema=False)
    app.include_router(extra_api, prefix="/api/extra")

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
        served_model = args.served_model_name
    else:
        served_model = args.model
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=engine_args.trust_remote_code,
    )

    openai_serving_chat = OpenAIServingChat(engine, served_model,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model, args.lora_modules)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)