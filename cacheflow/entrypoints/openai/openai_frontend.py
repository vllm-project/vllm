import asyncio
import argparse
import asyncio
import json
import time
from typing import Generator, Optional, Union, Dict, List, Any
from http import HTTPStatus

import fastapi
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from cacheflow.server.arg_utils import (
    add_server_arguments, create_server_configs_from_args)
from cacheflow.server.async_llm_server import AsyncLLMServer
from cacheflow.server.ray_utils import initialize_cluster
from cacheflow.logger import init_logger
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import random_uuid


from cacheflow.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorCode,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)


logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message).dict(),
        status_code=status_code.value
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret

@app.get("/v1/models")
async def show_available_models():
    model_cards = [ModelCard(id=served_model, root=served_model,
                             permission=[ModelPermission()])]
    return ModelList(data=model_cards)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    prompt = request.prompt
    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
            # TODO(zhuohan): support logit_bias
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = server.generate(prompt, sampling_params,
                                       request_id=request_id)

    async def generate_completion_stream_generator():
        previous_texts = [""] * request.n
        async for res in result_generator:
            for i, output in enumerate(res.outputs):
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text=delta_text,
                    logprobs=None,
                    finish_reason=None,
                )
                response = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model = model_name,
                    choices=[choice_data],
                )
                yield f"data: {response.json(exclude_unset=True, ensure_ascii=False)}\n\n"
                if output.finish_reason is not None:
                    choice_data = CompletionResponseStreamChoice(
                        index=i,
                        text="",
                        logprobs=None,
                        finish_reason=output.finish_reason,
                    )
                    response = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[choice_data],
                    )
                    yield f"data: {response.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    if request.stream:
        generator = generate_completion_stream_generator()
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        raise NotImplementedError("Not implemented yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CacheFlow OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser = add_server_arguments(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    served_model = args.model

    server_configs = create_server_configs_from_args(args)
    parallel_config = server_configs[2]
    distributed_init_method, stage_devices = initialize_cluster(parallel_config)

    server = AsyncLLMServer(
        args.use_ray, *server_configs, distributed_init_method, stage_devices)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
