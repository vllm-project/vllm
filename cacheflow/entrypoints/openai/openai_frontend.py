import asyncio
import argparse
import asyncio
import json
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
        ErrorCode.INVALID_MODEL,
        f"The model `{request.model}` does not exist.",
    )
    return ret

@app.get("/v1/models")
async def show_available_models():
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        ret = await client.post(controller_address + "/refresh_all_workers")
        ret = await client.post(controller_address + "/list_models")
    models = ret.json()["models"]
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs if request.logprobs else 0,
            use_beam_search=request.use_beam_search,
            # TODO: support stop, best_of, and logit_bias
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if request.stream:
        generator = generate_completion_stream_generator(payload, request.n)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for i in range(request.n):
            content = asyncio.create_task(generate_completion(payload))
            text_completions.append(content)

        try:
            all_tasks = await asyncio.gather(*text_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

        choices = []
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                CompletionResponseChoice(
                    index=i,
                    text=content["text"],
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
        )


async def generate_completion_stream_generator(payload: Dict[str, Any], n: int):
    model_name = payload["model"]
    id = f"cmpl-{random_uuid()}"
    finish_stream_events = []
    for i in range(n):
        previous_text = ""
        async for content in generate_completion_stream(payload):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            choice_data = CompletionResponseStreamChoice(
                index=i,
                text=delta_text,
                logprobs=content.get("logprobs", None),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = CompletionStreamResponse(
                id=id, object="text_completion", choices=[choice_data], model=model_name
            )
            if len(delta_text) == 0:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_completion_stream(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(payload["model"], client)

        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_completion_stream",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        ) as response:
            # content = await response.aread()
            async for raw_chunk in response.aiter_raw():
                for chunk in raw_chunk.split(delimiter):
                    if not chunk:
                        continue
                    data = json.loads(chunk.decode())
                    yield data


async def generate_completion(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(payload["model"], client)

        response = await client.post(
            worker_addr + "/worker_generate_completion",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        completion = response.json()
        return completion


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
