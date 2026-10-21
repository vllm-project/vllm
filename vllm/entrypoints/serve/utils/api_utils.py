# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import functools
from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

import orjson
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask, BackgroundTasks

from vllm.entrypoints.generate.base.protocol import StreamOptions
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def create_json_response(
    result: BaseModel | Sequence[BaseModel], *, status_code: int = HTTPStatus.OK
) -> Response:
    """Encode models with orjson, preserving error status and JSONResponse fallback."""
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    content = (
        result.model_dump()
        if isinstance(result, BaseModel)
        else [item.model_dump() for item in result]
    )
    try:
        body = orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)
    except TypeError:
        # Preserve JSONResponse support for integers outside the 64-bit range.
        return JSONResponse(content=content, status_code=status_code)
    return Response(
        content=body, status_code=status_code, media_type="application/json"
    )


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            # If load tracking is enabled *and* the counter exists, decrement
            # it. Combines the previous nested checks into a single condition
            # to satisfy the linter rule.
            if getattr(
                request.app.state, "enable_server_load_tracking", False
            ) and hasattr(request.app.state, "server_load_metrics"):
                request.app.state.server_load_metrics -= 1
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait(
            [handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        # The client is gone; an empty response skips response_model validation.
        return Response()

    return wrapper


_ERROR_REF = {"$ref": "#/components/schemas/ErrorResponse"}


def _sse_event_schema(chunk_ref: dict[str, str]) -> dict[str, Any]:
    # One SSE event: `data` is a JSON chunk, a JSON error, or "[DONE]". This is
    # the envelope FastAPI emits for EventSourceResponse routes, but published
    # under `schema` because OpenAPI 3.1 has no `itemSchema`.
    return {
        "type": "object",
        "required": ["data"],
        "properties": {
            "data": {
                "type": "string",
                "anyOf": [
                    {"const": "[DONE]"},
                    {
                        "contentMediaType": "application/json",
                        "contentSchema": chunk_ref,
                    },
                    {
                        "contentMediaType": "application/json",
                        "contentSchema": _ERROR_REF,
                    },
                ],
            },
            "event": {"type": "string"},
            "id": {"type": "string"},
            "retry": {"type": "integer", "minimum": 0},
        },
    }


def json_or_sse_responses(
    unary_model: type[BaseModel],
    chunk_model: type[BaseModel],
    error_responses: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Route kwargs for an endpoint answering with ``unary_model`` as JSON, or
    with SSE ``data`` events carrying ``chunk_model`` when ``stream=true``.

    FastAPI only registers models reachable from ``response_model`` or an
    additional response's ``model``. The chunk model rides on the 200 entry,
    which also overwrites the JSON schema, so ``openapi_extra`` restores it.
    """
    unary_ref = {"$ref": f"#/components/schemas/{unary_model.__name__}"}
    chunk_ref = {"$ref": f"#/components/schemas/{chunk_model.__name__}"}
    return {
        "response_model": unary_model,
        "responses": {
            HTTPStatus.OK.value: {
                "model": chunk_model,
                "description": (
                    "JSON body when stream=false; SSE events when stream=true."
                ),
                "content": {
                    "text/event-stream": {"schema": _sse_event_schema(chunk_ref)}
                },
            },
            **error_responses,
        },
        "openapi_extra": {
            "responses": {
                "200": {"content": {"application/json": {"schema": unary_ref}}}
            }
        },
    }


def decrement_server_load(request: Request):
    request.app.state.server_load_metrics -= 1


def load_aware_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        raw_request = kwargs.get("raw_request", args[1] if len(args) > 1 else None)

        if raw_request is None:
            raise ValueError(
                "raw_request required when server load tracking is enabled"
            )

        if not getattr(raw_request.app.state, "enable_server_load_tracking", False):
            return await func(*args, **kwargs)

        # ensure the counter exists
        if not hasattr(raw_request.app.state, "server_load_metrics"):
            raw_request.app.state.server_load_metrics = 0

        raw_request.app.state.server_load_metrics += 1
        try:
            response = await func(*args, **kwargs)
        except Exception:
            raw_request.app.state.server_load_metrics -= 1
            raise

        if isinstance(response, (JSONResponse, StreamingResponse)):
            if response.background is None:
                response.background = BackgroundTask(decrement_server_load, raw_request)
            elif isinstance(response.background, BackgroundTasks):
                response.background.add_task(decrement_server_load, raw_request)
            elif isinstance(response.background, BackgroundTask):
                # Convert the single BackgroundTask to BackgroundTasks
                # and chain the decrement_server_load task to it
                tasks = BackgroundTasks()
                tasks.add_task(
                    response.background.func,
                    *response.background.args,
                    **response.background.kwargs,
                )
                tasks.add_task(decrement_server_load, raw_request)
                response.background = tasks
        else:
            raw_request.app.state.server_load_metrics -= 1

        return response

    return wrapper


def get_max_tokens(
    max_model_len: int,
    max_tokens: int | None,
    input_length: int,
    default_sampling_params: dict,
    override_max_tokens: int | None = None,
    truncate_prompt_tokens: int | None = None,
) -> int:
    if truncate_prompt_tokens is not None:
        limit = truncate_prompt_tokens
        input_length = min(
            input_length,
            max_model_len if limit == -1 else limit,
        )
    if max_model_len < input_length:
        raise ValueError(
            f"Input length ({input_length}) exceeds model's maximum "
            f"context length ({max_model_len})."
        )
    model_max_tokens = max_model_len - input_length
    platform_max_tokens = current_platform.get_max_output_tokens(input_length)
    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )

    return min(
        val
        for val in (
            model_max_tokens,
            fallback_max_tokens,
            override_max_tokens,
            platform_max_tokens,
        )
        if val is not None
    )


def should_include_usage(
    stream_options: StreamOptions | None, enable_force_include_usage: bool
) -> tuple[bool, bool]:
    if enable_force_include_usage:
        return True, True
    if stream_options:
        include_usage = bool(stream_options.include_usage)
        include_continuous_usage = include_usage and bool(
            stream_options.continuous_usage_stats
        )
    else:
        include_usage, include_continuous_usage = False, False
    return include_usage, include_continuous_usage


def process_lora_modules(
    args_lora_modules: list[LoRAModulePath], default_mm_loras: dict[str, str] | None
) -> list[LoRAModulePath]:
    from vllm.entrypoints.openai.models.serving import LoRAModulePath

    lora_modules = args_lora_modules
    if default_mm_loras:
        default_mm_lora_paths = [
            LoRAModulePath(
                name=modality,
                path=lora_path,
            )
            for modality, lora_path in default_mm_loras.items()
        ]
        if args_lora_modules is None:
            lora_modules = default_mm_lora_paths
        else:
            lora_modules += default_mm_lora_paths
    return lora_modules


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=["Unsupported Media Type: Only 'application/json' is allowed"]
        )
