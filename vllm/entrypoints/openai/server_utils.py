# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import hashlib
import json
import secrets
import uuid
from argparse import Namespace
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from http import HTTPStatus

import pydantic
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.utils import sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.utils.gc_utils import freeze_gc_heap

logger = init_logger("vllm.entrypoints.openai.server_utils")


class AuthenticationMiddleware:
    """
    Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with /v1 (e.g. /health).
    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket") or scope["method"] == "OPTIONS":
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = URL(scope=scope).path.removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith("/v1") and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


class XRequestIdMiddleware:
    """
    Middleware the set's the X-Request-Id header for each response
    to a random uuid4 (hex) value if the header isn't already
    present in the request, otherwise use the provided request id.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            return self.app(scope, receive, send)

        # Extract the request headers.
        request_headers = Headers(scope=scope)

        async def send_with_request_id(message: Message) -> None:
            """
            Custom send function to mutate the response headers
            and append X-Request-Id to it.
            """
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(raw=message["headers"])
                request_id = request_headers.get("X-Request-Id", uuid.uuid4().hex)
                response_headers.append("X-Request-Id", request_id)
            await send(message)

        return self.app(scope, receive, send_with_request_id)


def load_log_config(log_config_file: str | None) -> dict | None:
    if not log_config_file:
        return None
    try:
        with open(log_config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(
            "Failed to load log config from file %s: error %s", log_config_file, e
        )
        return None


_METRICS_EXCLUDED_PATHS = ["/health", "/metrics"]


def _get_excluded_paths(args: Namespace) -> list[str]:
    """Collect endpoint paths that should be excluded from access logs.

    Merges paths from ``--disable-uvicorn-metrics-access-log`` (which
    implies ``/health`` and ``/metrics``) with any paths listed in
    ``--disable-access-log-for-endpoints``, removing duplicates while
    preserving order.
    """
    paths: list[str] = []

    if getattr(args, "disable_uvicorn_metrics_access_log", False):
        paths.extend(_METRICS_EXCLUDED_PATHS)

    if args.disable_access_log_for_endpoints:
        for p in args.disable_access_log_for_endpoints.split(","):
            p = p.strip()
            if p and p not in paths:
                paths.append(p)

    return paths


def get_uvicorn_log_config(args: Namespace) -> dict | None:
    """
    Get the uvicorn log config based on the provided arguments.

    Priority:
    1. If log_config_file is specified, use it
    2. If endpoints to exclude are specified (via
       --disable-uvicorn-metrics-access-log or
       --disable-access-log-for-endpoints), create a config with the
       access log filter
    3. Otherwise, return None (use uvicorn defaults)
    """
    # First, try to load from file if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        return log_config

    # Collect excluded paths from both options
    excluded_paths = _get_excluded_paths(args)
    if excluded_paths:
        from vllm.logging_utils import create_uvicorn_log_config

        return create_uvicorn_log_config(
            excluded_paths=excluded_paths,
            log_level=args.uvicorn_log_level,
        )

    return None


def _extract_content_from_chunk(chunk_data: dict) -> str:
    """Extract content from a streaming response chunk."""
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionStreamResponse,
        )
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionStreamResponse,
        )

        # Try using Completion types for type-safe parsing
        if chunk_data.get("object") == "chat.completion.chunk":
            chat_response = ChatCompletionStreamResponse.model_validate(chunk_data)
            if chat_response.choices and chat_response.choices[0].delta.content:
                return chat_response.choices[0].delta.content
        elif chunk_data.get("object") == "text_completion":
            completion_response = CompletionStreamResponse.model_validate(chunk_data)
            if completion_response.choices and completion_response.choices[0].text:
                return completion_response.choices[0].text
    except pydantic.ValidationError:
        # Fallback to manual parsing
        if "choices" in chunk_data and chunk_data["choices"]:
            choice = chunk_data["choices"][0]
            if "delta" in choice and choice["delta"].get("content"):
                return choice["delta"]["content"]
            elif choice.get("text"):
                return choice["text"]
    return ""


class SSEDecoder:
    """Robust Server-Sent Events decoder for streaming responses."""

    def __init__(self):
        self.buffer = ""
        self.content_buffer = []

    def decode_chunk(self, chunk: bytes) -> list[dict]:
        """Decode a chunk of SSE data and return parsed events."""
        import json

        try:
            chunk_str = chunk.decode("utf-8")
        except UnicodeDecodeError:
            # Skip malformed chunks
            return []

        self.buffer += chunk_str
        events = []

        # Process complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.rstrip("\r")  # Handle CRLF

            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    events.append({"type": "done"})
                elif data_str:
                    try:
                        event_data = json.loads(data_str)
                        events.append({"type": "data", "data": event_data})
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

        return events

    def extract_content(self, event_data: dict) -> str:
        """Extract content from event data."""
        return _extract_content_from_chunk(event_data)

    def add_content(self, content: str) -> None:
        """Add content to the buffer."""
        if content:
            self.content_buffer.append(content)

    def get_complete_content(self) -> str:
        """Get the complete buffered content."""
        return "".join(self.content_buffer)


def _log_streaming_response(response, response_body: list) -> None:
    """Log streaming response with robust SSE parsing."""
    from starlette.concurrency import iterate_in_threadpool

    sse_decoder = SSEDecoder()
    chunk_count = 0

    def buffered_iterator():
        nonlocal chunk_count

        for chunk in response_body:
            chunk_count += 1
            yield chunk

            # Parse SSE events from chunk
            events = sse_decoder.decode_chunk(chunk)

            for event in events:
                if event["type"] == "data":
                    content = sse_decoder.extract_content(event["data"])
                    sse_decoder.add_content(content)
                elif event["type"] == "done":
                    # Log complete content when done
                    full_content = sse_decoder.get_complete_content()
                    if full_content:
                        # Truncate if too long
                        if len(full_content) > 2048:
                            full_content = full_content[:2048] + ""
                            "...[truncated]"
                        logger.info(
                            "response_body={streaming_complete: content=%r, chunks=%d}",
                            full_content,
                            chunk_count,
                        )
                    else:
                        logger.info(
                            "response_body={streaming_complete: no_content, chunks=%d}",
                            chunk_count,
                        )
                    return

    response.body_iterator = iterate_in_threadpool(buffered_iterator())
    logger.info("response_body={streaming_started: chunks=%d}", len(response_body))


def _log_non_streaming_response(response_body: list) -> None:
    """Log non-streaming response."""
    try:
        decoded_body = response_body[0].decode()
        logger.info("response_body={%s}", decoded_body)
    except UnicodeDecodeError:
        logger.info("response_body={<binary_data>}")


async def log_response(request: Request, call_next):
    response = await call_next(request)
    response_body = [section async for section in response.body_iterator]
    response.body_iterator = iterate_in_threadpool(iter(response_body))
    # Check if this is a streaming response by looking at content-type
    content_type = response.headers.get("content-type", "")
    is_streaming = content_type == "text/event-stream; charset=utf-8"

    # Log response body based on type
    if not response_body:
        logger.info("response_body={<empty>}")
    elif is_streaming:
        _log_streaming_response(response, response_body)
    else:
        _log_non_streaming_response(response_body)
    return response


async def http_exception_handler(_: Request, exc: HTTPException):
    err = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(exc.detail),
            type=HTTPStatus(exc.status_code).phrase,
            code=exc.status_code,
        )
    )
    return JSONResponse(err.model_dump(), status_code=exc.status_code)


async def validation_exception_handler(_: Request, exc: RequestValidationError):
    param = None
    errors = exc.errors()
    for error in errors:
        if "ctx" in error and "error" in error["ctx"]:
            ctx_error = error["ctx"]["error"]
            if isinstance(ctx_error, VLLMValidationError):
                param = ctx_error.parameter
                break

    exc_str = str(exc)
    errors_str = str(errors)

    if errors and errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    err = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            type=HTTPStatus.BAD_REQUEST.phrase,
            code=HTTPStatus.BAD_REQUEST,
            param=param,
        )
    )
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


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
        freeze_gc_heap()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state
