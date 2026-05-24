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

import hvac
import pydantic
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import terminate_if_errored
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    GenerationError,
)
from vllm.entrypoints.utils import create_error_response, sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger("vllm.entrypoints.openai.server_utils")


GUARDED_PREFIX = ("/v1", "/v2", "/inference")


class AuthenticationMiddleware:
    """
    Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health).
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
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


class VaultAuthenticationMiddleware:
    """
    HVAC library usage to check a passed token against a Hashicorp Vault path.

    ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals secret at set "{vault_path}"

    Notes
    -----
    Uses 'hvac' library to access Vault.
    Requires values set including:
        - vault_token
        - vault_secret_path (including mount point)
        - vault_key
        - vault_url

    Follows existing AuthenticationMiddleware pattern and skips two cases:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with /v1 (e.g. /health).

    TTL Cache = 60 seconds

    Employ's asyncio.Lock()

    """

    def __init__(
        self,
        app: ASGIApp,
        vault_url,
        vault_token,
        secret_path,
        vault_key,
        mount_point="secret",
        cache_ttl: int = 60,
    ):
        """
        :param vault_url: The full URL to your Vault server.
        :param vault_token: A token with read permissions for the secret_path.
        :param secret_path: The path to the secret (e.g., 'myapp/api-keys').
        :param vault_key: key in the returned object that the token is in
        :param mount_point: The KV engine mount point (default is 'secret' for KV V2).
        :param cache: TTL cache for tokens
        """

        self.app = app
        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.secret_path = secret_path
        self.key = vault_key
        self.mount_point = mount_point
        self.cache: TTLCache[str, bytes] = TTLCache(maxsize=1, ttl=cache_ttl)
        self._lock = asyncio.Lock()

    def _get_vault_secret_sync(self) -> bytes | None:
        """
        Synchronous process to check with Vault
        Uses a cache and the short TTL
        """
        cached_token = self.cache.get("vault_token")  # simple cache name
        if cached_token:
            return cached_token

        try:
            read_response = self.client.secrets.kv.v2.read_secret_version(
                path=self.secret_path, mount_point=self.mount_point
            )
            expected_token = read_response["data"]["data"].get(self.key)

            if expected_token:
                self.cache["vault_token"] = hashlib.sha256(
                    expected_token.encode("utf-8")
                ).digest()
                return self.cache["vault_token"]

            logger.error(
                "Key '%s' not found in Vault path '%s'", self.key, self.secret_path
            )
        except hvac.exceptions.Forbidden:
            logger.error("Vault access denied. Check token permissions.")
        except Exception as e:
            logger.error("Unexpected Vault error: %s", e)

        return None

    async def verify_token(self, headers: Headers) -> bool:
        auth_header = headers.get("Authorization")
        if not auth_header:
            return False

        scheme, _, param = auth_header.partition(" ")
        if scheme.lower() != "bearer":
            return False

        # Cache check
        expected_token = self.cache.get("vault_token")

        if not expected_token:
            # Use lock to ensure only one thread-offload happens
            async with self._lock:
                # Check cache inside lock in case another request filled it
                expected_token = self.cache.get("vault_token")
                if not expected_token:
                    expected_token = await asyncio.to_thread(
                        self._get_vault_secret_sync
                    )  # handle blocking in async

        if not expected_token:
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()
        return secrets.compare_digest(param_hash, expected_token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return

        # get path for filtering
        root_path = scope.get("root_path", "")
        url_path = URL(scope=scope).path.removeprefix(root_path)

        # authenticate /v1 requests
        if url_path.startswith(GUARDED_PREFIX):
            headers = Headers(scope=scope)
            authenticated = await self.verify_token(headers)

            if not authenticated:
                response = JSONResponse(
                    content={"error": "Unauthorized"}, status_code=401
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


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


def get_uvicorn_log_config(args: Namespace) -> dict | None:
    """
    Get the uvicorn log config based on the provided arguments.

    Priority:
    1. If log_config_file is specified, use it
    2. If disable_access_log_for_endpoints is specified, create a config with
       the access log filter
    3. Otherwise, return None (use uvicorn defaults)
    """
    # First, try to load from file if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        return log_config

    # If endpoints to filter are specified, create a config with the filter
    if args.disable_access_log_for_endpoints:
        from vllm.logging_utils import create_uvicorn_log_config

        # Parse comma-separated string into list
        excluded_paths = [
            p.strip()
            for p in args.disable_access_log_for_endpoints.split(",")
            if p.strip()
        ]
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


async def engine_error_handler(
    req: Request, exc: EngineDeadError | EngineGenerateError
):
    """
    VLLM V1 AsyncLLM catches exceptions and returns
    only two types: EngineGenerateError and EngineDeadError.

    EngineGenerateError is raised by the per request generate()
    method. This error could be request specific (and therefore
    recoverable - e.g. if there is an error in input processing).

    EngineDeadError is raised by the background output_handler
    method. This error is global and therefore not recoverable.

    We register these @app.exception_handlers to return nice
    responses to the end user if they occur and shut down if needed.
    See https://fastapi.tiangolo.com/tutorial/handling-errors/
    for more details on how exception handlers work.

    If an exception is encountered in a StreamingResponse
    generator, the exception is not raised, since we already sent
    a 200 status. Rather, we send an error message as the next chunk.
    Since the exception is not raised, this means that the server
    will not automatically shut down. Instead, we use the watchdog
    background task for check for errored state.
    """

    if req.app.state.args.log_error_stack:
        logger.exception(
            "Engine Exception caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    terminate_if_errored(
        server=req.app.state.server,
        engine=req.app.state.engine_client,
    )
    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)


async def generation_error_handler(req: Request, exc: GenerationError):
    """Handle GenerationError without logging stack traces.

    GenerationError is a known, expected error (e.g. KV cache load failure)
    that should be returned to the client as a 500 response without polluting
    server logs with stack traces.
    """
    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)


async def exception_handler(req: Request, exc: Exception):
    if req.app.state.args.log_error_stack:
        logger.error(
            "Exception caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    err = create_error_response(exc)
    return JSONResponse(err.model_dump(), status_code=err.error.code)


async def http_exception_handler(req: Request, exc: HTTPException):
    if req.app.state.args.log_error_stack:
        logger.exception(
            "HTTPException caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )
    err = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(exc.detail),
            type=HTTPStatus(exc.status_code).phrase,
            code=exc.status_code,
        )
    )
    return JSONResponse(err.model_dump(), status_code=exc.status_code)


async def validation_exception_handler(req: Request, exc: RequestValidationError):
    if req.app.state.args.log_error_stack:
        logger.exception(
            "RequestValidationError caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

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
