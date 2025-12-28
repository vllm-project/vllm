# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import signal
import socket
from http import HTTPStatus
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils.network_utils import find_process_using_port
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)


class MetricsEndpointFilter(logging.Filter):
    """Filter to exclude /metrics and /health endpoints from access logs.

    This reduces log noise from monitoring systems that frequently poll
    these endpoints.
    """

    # Endpoints to exclude from access logs
    EXCLUDED_PATHS = {"/metrics", "/health"}

    def filter(self, record: logging.LogRecord) -> bool:
        path = self._extract_path(record)
        # Return False to exclude, True to include
        return not (path and path in self.EXCLUDED_PATHS)

    def _extract_path(self, record: logging.LogRecord) -> str | None:
        """Extract the request path from a uvicorn access log record.

        Handles different uvicorn versions:
        - Tuple format with path at index 2: (client, method, path, ...)
        - Tuple format with request_line at index 1
        - Dictionary format with 'request_line' key
        """
        args = record.args
        if not args:
            return None

        # Handle dictionary format (newer uvicorn versions)
        if isinstance(args, dict):
            request_line = args.get("request_line", "")
            if isinstance(request_line, str):
                return self._parse_path_from_request_line(request_line)
            return None

        # Handle tuple format
        if isinstance(args, tuple):
            # Try path at index 2 first (common format)
            if len(args) >= 3:
                path = args[2]
                if isinstance(path, str):
                    # Check if it looks like a path (starts with /)
                    if path.startswith("/"):
                        return path
                    # It might be a request line like "GET /path HTTP/1.1"
                    return self._parse_path_from_request_line(path)

            # Try request_line at index 1 (older format)
            if len(args) >= 2:
                request_line = args[1]
                if isinstance(request_line, str):
                    return self._parse_path_from_request_line(request_line)

        return None

    def _parse_path_from_request_line(self, request_line: str) -> str | None:
        """Parse path from HTTP request line like 'GET /path HTTP/1.1'."""
        if not request_line:
            return None
        try:
            parts = request_line.split()
            if len(parts) >= 2:
                path = parts[1]
                # Remove query string if present
                if "?" in path:
                    path = path.split("?")[0]
                return path
        except (AttributeError, IndexError):
            pass
        return None


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    disable_metrics_access_log: bool = False,
    **uvicorn_kwargs: Any,
):
    """
    Start a FastAPI app using Uvicorn, with support for custom Uvicorn config
    options.  Supports http header limits via h11_max_incomplete_event_size and
    h11_max_header_count.

    Args:
        disable_metrics_access_log: If True, filter out access logs for
            /metrics and /health endpoints to reduce log noise from
            monitoring systems.
    """
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    # Extract header limit options if present
    h11_max_incomplete_event_size = uvicorn_kwargs.pop(
        "h11_max_incomplete_event_size", None
    )
    h11_max_header_count = uvicorn_kwargs.pop("h11_max_header_count", None)

    # Set safe defaults if not provided
    if h11_max_incomplete_event_size is None:
        h11_max_incomplete_event_size = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    if h11_max_header_count is None:
        h11_max_header_count = H11_MAX_HEADER_COUNT_DEFAULT

    config = uvicorn.Config(app, **uvicorn_kwargs)
    # Set header limits
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    config.load()

    # Apply filter to exclude /metrics and /health from access logs
    if disable_metrics_access_log:
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        uvicorn_access_logger.addFilter(MetricsEndpointFilter())
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    ssl_cert_refresher = (
        None
        if not enable_ssl_refresh
        else SSLCertRefresher(
            ssl_context=config.ssl,
            key_path=config.ssl_keyfile,
            cert_path=config.ssl_certfile,
            ca_path=config.ssl_ca_certs,
        )
    )

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.warning(
                "port %s is used by process %s launched with command:\n%s",
                port,
                process,
                " ".join(process.cmdline()),
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()
    finally:
        watchdog_task.cancel()


async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    """
    # Watchdog task that runs in the background, checking
    # for error state in the engine. Needed to trigger shutdown
    # if an exception arises is StreamingResponse() generator.
    """
    VLLM_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)


def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
    """
    See discussions here on shutting down a uvicorn server
    https://github.com/encode/uvicorn/discussions/1103
    In this case we cannot await the server shutdown here
    because handler must first return to close the connection
    for this request.
    """
    engine_errored = engine.errored and not engine.is_running
    if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
        server.should_exit = True


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
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

    @app.exception_handler(RuntimeError)
    @app.exception_handler(EngineDeadError)
    @app.exception_handler(EngineGenerateError)
    async def runtime_exception_handler(request: Request, __):
        terminate_if_errored(
            server=server,
            engine=request.app.state.engine_client,
        )

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
