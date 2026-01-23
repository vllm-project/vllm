# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import signal
import socket
import time
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
from vllm.entrypoints.openai.cli_args import FrontendArgs
from vllm.entrypoints.serve.middleware import set_rejecting_requests
from vllm.entrypoints.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils.network_utils import find_process_using_port
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    """
    Start a FastAPI app using Uvicorn, with support for custom Uvicorn config
    options.  Supports http header limits via h11_max_incomplete_event_size and
    h11_max_header_count.
    """
    logger.info("Available routes are:")
    # post endpoints
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    # other endpoints
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if endpoint is None or path is None or methods is not None:
            continue

        logger.info("Route: %s, Endpoint: %s", path, endpoint.__name__)

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

    args = getattr(app.state, "args", None)
    engine_client: EngineClient = app.state.engine_client
    enable_graceful = (
        args is not None and getattr(args, "shutdown_mode", "immediate") == "drain"
    )

    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    if enable_graceful:
        config.install_signal_handlers = False
    config.load()
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(watchdog_loop(server, engine_client))
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

    async def graceful_drain() -> None:
        """Perform graceful drain before shutdown."""
        drain_timeout = getattr(
            args, "shutdown_drain_timeout", FrontendArgs.shutdown_drain_timeout
        )

        inflight_count = engine_client.get_num_unfinished_requests()
        logger.info(
            "Graceful shutdown: draining %d in-flight requests",
            inflight_count,
        )

        set_rejecting_requests(True)

        start_time = time.monotonic()
        try:
            # send graceful shutdown to engines via IPC
            core = getattr(engine_client, "engine_core", None)
            if core is not None and hasattr(core, "_send_graceful_shutdown_to_engines"):
                core._send_graceful_shutdown_to_engines()

                # wait for ready_to_exit event (set when engines finish draining)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            core.resources.ready_to_exit_event.wait, drain_timeout
                        ),
                        timeout=drain_timeout + 1,
                    )

                elapsed = time.monotonic() - start_time
                if core.resources.ready_to_exit_event.is_set():
                    logger.info(
                        "Graceful shutdown: drain complete in %.1fs",
                        elapsed,
                    )
                elif core.resources.engine_dead:
                    logger.warning("Graceful shutdown: engine died during drain")
                else:
                    remaining = engine_client.get_num_unfinished_requests()
                    logger.warning(
                        "Graceful shutdown: drain timed out after %.1fs, "
                        "%d requests remaining, proceeding with shutdown",
                        elapsed,
                        remaining,
                    )
            else:
                # fallback for non-MP clients
                logger.info("Graceful shutdown: no engine core, proceeding")
        except Exception as e:
            logger.warning("Graceful shutdown: drain failed: %s", e)

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    async def graceful_signal_handler() -> None:
        """Async wrapper for graceful shutdown."""
        try:
            await graceful_drain()
        except asyncio.CancelledError:
            logger.info("Graceful drain cancelled, proceeding with immediate shutdown")
        signal_handler()

    shutting_down = False
    graceful_task: asyncio.Task | None = None

    def on_signal() -> None:
        """Signal callback that spawns the graceful shutdown task."""
        nonlocal shutting_down, graceful_task
        if shutting_down:
            if graceful_task is not None and not graceful_task.done():
                logger.warning("Received second signal, forcing immediate shutdown")
                graceful_task.cancel()
            return
        shutting_down = True

        if enable_graceful:
            # reset should_exit to keep uvicorn's serve loop running during drain
            server.should_exit = False

            drain_timeout = getattr(
                args, "shutdown_drain_timeout", FrontendArgs.shutdown_drain_timeout
            )
            logger.info(
                "Graceful shutdown initiated (timeout: %ds). "
                "Send SIGTERM again to force immediate shutdown.",
                drain_timeout,
            )
            graceful_task = loop.create_task(graceful_signal_handler())
        else:
            signal_handler()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, on_signal)
    loop.add_signal_handler(signal.SIGTERM, on_signal)

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
