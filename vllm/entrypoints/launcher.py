# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import signal
import socket
from functools import partial
from typing import Any

import uvicorn
from fastapi import FastAPI

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.serve.utils.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils.network_utils import find_process_using_port

logger = init_logger(__name__)


class _HypercornAdapter:
    """Thin adapter so terminate_if_errored() works uniformly with both
    Hypercorn (asyncio.Event) and Uvicorn (server.should_exit) backends.
    """

    def __init__(self, shutdown_event: asyncio.Event) -> None:
        self._shutdown_event = shutdown_event

    @property
    def should_exit(self) -> bool:
        return self._shutdown_event.is_set()

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        if value:
            self._shutdown_event.set()

    async def shutdown(self) -> None:
        self._shutdown_event.set()


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    enable_http2: bool = False,
    **uvicorn_kwargs: Any,
):
    """
    Start a FastAPI app. When enable_http2=True uses Hypercorn (requires the
    'hypercorn' and 'h2' packages) to serve HTTP/2 with ALPN negotiation,
    falling back to HTTP/1.1 for clients that don't support h2. Otherwise
    uses Uvicorn (HTTP/1.1, default behaviour).
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

    if enable_http2:
        return await _serve_hypercorn(app, sock, enable_ssl_refresh, **uvicorn_kwargs)

    # ------------------------------------------------------------------ #
    # Uvicorn path (HTTP/1.1, default)                                     #
    # ------------------------------------------------------------------ #

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
    server = uvicorn.Server(config)
    app.state.server = server

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

    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        if shutdown_event.is_set():
            return
        logger.info_once("[shutdown] API server: shutdown triggered")
        shutdown_event.set()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async def handle_shutdown() -> None:
        await shutdown_event.wait()

        engine_client = app.state.engine_client
        timeout = engine_client.vllm_config.shutdown_timeout
        mode = "abort" if timeout == 0 else "drain"

        logger.info(
            "[shutdown] API server: stopping engine client mode=%s timeout=%ss",
            mode,
            timeout,
        )

        await loop.run_in_executor(
            None, partial(engine_client.shutdown, timeout=timeout)
        )
        logger.info_once("[shutdown] API server: engine client stopped")

        server.should_exit = True
        logger.info_once("[shutdown] API server: signalling HTTP server shutdown")
        server_task.cancel()
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    shutdown_task = loop.create_task(handle_shutdown())

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
        logger.info_once("[shutdown] API server: shutting down FastAPI HTTP server")
        return server.shutdown()
    finally:
        shutdown_task.cancel()
        watchdog_task.cancel()


async def _serve_hypercorn(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **kwargs: Any,
) -> Any:
    """Serve using Hypercorn with HTTP/2 + HTTP/1.1 (ALPN negotiation).

    Requires: pip install hypercorn h2
    """
    try:
        from hypercorn.asyncio import serve as hypercorn_serve
        from hypercorn.config import Config as HypercornConfig
    except ImportError as exc:
        raise RuntimeError(
            "HTTP/2 support requires Hypercorn and h2. "
            "Install them with: pip install hypercorn h2"
        ) from exc

    hconfig = HypercornConfig()
    # Negotiate h2 first, fall back to HTTP/1.1 for clients that don't support h2
    hconfig.alpn_protocols = ["h2", "http/1.1"]

    host = kwargs.get("host") or "0.0.0.0"
    port = kwargs.get("port", 8000)
    if sock is not None:
        # Pass pre-bound socket via file-descriptor URI (Hypercorn's API)
        hconfig.bind = [f"fd://{sock.fileno()}"]
    else:
        hconfig.bind = [f"{host}:{port}"]

    # Map SSL kwargs
    if kwargs.get("ssl_certfile"):
        hconfig.certfile = kwargs["ssl_certfile"]
    if kwargs.get("ssl_keyfile"):
        hconfig.keyfile = kwargs["ssl_keyfile"]
    if kwargs.get("ssl_ca_certs"):
        hconfig.ca_certs = kwargs["ssl_ca_certs"]

    # Log level
    if kwargs.get("log_level"):
        hconfig.loglevel = kwargs["log_level"]

    # Keep-alive timeout
    if kwargs.get("timeout_keep_alive"):
        hconfig.keep_alive_timeout = kwargs["timeout_keep_alive"]

    loop = asyncio.get_running_loop()
    h2_shutdown_event = asyncio.Event()
    adapter = _HypercornAdapter(h2_shutdown_event)
    # Expose same interface as uvicorn.Server so terminate_if_errored works
    app.state.server = adapter

    ssl_cert_refresher: SSLCertRefresher | None = None
    if enable_ssl_refresh and kwargs.get("ssl_certfile"):
        import ssl as _ssl

        ssl_ctx = _ssl.create_default_context(_ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(
            certfile=kwargs["ssl_certfile"],
            keyfile=kwargs.get("ssl_keyfile"),
        )
        ssl_cert_refresher = SSLCertRefresher(
            ssl_context=ssl_ctx,
            key_path=kwargs.get("ssl_keyfile"),
            cert_path=kwargs["ssl_certfile"],
            ca_path=kwargs.get("ssl_ca_certs"),
        )

    sig_shutdown = asyncio.Event()

    def signal_handler() -> None:
        sig_shutdown.set()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    server_task = loop.create_task(
        hypercorn_serve(
            app,  # type: ignore[arg-type]
            hconfig,
            shutdown_trigger=h2_shutdown_event.wait,
        )
    )

    watchdog_task = loop.create_task(watchdog_loop(adapter, app.state.engine_client))

    async def handle_shutdown() -> None:
        await sig_shutdown.wait()
        engine_client = app.state.engine_client
        timeout = engine_client.vllm_config.shutdown_timeout
        await loop.run_in_executor(
            None, partial(engine_client.shutdown, timeout=timeout)
        )
        adapter.should_exit = True  # sets h2_shutdown_event
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    shutdown_task = loop.create_task(handle_shutdown())

    protocol = "https" if kwargs.get("ssl_certfile") else "http"
    logger.info("Hypercorn HTTP/2 server listening on %s://%s:%s", protocol, host, port)

    async def dummy_shutdown() -> None:
        pass

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        logger.info("Shutting down Hypercorn HTTP/2 server.")
        return adapter.shutdown()
    finally:
        shutdown_task.cancel()
        watchdog_task.cancel()


async def watchdog_loop(
    server: uvicorn.Server | _HypercornAdapter, engine: EngineClient
):
    """
    # Watchdog task that runs in the background, checking
    # for error state in the engine. Needed to trigger shutdown
    # if an exception arises is StreamingResponse() generator.
    """
    VLLM_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)


def terminate_if_errored(
    server: uvicorn.Server | _HypercornAdapter, engine: EngineClient
):
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
