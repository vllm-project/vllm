# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import signal
import socket
from collections.abc import Generator, MutableSequence
from functools import partial
from typing import Any

import uvicorn
from fastapi import FastAPI

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launchers.utils.ssl import SSLCertRefresher
from vllm.entrypoints.serve.utils.api_utils import (
    log_non_default_args,
    log_version_and_model,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.tracing import instrument
from vllm.utils.network_utils import find_process_using_port, is_valid_ipv6_address
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION

from .utils.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)

logger = init_logger(__name__)

# A worker with more connections than another leaves a pending connection to
# the less loaded one, rechecking every 1 ms, for up to 5 ms (it may be busy).
_ACCEPT_DEFER_CHECKS = 5
# The load of a worker that is not accepting (also the initial value of the
# shared array), so it is never the least loaded.
_NOT_ACCEPTING = 2**31 - 1


class NoSignalServer(uvicorn.Server):
    """Uvicorn server that never installs its own SIGINT/SIGTERM handlers.

    Callers register their own handlers on the event loop for graceful
    shutdown; uvicorn's would race with and override them (see #49668).

    With ``peer_loads`` (several API workers on one shared socket: a shared
    array with one slot per worker, and this worker's index), each worker
    publishes its number of open connections, and a pending connection is
    accepted by the least loaded worker. The event loop's own server would
    accept every queued connection in whichever worker wakes first.
    """

    def __init__(
        self,
        config: uvicorn.Config,
        peer_loads: tuple[MutableSequence[int], int] | None = None,
    ):
        super().__init__(config)
        self.peer_loads = peer_loads
        self._accept_tasks: set[asyncio.Task] = set()
        self._connecting = 0

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield

    def _publish_load(self) -> None:
        assert self.peer_loads is not None
        loads, index = self.peer_loads
        loads[index] = len(self.server_state.connections) + self._connecting

    def _least_loaded(self) -> bool:
        assert self.peer_loads is not None
        loads, index = self.peer_loads
        return loads[index] <= min(loads)

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        if not (self.peer_loads and sockets):
            return await super().startup(sockets)
        # With no sockets uvicorn starts no asyncio server; _accept serves
        # them with uvicorn's protocol instead.
        await super().startup(sockets=[])
        if not self.started:  # lifespan startup failed (uvicorn < 0.50)
            return
        self._publish_load()
        for sock in sockets:
            sock.listen(self.config.backlog)
            sock.setblocking(False)
            self._track(self._accept(sock))

    def _track(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._accept_tasks.add(task)
        task.add_done_callback(self._accept_tasks.discard)

    async def _accept(self, sock: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await _readable(loop, sock)
            # Connections closed since the last accept; peers refresh likewise.
            self._publish_load()
            for _ in range(_ACCEPT_DEFER_CHECKS):
                if self._least_loaded():
                    break
                await asyncio.sleep(0.001)
            try:
                conn, _ = sock.accept()
            except (BlockingIOError, InterruptedError, ConnectionAbortedError):
                continue  # another worker took it
            except OSError:
                # e.g. EMFILE; asyncio's own server also retries after 1 s.
                logger.exception("Error accepting a connection")
                await asyncio.sleep(1)
                continue
            self._connecting += 1
            self._publish_load()
            # Its own task, so a TLS handshake does not hold up the next accept.
            self._track(self._connect(loop, conn))

    async def _connect(self, loop: asyncio.AbstractEventLoop, conn: socket.socket):
        config = self.config
        try:
            await loop.connect_accepted_socket(
                lambda: config.http_protocol_class(  # type: ignore[call-arg]
                    config=config,
                    server_state=self.server_state,
                    app_state=self.lifespan.state,
                    _loop=loop,
                ),
                conn,
                ssl=config.ssl,
            )
        except asyncio.CancelledError:
            conn.close()
            raise
        except Exception:
            # e.g. a failed TLS handshake
            logger.debug("Accepted connection failed to start", exc_info=True)
            conn.close()
        finally:
            self._connecting -= 1
            self._publish_load()

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # Before super() closes the listening sockets.
        tasks = list(self._accept_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.peer_loads is not None:
            loads, index = self.peer_loads
            loads[index] = _NOT_ACCEPTING
        await super().shutdown(sockets)


async def _readable(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> None:
    ready: asyncio.Future[None] = loop.create_future()

    def wake() -> None:
        if not ready.done():
            ready.set_result(None)

    loop.add_reader(sock, wake)
    try:
        await ready
    finally:
        loop.remove_reader(sock)


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    peer_loads: tuple[MutableSequence[int], int] | None = None,
    **uvicorn_kwargs: Any,
):
    """Start a FastAPI app using Uvicorn, with support for custom Uvicorn config
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

    config = uvicorn.Config(app, **uvicorn_kwargs)
    # Set header limits
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    config.load()
    server = NoSignalServer(config, peer_loads=peer_loads)
    app.state.server = server

    loop = asyncio.get_running_loop()

    engine_client = app.state.engine_client
    watchdog_task = (
        loop.create_task(watchdog_loop(server, engine_client))
        if engine_client is not None
        else None
    )
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

        if engine_client is not None:
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
        if watchdog_task is not None:
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
        if watchdog_task is not None:
            watchdog_task.cancel()


async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    """# Watchdog task that runs in the background, checking
    # for error state in the engine. Needed to trigger shutdown
    # if an exception arises is StreamingResponse() generator.
    """
    VLLM_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)


def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
    """See discussions here on shutting down a uvicorn server
    https://github.com/encode/uvicorn/discussions/1103
    In this case we cannot await the server shutdown here
    because handler must first return to close the connection
    for this request.
    """
    engine_errored = engine.errored and not engine.is_running
    if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
        server.should_exit = True


def create_server_socket(
    addr: tuple[str, int],
    *,
    reuse_port: bool,
) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if reuse_port:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


def create_server_unix_socket(path: str) -> socket.socket:
    sock = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
    sock.bind(path)
    return sock


def validate_api_server_args(args):
    valid_tool_parses = ToolParserManager.list_registered()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parsers = ReasoningParserManager.list_registered()
    if (
        reasoning_parser := args.structured_outputs_config.reasoning_parser
    ) and reasoning_parser not in valid_reasoning_parsers:
        raise KeyError(
            f"invalid reasoning parser: {reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parsers)} }})"
        )


@instrument(span_name="API server setup")
def setup_server(args, *, reuse_port: bool):
    """Validate API server args and create the server socket."""
    log_version_and_model(logger, VLLM_VERSION, args.model)
    log_non_default_args(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr, reuse_port=reuse_port)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    if args.uds:
        listen_address = f"unix:{args.uds}"
    else:
        addr, port = sock_addr
        is_ssl = args.ssl_keyfile and args.ssl_certfile
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock
