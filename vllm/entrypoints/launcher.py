import asyncio
import signal
from http import HTTPStatus
from typing import Any

import uvicorn
from fastapi import FastAPI, Response

from vllm import envs
from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.engine.protocol import AsyncEngineClient
from vllm.logger import init_logger
from vllm.utils import find_process_using_port

logger = init_logger(__name__)


async def serve_http(app: FastAPI, engine: AsyncEngineClient,
                     **uvicorn_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    # Set concurrency limits in uvicorn if running in multiprocessing mode
    # since zmq has maximum socket limit of zmq.constants.SOCKET_LIMIT (65536).
    if engine.limit_concurrency is not None:
        logger.info(
            "Launching Uvicorn with --limit_concurrency %s. To avoid this "
            "limit at the expense of performance run with "
            "--disable-frontend-multiprocessing", engine.limit_concurrency)
        uvicorn_kwargs["limit_concurrency"] = engine.limit_concurrency

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server, engine)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

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
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logger.info("Gracefully stopping http server")
        return server.shutdown()


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server,
                           engine: AsyncEngineClient) -> None:
    """Adds handlers for fatal errors that should crash the server"""

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_, __):
        """On generic runtime error, check to see if the engine has died.
        It probably has, in which case the server will no longer be able to
        handle requests. Trigger a graceful shutdown with a SIGTERM."""
        if (not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine.errored
                and not engine.is_running):
            logger.fatal("AsyncLLMEngine has failed, terminating server "
                         "process")
            # See discussions here on shutting down a uvicorn server
            # https://github.com/encode/uvicorn/discussions/1103
            # In this case we cannot await the server shutdown here because
            # this handler must first return to close the connection for
            # this request.
            server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @app.exception_handler(AsyncEngineDeadError)
    async def engine_dead_handler(_, __):
        """Kill the server if the async engine is already dead. It will
        not handle any further requests."""
        if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH:
            logger.fatal("AsyncLLMEngine is already dead, terminating server "
                         "process")
            server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
