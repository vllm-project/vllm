# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
import socket
from http import HTTPStatus
from typing import Any, Optional

import zmq
import zmq.asyncio

import uvicorn
from fastapi import FastAPI, Request, Response

from vllm.entrypoints.openai.connect_worker import worker_routine
from vllm import envs
from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.engine.multiprocessing import MQEngineDeadError
from vllm.entrypoints.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils import find_process_using_port

logger = init_logger(__name__)


async def serve_http(app: FastAPI,
                     sock: Optional[socket.socket],
                     enable_ssl_refresh: bool = False,
                     **uvicorn_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.load()
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(
        server.serve(sockets=[sock] if sock else None))

    ssl_cert_refresher = None if not enable_ssl_refresh else SSLCertRefresher(
        ssl_context=config.ssl,
        key_path=config.ssl_keyfile,
        cert_path=config.ssl_certfile,
        ca_path=config.ssl_ca_certs)

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
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
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()    

async def serve_zmq(arg, zmq_server_port: int, app: FastAPI) -> None:
    """Server routine"""
    logger.info(f"zmq Server start arg: {arg}, zmq_port: {zmq_server_port}")
    url_worker = "inproc://workers"
    url_client = f"tcp://0.0.0.0:{zmq_server_port}"
    # Prepare our context and sockets
    context = zmq.asyncio.Context()

    # Socket to talk to clients
    clients = context.socket(zmq.ROUTER)
    clients.bind(url_client)
    logger.info(f"ZMQ Server ROUTER started at {url_client}")
    # Socket to talk to workers
    workers = context.socket(zmq.DEALER)
    workers.bind(url_worker)
    logger.info(f"ZMQ Worker DEALER started at {url_worker}")

    tasks = [asyncio.create_task(worker_routine(url_worker, app, context, i)) for i in range(5)]
    proxy_task =  asyncio.to_thread(zmq.proxy, clients, workers)
    
    try:
        await asyncio.gather(*tasks, proxy_task)
    except KeyboardInterrupt:
        print("ZMQ Server interrupted")
    except zmq.ZMQError as e:
        print("ZMQError:", e)
    finally:
        # We never get here but clean up anyhow
        clients.close()
        workers.close()
        context.destroy(linger=0)

def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """Adds handlers for fatal errors that should crash the server"""

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, __):
        """On generic runtime error, check to see if the engine has died.
        It probably has, in which case the server will no longer be able to
        handle requests. Trigger a graceful shutdown with a SIGTERM."""
        engine = request.app.state.engine_client
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
    async def async_engine_dead_handler(_, __):
        """Kill the server if the async engine is already dead. It will
        not handle any further requests."""
        if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH:
            logger.fatal("AsyncLLMEngine is already dead, terminating server "
                         "process")
            server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @app.exception_handler(MQEngineDeadError)
    async def mq_engine_dead_handler(_, __):
        """Kill the server if the mq engine is already dead. It will
        not handle any further requests."""
        if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH:
            logger.fatal("MQLLMEngine is already dead, terminating server "
                         "process")
            server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
