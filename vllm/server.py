import asyncio
import signal
from typing import Any

import uvicorn
from fastapi import FastAPI

from vllm.logger import init_logger

logger = init_logger(__name__)


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any) -> None:
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        methods = ', '.join(methods)
        logger.info("Route: %s, Methods: %s", path, methods)

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Gracefully stopping http server")
        await server.shutdown()
