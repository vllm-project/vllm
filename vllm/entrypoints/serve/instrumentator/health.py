# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import http.server
import multiprocessing
import socket
from multiprocessing import Process

from fastapi import APIRouter, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _health_server_target(
    host: str,
    port: int,
    engine_dead_val: multiprocessing.Value,
) -> None:
    """
    Minimal HTTP server that runs in a dedicated subprocess.
    Responds to GET /health without sharing the main event loop.
    Returns 200 when the engine is alive, 503 when it is dead.
    """

    class HealthHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                status = 503 if bool(engine_dead_val.value) else 200
                self.send_response(status)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            pass  # suppress per-request access logs

    if ":" in host:
        # IPv6: subclass to override address_family before socket is created.
        class IPv6HTTPServer(http.server.HTTPServer):
            address_family = socket.AF_INET6

        server_cls = IPv6HTTPServer
    else:
        server_cls = http.server.HTTPServer

    with server_cls((host, port), HealthHandler) as server:
        server.serve_forever()


def start_health_process(
    host: str,
    port: int,
    engine_dead_val: multiprocessing.Value,
) -> Process:
    """Start the out-of-band health check subprocess."""
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_health_server_target,
        args=(host, port, engine_dead_val),
        daemon=True,
        name="HealthCheckServer",
    )
    proc.start()
    return proc


def stop_health_process(proc: Process) -> None:
    """Terminate the out-of-band health check subprocess."""
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)
