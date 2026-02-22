# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight frontend server for multi-API-server pod deployments.

When running N vLLM API servers (each on its own port) inside a single
Kubernetes pod, this module provides a thin FastAPI frontend that:

  1. Aggregates pod-level /health checks: returns 200 only when ALL N
     backend servers are healthy.  K8s liveness and startup probes should
     point at this frontend's port so that the pod is only considered live
     once every backend is ready.

  2. Monitors backend processes and triggers a clean shutdown of the
     frontend (and therefore the pod) if any backend crashes.  This ensures
     K8s will restart the pod rather than leaving it in a partially-working
     state.

Usage (internal to vllm/entrypoints/cli/serve.py):

    uvloop.run(
        run_frontend(
            host="0.0.0.0",
            port=8000,              # pod-level port that K8s knows about
            sock=frontend_sock,     # pre-bound socket
            backend_urls=[          # one per backend server
                "http://127.0.0.1:8001",
                "http://127.0.0.1:8002",
            ],
            processes=backend_procs,  # multiprocessing.Process objects
        )
    )
"""

import asyncio
import contextlib
import signal
import sys
from multiprocessing.process import BaseProcess
from socket import socket as Socket

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

from vllm.logger import init_logger

logger = init_logger(__name__)

# Timeout for each individual backend /health HTTP request.
_HEALTH_TIMEOUT_S: float = 5.0

# How often to poll backend processes for unexpected exits.
_PROCESS_POLL_INTERVAL_S: float = 2.0


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def build_frontend_app(backend_urls: list[str]) -> FastAPI:
    """Return a FastAPI app that aggregates /health across all backends.

    Args:
        backend_urls: Base URLs of each backend vLLM server,
            e.g. ["http://127.0.0.1:8001", "http://127.0.0.1:8002"].
    """
    app = FastAPI(
        title="vLLM multi-server frontend",
        description=(
            "Thin health-aggregation proxy for multi-API-server pod deployments."
        ),
    )

    @app.get("/health", response_class=Response)
    async def health() -> Response:
        """Pod-level liveness / startup check.

        Returns 200 only when every backend server reports 200 on its own
        /health endpoint.  Use this as the K8s livenessProbe and
        startupProbe target on the frontend port.
        """
        async with httpx.AsyncClient() as client:
            for url in backend_urls:
                try:
                    resp = await client.get(
                        f"{url}/health",
                        timeout=_HEALTH_TIMEOUT_S,
                    )
                    if resp.status_code != 200:
                        logger.debug(
                            "Backend %s /health returned %d",
                            url,
                            resp.status_code,
                        )
                        return Response(status_code=503)
                except Exception as exc:
                    logger.debug("Backend %s unreachable: %s", url, exc)
                    return Response(status_code=503)

        return Response(status_code=200)

    return app


# ---------------------------------------------------------------------------
# Process watcher
# ---------------------------------------------------------------------------


async def _process_watcher(
    processes: list[BaseProcess],
    server: uvicorn.Server,
) -> None:
    """Background coroutine that monitors backend process health.

    If any backend process exits (regardless of exit code), the frontend
    uvicorn server is told to stop.  The caller is responsible for
    propagating that into a non-zero pod exit so that K8s will restart it.

    Args:
        processes: Backend API server processes to watch.
        server: The running uvicorn server instance; setting
            ``server.should_exit = True`` triggers a graceful stop.
    """
    while True:
        await asyncio.sleep(_PROCESS_POLL_INTERVAL_S)
        for proc in processes:
            if not proc.is_alive():
                exitcode = proc.exitcode or 0
                if exitcode != 0:
                    logger.error(
                        "Backend process %s (PID %d) exited with code %d. "
                        "Initiating pod shutdown.",
                        proc.name,
                        proc.pid,
                        exitcode,
                    )
                else:
                    logger.warning(
                        "Backend process %s (PID %d) exited cleanly (code 0). "
                        "Initiating pod shutdown.",
                        proc.name,
                        proc.pid,
                    )
                server.should_exit = True
                return


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_frontend(
    host: str,
    port: int,
    sock: Socket,
    backend_urls: list[str],
    processes: list[BaseProcess],
    log_level: str = "info",
) -> int:
    """Run the multi-server frontend until shutdown.

    The frontend exits (and returns a non-zero status) if any backend
    process crashes so that the caller can propagate the failure to the
    operating system (allowing K8s to restart the pod).

    Args:
        host: Host to bind the frontend server (used for uvicorn config
            only; the actual listening socket is ``sock``).
        port: Port number (used for uvicorn config logging only).
        sock: Pre-bound socket for the frontend to accept on.
        backend_urls: HTTP base URLs of the N backend vLLM servers.
        processes: Multiprocessing Process objects for each backend.
        log_level: Uvicorn log level string.

    Returns:
        Exit code: 0 for a clean shutdown, 1 if a backend crashed.
    """
    logger.info(
        "Starting multi-server frontend on %s:%d (aggregating %d backends: %s)",
        host,
        port,
        len(backend_urls),
        ", ".join(backend_urls),
    )

    app = build_frontend_app(backend_urls)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )
    server = uvicorn.Server(config)

    # Track whether shutdown was triggered by a backend crash (vs clean
    # SIGINT / SIGTERM from the outside).
    backend_crashed = False
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        server.should_exit = True

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    async def _watched_process_watcher() -> None:
        nonlocal backend_crashed
        await _process_watcher(processes, server)
        # If we get here a backend has already set server.should_exit.
        # Record it so we can return a non-zero exit code.
        for proc in processes:
            if not proc.is_alive() and (proc.exitcode or 0) != 0:
                backend_crashed = True
                break
        # If a backend exited cleanly but unexpectedly still treat it
        # as a crash so K8s knows something went wrong.
        backend_crashed = True

    watcher_task = loop.create_task(_watched_process_watcher())
    server_task = loop.create_task(server.serve(sockets=[sock]))

    try:
        await asyncio.gather(server_task, watcher_task)
    except asyncio.CancelledError:
        pass
    finally:
        watcher_task.cancel()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher_task
        with contextlib.suppress(asyncio.CancelledError):
            await server_task

    return 1 if backend_crashed else 0


def main_frontend(
    host: str,
    port: int,
    sock: Socket,
    backend_urls: list[str],
    processes: list[BaseProcess],
    log_level: str = "info",
) -> None:
    """Synchronous wrapper around :func:`run_frontend` for use in the main
    process after spawning backend workers.

    Calls ``sys.exit(1)`` if a backend crashed so the OS / K8s sees a
    non-zero exit code and restarts the pod.
    """
    import uvloop

    exit_code = uvloop.run(
        run_frontend(
            host=host,
            port=port,
            sock=sock,
            backend_urls=backend_urls,
            processes=processes,
            log_level=log_level,
        )
    )
    sys.exit(exit_code)
