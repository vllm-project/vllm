# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-process launcher for the vLLM render server."""

import argparse
import contextlib
import multiprocessing
import signal
from multiprocessing.connection import wait as mp_wait
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Barrier
from socket import socket
from typing import Any

import uvloop

from vllm.logger import configure_logging_from_args, init_logger
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.utils import shutdown

from ..launcher import _NOT_ACCEPTING, setup_server
from .entry import build_and_serve_renderer, build_vllm_config

logger = init_logger(__name__)

# Safety net for the startup barrier.
STARTUP_BARRIER_TIMEOUT = 300.0


def run_worker_proc(
    args: argparse.Namespace,
    api_process_rank: int,
    startup_barrier: Barrier,
    listen_address: str,
    sock: socket,
    api_server_loads,
) -> None:
    """Entrypoint for an individual API server worker process."""
    configure_logging_from_args(args)

    # Interrupt initialization if SIGTERM arrives before uvicorn installs
    # its own signal handlers. Once uvicorn is running it replaces this.
    def _interrupt_init(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)

    # Tag this process with its API rank for logs and introspection.
    set_process_title("APIServer", str(api_process_rank))
    decorate_logs()

    # Build VllmConfig locally to avoid pickling it across ``spawn``.
    vllm_config = build_vllm_config(args)
    vllm_config.parallel_config._api_process_rank = api_process_rank
    vllm_config.parallel_config._api_process_count = args.api_server_count

    peer_loads = (
        (api_server_loads, api_process_rank) if api_server_loads is not None else None
    )

    async def _serve() -> None:
        shutdown_task = await build_and_serve_renderer(
            vllm_config,
            listen_address,
            sock,
            args,
            peer_loads=peer_loads,
        )
        await shutdown_task

    with contextlib.ExitStack() as exit_stack:
        exit_stack.callback(sock.close)

        # ---- Rendezvous ---------------------------------------------- #
        if startup_barrier is not None:
            try:
                startup_barrier.wait(timeout=STARTUP_BARRIER_TIMEOUT)
            except Exception:
                return

        # ---- All peers ready, serve ---------------------------------- #
        uvloop.run(_serve())


def _wait_for_first_exit(processes: list[BaseProcess]) -> BaseProcess:
    """Block until *any* worker exits; return the one that finished first."""
    sentinels: dict[Any, BaseProcess] = {p.sentinel: p for p in processes}
    ready = mp_wait(list(sentinels.keys()))
    first = sentinels[ready[0]]
    # join() populates exitcode; it returns immediately for a dead process.
    first.join()
    if first.exitcode not in (0, None):
        logger.error(
            "API server worker %s exited unexpectedly with code %s",
            first.name,
            first.exitcode,
        )
    else:
        logger.info(
            "API server worker %s exited with code %s",
            first.name,
            first.exitcode,
        )
    return first


def _run_multi_launch(
    args: argparse.Namespace,
    exit_stack: contextlib.ExitStack,
) -> None:
    """Spawn ``args.api_server_count`` workers on one shared listening socket."""
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 0

    spawn_context = multiprocessing.get_context("spawn")
    startup_barrier = (
        spawn_context.Barrier(num_api_servers) if num_api_servers > 1 else None
    )

    # Shared-fd mode: bind once in the launcher.
    # NOTE: we deliberately do *not* use SO_REUSEPORT here. The peer_loads
    # mechanism requires all workers to share the *same* listening socket,
    # so that accept() can be arbitrated across processes.
    listen_address, sock = setup_server(args, reuse_port=False)
    exit_stack.callback(sock.close)

    # Shared per-worker load array. Each slot starts at _NOT_ACCEPTING so
    # that a worker which has not started yet is never considered "least
    # loaded". Only allocated for the multi-worker case.
    api_server_loads = (
        spawn_context.RawArray("i", [_NOT_ACCEPTING] * num_api_servers)
        if num_api_servers > 1
        else None
    )

    processes: list[BaseProcess] = []
    exit_stack.callback(shutdown, processes)

    for i in range(num_api_servers):
        proc = spawn_context.Process(
            target=run_worker_proc,
            name=f"ApiServer_{i}",
            args=(
                args,
                i,
                startup_barrier,
                listen_address,
                sock,
                api_server_loads,
            ),
        )
        processes.append(proc)
        proc.start()

    # As soon as ANY worker exits (normal or crash), tear down the whole set.
    _wait_for_first_exit(processes)


def run_multi_launch(args: argparse.Namespace) -> None:
    """Run the render server with multiple API workers."""

    def _interrupt_launcher(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_launcher)

    with contextlib.ExitStack() as exit_stack:
        try:
            _run_multi_launch(args, exit_stack)
        except KeyboardInterrupt:
            logger.info_once("[shutdown] API server: interrupted by user.")
        except Exception:
            logger.exception("[shutdown] API server: unexpected error during shutdown")
            raise
