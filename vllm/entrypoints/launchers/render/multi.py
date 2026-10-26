# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-process launcher for the vLLM render server."""

import argparse
import contextlib
import multiprocessing
import signal
from multiprocessing.process import BaseProcess

import uvloop

from vllm.logger import configure_logging_from_args, init_logger
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.utils import shutdown

from ..launcher import setup_server
from .entry import build_and_serve_renderer, build_vllm_config

logger = init_logger(__name__)


def run_worker_proc(
    args: argparse.Namespace,
    api_process_rank: int = 0,
) -> None:
    """Entrypoint for an individual API server worker process."""
    num_api_servers: int = args.api_server_count

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

    async def _serve() -> None:
        await build_and_serve_renderer(vllm_config, listen_address, sock, args)

    with contextlib.ExitStack() as exit_stack:
        listen_address, sock = setup_server(args, reuse_port=num_api_servers > 1)
        exit_stack.callback(sock.close)
        uvloop.run(_serve())


def _run_multi_launch(
    args: argparse.Namespace,
    exit_stack: contextlib.ExitStack,
) -> None:
    """Spawn ``args.api_server_count`` workers, each binding its own socket."""
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 0

    spawn_context = multiprocessing.get_context("spawn")
    processes: list[BaseProcess] = []
    exit_stack.callback(shutdown, processes)

    for i in range(num_api_servers):
        proc = spawn_context.Process(
            target=run_worker_proc,
            name=f"ApiServer_{i}",
            args=(args, i),
        )
        processes.append(proc)
        proc.start()

    # Block until all workers exit.
    for p in processes:
        p.join()
        if p.exitcode not in (0, None):
            logger.error(
                "API server worker %s exited unexpectedly with code %s",
                p.name,
                p.exitcode,
            )
            break


def run_multi_launch(args: argparse.Namespace) -> None:
    """Run the render server with multiple API workers."""
    with contextlib.ExitStack() as exit_stack:
        try:
            _run_multi_launch(args, exit_stack)
        except KeyboardInterrupt:
            logger.info_once("[shutdown] API server: interrupted by user.")
        except Exception:
            logger.exception("[shutdown] API server: unexpected error during shutdown")
            raise
