#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import multiprocessing
import os
import signal
import time
from functools import partial
from http import HTTPStatus
from multiprocessing.process import BaseProcess

import aiohttp
import uvicorn
import uvloop
from fastapi import FastAPI, Response

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import (
    decorate_logs,
    kill_process_tree,
    set_process_title,
    update_environment_variables,
)

logger = init_logger(__name__)

CHILD_EXIT_GRACE_S = 5.0


def infer_multi_port_external_lb_start_rank(args: argparse.Namespace) -> int:
    start_rank = getattr(args, "data_parallel_start_rank", None)
    if start_rank is not None:
        return start_rank

    node_rank = getattr(args, "node_rank", 0) or 0
    local_size = getattr(args, "data_parallel_size_local", 0) or 0
    return node_rank * local_size


def validate_multi_port_external_lb_args(args: argparse.Namespace) -> None:
    if getattr(args, "grpc", False):
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb does not support --grpc"
        )
    if args.uds is not None:
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb does not support --uds"
        )
    if any((args.ssl_keyfile, args.ssl_certfile, args.ssl_ca_certs)):
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb does not support HTTPS yet"
        )
    if args.api_server_count not in (None, 1):
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb currently requires "
            "--api-server-count=1"
        )
    if args.data_parallel_rank is not None:
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb manages child "
            "--data-parallel-rank values internally"
        )
    if args.data_parallel_external_lb or args.data_parallel_hybrid_lb:
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb cannot be combined with "
            "--data-parallel-external-lb or --data-parallel-hybrid-lb"
        )
    if args.data_parallel_size < 2:
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb requires "
            "--data-parallel-size > 1"
        )

    local_size = args.data_parallel_size_local
    if local_size is None or local_size < 2:
        raise ValueError(
            "Error: --data-parallel-multi-port-external-lb requires "
            "--data-parallel-size-local >= 2"
        )
    if local_size > args.data_parallel_size:
        raise ValueError(
            "Error: --data-parallel-size-local cannot exceed --data-parallel-size"
        )
    if args.data_parallel_size % local_size != 0:
        raise ValueError(
            "Error: --data-parallel-size must be divisible by "
            "--data-parallel-size-local"
        )

    start_rank = infer_multi_port_external_lb_start_rank(args)
    if start_rank + local_size > args.data_parallel_size:
        raise ValueError(
            "Error: multi-port supervised ranks would exceed --data-parallel-size"
        )

    supervisor_port = args.data_parallel_supervisor_port
    child_port_min = args.port
    child_port_max = args.port + local_size - 1
    if child_port_min <= supervisor_port <= child_port_max:
        raise ValueError(
            f"Error: --data-parallel-supervisor-port {supervisor_port} "
            f"overlaps with child rank ports {child_port_min}-{child_port_max}"
        )


def build_multi_port_external_lb_child_args(
    args: argparse.Namespace, local_rank: int
) -> argparse.Namespace:
    child_args = copy.copy(args)
    child_args.port = args.port + local_rank
    child_args.data_parallel_rank = (
        infer_multi_port_external_lb_start_rank(args) + local_rank
    )
    child_args.data_parallel_start_rank = None
    child_args.data_parallel_size_local = 1
    child_args.data_parallel_external_lb = True
    child_args.data_parallel_hybrid_lb = False
    child_args.data_parallel_multi_port_external_lb = False
    child_args.data_parallel_supervisor_port = None
    child_args.api_server_count = 1
    return child_args


def _build_multi_port_external_lb_child_env(
    args: argparse.Namespace, local_rank: int
) -> dict[str, str]:
    # set visible devices for the child process
    devices_per_rank = args.tensor_parallel_size * args.pipeline_parallel_size
    start = local_rank * devices_per_rank
    stop = start + devices_per_rank
    device_env = current_platform.device_control_env_var
    visible_devices = ",".join(
        str(current_platform.device_id_to_physical_device_id(idx))
        for idx in range(start, stop)
    )
    return {device_env: visible_devices}


def _child_base_url(args: argparse.Namespace, port: int) -> str:
    host = args.host or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    return f"http://{host}:{port}"


def _join_processes_with_timeout(processes: list[BaseProcess], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if process.is_alive():
            process.join(timeout=remaining)


async def _probe_endpoint(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    port: int,
    path: str,
) -> bool:
    try:
        async with session.get(_child_base_url(args, port) + path) as response:
            return response.status == HTTPStatus.OK
    except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, OSError):
        return False


def _build_dp_supervisor_app(
    supervisor: DPSupervisor,
) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.state.supervisor = supervisor

    def _status_response(ok: bool) -> Response:
        return Response(
            status_code=(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE)
        )

    @app.get("/health", include_in_schema=False)
    async def health() -> Response:
        return _status_response(app.state.supervisor.is_healthy())

    @app.get("/ready", include_in_schema=False)
    @app.get("/readyz", include_in_schema=False)
    async def ready() -> Response:
        # when child servers is healthy, it is ready already
        return _status_response(app.state.supervisor.is_healthy())

    return app


def _run_multi_port_external_lb_child(
    child_args: argparse.Namespace, env_updates: dict[str, str]
) -> None:
    from vllm.entrypoints.openai.api_server import run_server

    rank = child_args.data_parallel_rank
    os.setpgrp()
    update_environment_variables(env_updates)
    set_process_title("ExternalLBRank", str(rank))
    decorate_logs(f"ExternalLBRank{rank}")
    uvloop.run(run_server(child_args))


class DPSupervisor:
    def __init__(self, args: argparse.Namespace):
        validate_multi_port_external_lb_args(args)
        self.args = args
        self.supervisor_port = args.data_parallel_supervisor_port
        self.child_ports = [
            args.port + local_rank
            for local_rank in range(args.data_parallel_size_local)
        ]
        self.children_healthy = False
        self.processes: list[BaseProcess] = []
        self._shutdown_event = asyncio.Event()
        self._shutdown_signal = signal.SIGTERM

    def is_healthy(self) -> bool:
        return not self._shutdown_event.is_set() and self.children_healthy

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        def on_server_exit(_task: asyncio.Task[None]) -> None:
            self._shutdown_event.set()

        host = self.args.host or "0.0.0.0"
        app = _build_dp_supervisor_app(self)
        config = uvicorn.Config(
            app,
            host=host,
            port=self.supervisor_port,
            log_level=self.args.uvicorn_log_level,
            access_log=False,
        )
        supervisor_server = uvicorn.Server(config)
        supervisor_server_task = asyncio.create_task(
            supervisor_server.serve(),
            name="multi-port-external-lb-supervisor",
        )
        supervisor_server_task.add_done_callback(on_server_exit)
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, partial(self._handle_signal, sig))

        try:
            while (
                not supervisor_server.started
                and not supervisor_server_task.done()
                and not self._shutdown_event.is_set()
            ):
                await asyncio.sleep(0)
            if supervisor_server_task.done():
                await supervisor_server_task
                return
            if self._shutdown_event.is_set():
                return
            logger.info(
                "Started multi-port external LB supervisor on %s:%d",
                host,
                self.supervisor_port,
            )
            self._start_children()
            await self._monitor_children()
        finally:
            self._shutdown_event.set()
            await self._shutdown_children()
            supervisor_server.should_exit = True
            await supervisor_server_task

    def _handle_signal(self, signum: int) -> None:
        if self._shutdown_event.is_set():
            return
        self._shutdown_signal = signal.Signals(signum)
        logger.info(
            "Received signal %d, forwarding graceful termination to "
            "multi-port external LB child ranks",
            signum,
        )
        self._shutdown_event.set()

    def _start_children(self) -> None:
        context = multiprocessing.get_context("spawn")
        for local_rank in range(self.args.data_parallel_size_local):
            child_args = build_multi_port_external_lb_child_args(self.args, local_rank)
            child_env = _build_multi_port_external_lb_child_env(self.args, local_rank)
            process = context.Process(
                target=_run_multi_port_external_lb_child,
                name=f"ExternalLBRank_{child_args.data_parallel_rank}",
                args=(child_args, child_env),
            )
            process.start()
            self.processes.append(process)

    async def _monitor_children(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.args.data_parallel_probe_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self._shutdown_event.is_set():
                child_health = await asyncio.gather(
                    *(
                        _probe_endpoint(session, self.args, port, "/health")
                        for port in self.child_ports
                    )
                )
                self.children_healthy = all(child_health)
                failed_process = next(
                    (
                        process
                        for process in self.processes
                        if process.exitcode is not None
                    ),
                    None,
                )
                if failed_process is not None:
                    raise RuntimeError(
                        f"Multi-port external LB child exited unexpectedly: "
                        f"{failed_process.name} "
                        f"exit code {failed_process.exitcode}"
                    )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.args.data_parallel_probe_interval_s,
                    )

    async def _shutdown_children(self) -> None:
        if self.processes:
            logger.info(
                "Forwarding signal %d to %d multi-port external LB child processes",
                self._shutdown_signal,
                len(self.processes),
            )
            timeout = self.args.shutdown_timeout + CHILD_EXIT_GRACE_S
            for process in self.processes:
                if not process.is_alive() or (pid := process.pid) is None:
                    continue
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pid, self._shutdown_signal)

            await asyncio.to_thread(
                _join_processes_with_timeout,
                self.processes,
                timeout,
            )

            for process in self.processes:
                if process.is_alive() and (pid := process.pid) is not None:
                    kill_process_tree(pid)


def run_dp_supervisor(args: argparse.Namespace) -> None:
    uvloop.run(DPSupervisor(args).run())
