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

HEALTHCHECK_INTERVAL_S = 5.0
HEALTHCHECK_TIMEOUT_S = 5.0
DEFAULT_CHILD_GRACEFUL_TERMINATION = 30.


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



async def _probe_endpoint(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    port: int,
    path: str,
) -> tuple[bool, str | None]:
    try:
        async with session.get(_child_base_url(args, port) + path) as response:
            if response.status == HTTPStatus.OK:
                return True, None
            return False, f"HTTP {response.status}"
    except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, OSError) as exc:
        return False, str(exc)


def _build_supervisor_app() -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.state.is_vllm_ready = False

    def _status_response(ok: bool) -> Response:
        return Response(
            status_code=(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE)
        )

    @app.get("/health", include_in_schema=False)
    async def health() -> Response:
        return _status_response(app.state.is_vllm_ready)

    @app.get("/ready", include_in_schema=False)
    @app.get("/readyz", include_in_schema=False)
    async def ready() -> Response:
        return _status_response(app.state.is_vllm_ready)

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
        host = self.args.host or "0.0.0.0"
        self.app = _build_supervisor_app()

        self.supervisor_port = args.data_parallel_supervisor_port
        self.child_ports = [
            args.port + local_rank
            for local_rank in range(args.data_parallel_size_local)
        ]


        self.processes: list[BaseProcess] = []
        self._failed_process: BaseProcess | None = None
        self._shutdown_event = asyncio.Event()
        self._shutdown_signal = signal.SIGTERM

    async def run(self) -> None:
        """
        This is the main coroutine running on pid 1 in K8s.

        K8s pod termination lifecycle will send a SIGTERM to pid 1
        during shutdown, with a terminationGracePeriodSeconds to
        enable things like request draining. This eventloop is
        responsible for handling this signal and coordinating
        with the background workers.

        Additionally, the K8s API server will send health and
        liveness probes to a single endpoint in the pod. This
        coroutine is responsible for monitoring the background
        vLLM servers and responding to those probes.
        """

        def _handle_signal(signum: int) -> None:
            signal = signal.Signals(signum)
            logger.info("Received %s, beginning shutdown", signal.name)
            self._shutdown_event.set()
            self._shutdown_signal = signal

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _handle_signal, sig)

        server, server_task = await self._start_supervisor_server()

        try:
            self._start_children()
            await self._monitor_children()
        finally:
            await self._shutdown_children()

            # Shutdown the supervisor server.
            server.should_exit = True
            await server_task

    async def _start_supervisor_server(self) -> None:
        """
        Start the supervisor server, ensuring its on the event loop.
        """

        # Create and launch the supervisor server.
        server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host=host,
                port=self.supervisor_port,
                log_level=self.args.uvicorn_log_level,
            )
        )
        server_task = asyncio.create_task(
            server.serve(),
            name="dp-supervisor-server",
        )

        def on_server_exit(task):
            try:
                # Raises the exception if the server crashed.
                task.result()
                logger.info("DP Supervisor server finished gracefully.")
            except Exception as e:
                logger.info(f"DP Supervisor server crashed with error: {e}")
            # Set shutdown even so background coroutines clean up.
            self._shutdown_event.set()
        server_task.add_done_callback(on_server_exit)

        # Ensure the server task is running on the event loop.
        while not server.started:
            if server_task.done():
                exception = server_task.exception()
                raise RuntimeError(f"Server failed to start: {exception}")
            await asyncio.sleep(0)
        
        return server, server_task

    def _start_children(self) -> None:
        context = multiprocessing.get_context("spawn")
        for local_rank in range(self.args.data_parallel_size_local):
            child_args = build_multi_port_external_lb_child_args(self.args, local_rank)
            child_env = _build_multi_port_external_lb_child_env(self.args, local_rank)
            process = context.Process(
                target=_run_multi_port_external_lb_child,
                name=f"VLLM_DP_{child_args.data_parallel_rank}",
                args=(child_args, child_env),
            )
            process.start()
            self.processes.append(process)

    async def _collect_child_health(
        self, session: aiohttp.ClientSession, port: int, process: BaseProcess
    ) -> bool:
        # TODO: we need to implement some sort of retry system here.
        # See how K8s health checking works for inspiration.
        healthy, _ = await _probe_endpoint(session, self.args, port, "/health")
        return healthy

    async def _monitor_children(self) -> None:
        """
        Background asyncio task that monitors the vLLM servers.

        It works by:
        - A) sleeping for HEALTHCHECK_INTERVAL_S or until shutdown event
        - B) checking the pids to see if they are alive
        - C) probing the /health ports of the vLLM servers

        Before the vLLM servers are /ready:
        - if the pid is dead, we will shut down
        - if the probe fails, we try again after HEALTHCHECK_INTERVAL_S

        After the vLLM servers are /ready:
        - if the pid is dead, we will shut down
        - if the probe fails, we will shut down
        """

        async def block_until_probe() -> bool:
        """Block until shutdown event is set or probe interval."""
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    HEALTHCHECK_INTERVAL_S
                )
                return False
            except TimeoutError:
                return True

        def begin_shutdown(processes: list[process.Process]):
            logger.info(
                "DP supervisor found failed vLLM DP Servers: %s", p
            )
            self.app.state.is_vllm_ready = False
            self._shutdown_event.set()

        timeout = aiohttp.ClientTimeout(total=HEALTHCHECK_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while await block_until_probe():
                # 1) check if vLLM processes are still running.
                failed = [p for p in self.processes is not p.is_alive()]
                if len(failed) > 0:
                    begin_shutdown(failed)
                    return

                # 2) Gather readiness from the vLLM process API servers.
                ready_status = await asyncio.gather(
                    *(self._collect_child_health(session, port, process)
                    for port, process in zip(self.child_ports, self.processes)),
                    return_exceptions=True,
                )
                unready_processes = [
                    p for ready, p in zip(ready_status, process) if (
                        not ready or isinstance(ready, BaseException)
                    )
                ]

                logger.info(
                    "DP Supervisor found %s/%s ready vLLM DP Servers.",
                    len(processes) - len(unready_processes), len(processes)
                )

                # If all are ready, we are good to go.
                if len(unready_processes) == 0:
                    self.app.state.is_vllm_ready = True
                else:
                    # If is_ready but a probe failed, we begin the shutdown.
                    # Otherwise, we are still waiting for the first startup.
                    if self.app.state.is_vllm_ready:
                        begin_shutdown(unready_processes)
                        return

    async def _shutdown_children(self) -> None:
        """
        Shutdown the vLLM API server processes.
        """
        # 1. Send SIGTERM or SIGINT to all children
        for process in self.processes:
            logger.info(
                "Forwarding %s to vLLM server pid: %s",
                self._shutdown_signal.name, process.pid
            )
            if process.is_alive():
                os.kill(process, self._shutdown_signal)
        
        # 2. Wait briefly for them to exit gracefully
        for process in self.processes:
            await asyncio.to_thread(
                process.join(timeout=DEFAULT_CHILD_GRACEFUL_TERMINATION)
            )
        
        # 3. Force kill the process tree if it exceeds the timeout.
        for process in self.processes:
            if process.is_alive():
                logger.info(
                    "vLLM Server Process still alive after ", signal, process.pid
                )
                kill_process_tree(process.pid)


def run_multi_port_external_lb_supervisor(args: argparse.Namespace) -> None:
    validate_multi_port_external_lb_args(args)
    supervisor = DPSupervisor(args)
    uvloop.run(supervisor.run())
