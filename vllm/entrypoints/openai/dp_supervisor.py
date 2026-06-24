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
import psutil
import uvicorn
import uvloop
from fastapi import FastAPI, Response

from vllm.logger import init_logger
from vllm.utils.system_utils import (
    decorate_logs,
    kill_process_tree,
    set_process_title,
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
    if bool(args.ssl_keyfile) != bool(args.ssl_certfile):
        raise ValueError(
            "Error: --ssl-keyfile and --ssl-certfile must be provided together"
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


def _build_vllm_dp_server_args(
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
    child_args.snapshot_provider = None
    child_args.device_ids = _build_device_ids(args, local_rank)
    return child_args


def _build_device_ids(args: argparse.Namespace, local_rank: int) -> list[int | str]:
    """Build the --device-ids value for a DP child process.

    The child resolves these against its own inherited device-control env
    var (e.g. CUDA_VISIBLE_DEVICES), so integer IDs must stay env-relative
    here rather than being translated to physical IDs.
    """
    devices_per_rank = args.tensor_parallel_size * args.pipeline_parallel_size
    start = local_rank * devices_per_rank
    stop = start + devices_per_rank
    device_ids = getattr(args, "device_ids", None)
    if device_ids is not None:
        if stop > len(device_ids):
            raise ValueError(
                f"--device-ids has {len(device_ids)} entries, but DP rank "
                f"{local_rank} needs devices [{start}, {stop})"
            )
        return device_ids[start:stop]
    return list(range(start, stop))


def _child_base_url(args: argparse.Namespace, port: int) -> str:
    host = args.host or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    scheme = "https" if args.ssl_keyfile and args.ssl_certfile else "http"
    return f"{scheme}://{host}:{port}"


def _join_processes_with_timeout(processes: list[BaseProcess], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        process.join(timeout=remaining)


async def _probe_endpoint(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    port: int,
    path: str,
    conn_err_failure_threshold: int = 3,
    conn_err_retry_delay: float = 5.0,
) -> bool:
    """
    Probe /health endpoint for 200 status.

    If there is a connection error, retry every N seconds.
    """
    for iteration in range(conn_err_failure_threshold):
        try:
            probe_ssl = None
            if args.ssl_keyfile and args.ssl_certfile:
                # Probes target node-local child servers over loopback, so skip
                # certificate verification to avoid SAN/hostname mismatches for
                # localhost/127.0.0.1 deployments.
                probe_ssl = False
            async with session.get(
                _child_base_url(args, port) + path, ssl=probe_ssl
            ) as response:
                # vLLM returns 503 on EngineDeadError, so we should return
                # immediately if vLLM responds with a non-200 status code.
                return response.status == HTTPStatus.OK
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Allow retry of connection errors.
            logger.debug(
                "Probe attempt %d/%d failed on port %d: %r",
                iteration + 1,
                conn_err_failure_threshold,
                port,
                e,
            )

        if iteration < conn_err_failure_threshold - 1:
            await asyncio.sleep(conn_err_retry_delay)

    return False


def _build_dp_supervisor_app(supervisor: DPSupervisor) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.state.supervisor = supervisor

    def _status_response(ok: bool) -> Response:
        return Response(
            status_code=(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE)
        )

    @app.get("/health", include_in_schema=False)
    async def health() -> Response:
        return _status_response(app.state.supervisor.is_ready)

    @app.get("/ready", include_in_schema=False)
    @app.get("/readyz", include_in_schema=False)
    async def ready() -> Response:
        return _status_response(app.state.supervisor.is_ready)

    return app


def _run_vllm_dp_server(child_args: argparse.Namespace) -> None:
    """
    Entrypoint function for the vLLM DP Server.
    """
    from vllm.entrypoints.openai.api_server import run_server

    # Create a fresh process group for the vLLM DP Server,
    # so that CTRL-C is propagated cleanly.
    os.setpgrp()

    name = f"APIServer_DP{child_args.data_parallel_rank}"
    set_process_title(name)
    decorate_logs(name)
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
        self._is_ready = False
        self._processes: list[BaseProcess] = []
        self._shutdown_event = asyncio.Event()
        self._shutdown_signal = signal.SIGTERM

    @property
    def is_ready(self) -> bool:
        return self._is_ready and not self._shutdown_event.is_set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        # K8s sends SIGTERM for shutdown - begin graceful termination.
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, partial(self._handle_signal, sig))

        # Launch DPSupervisor Server.
        app = _build_dp_supervisor_app(self)
        decorate_logs("DPSupervisor")
        host = self.args.host or "0.0.0.0"
        config = uvicorn.Config(
            app,
            host=host,
            port=self.supervisor_port,
            log_level=self.args.uvicorn_log_level,
            ssl_keyfile=self.args.ssl_keyfile,
            ssl_certfile=self.args.ssl_certfile,
            ssl_ca_certs=self.args.ssl_ca_certs,
            ssl_cert_reqs=self.args.ssl_cert_reqs,
            ssl_ciphers=self.args.ssl_ciphers,
        )
        supervisor_server = uvicorn.Server(config)
        supervisor_server_task = asyncio.create_task(
            supervisor_server.serve(),
            name="dp-supervisor",
        )
        supervisor_server_task.add_done_callback(
            lambda _task: self._shutdown_event.set()
        )

        # Ensure DPSupervisor task starts on the event loop.
        while not supervisor_server.started:
            if supervisor_server_task.done():
                supervisor_server_task.result()
                raise RuntimeError("DPSupervisor exited before startup.")
            await asyncio.sleep(0.05)
        logger.info("Started DPSupervisor on %s:%d", host, self.supervisor_port)

        # Launch and Monitor vLLM Server Processes.
        try:
            self._start_children()
            await self._monitor_children()
        finally:
            self._is_ready = False
            await self._shutdown_children()

            # Shutdown the DP Supervisor server.
            supervisor_server.should_exit = True
            await supervisor_server_task

    def _handle_signal(self, signum: int) -> None:
        """
        Signal handler that is added to the event loop.

        This catches the SIGTERM from K8s and begins graceful shutdown,
        by setting the _shutdown_event(), which is watched by the main
        coroutine monitoring the vLLM DP Servers.
        """

        if self._shutdown_event.is_set():
            return

        self._shutdown_signal = signal.Signals(signum)
        logger.info(
            "DPSupervisor received %s, shutting down.",
            self._shutdown_signal.name,
        )

        self._shutdown_event.set()
        self._is_ready = False

    def _start_children(self) -> None:
        """
        Launch vLLM DP Servers on separate GPUs.
        """
        logger.info("Launching vLLM DP Servers")
        context = multiprocessing.get_context("spawn")
        for local_rank in range(self.args.data_parallel_size_local):
            child_args = _build_vllm_dp_server_args(self.args, local_rank)
            process = context.Process(
                target=_run_vllm_dp_server,
                name=f"APIServer_DPRank_{child_args.data_parallel_rank}",
                args=(child_args,),
            )
            process.start()
            self._processes.append(process)

    async def _probe_all_children(self) -> None:
        """
        Background coroutine: probes all child endpoints on each interval.

        Exits when any server becomes unhealthy after being ready, signalling
        _monitor_children to initiate shutdown.
        """
        timeout = aiohttp.ClientTimeout(total=self.args.dp_supervisor_probe_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self._shutdown_event.is_set():
                threshold = (
                    self.args.dp_supervisor_probe_failure_threshold
                    if self._is_ready
                    else 1
                )
                results = await asyncio.gather(
                    *(
                        _probe_endpoint(
                            session,
                            self.args,
                            port,
                            "/health",
                            conn_err_failure_threshold=threshold,
                            conn_err_retry_delay=self.args.dp_supervisor_probe_interval_s,
                        )
                        for port in self.child_ports
                    ),
                    return_exceptions=True,
                )
                all_healthy = all(r is True for r in results)

                if all_healthy:
                    # If all healthy, we are ready to receive requests.
                    # This conditional avoids a potential race condition
                    # where shutdown is set, THEN the probe returns true.
                    if not self._shutdown_event.is_set():
                        provider = getattr(self.args, "snapshot_provider", None)
                        if not self._is_ready and provider is not None:
                            from vllm.engine.snapshot.manager import (
                                SnapshotManager,
                            )

                            logger.info("All DP children ready. Triggering snapshot...")
                            mgr = SnapshotManager(provider)
                            await asyncio.to_thread(mgr.run_snapshot)
                        self._is_ready = True
                elif self._is_ready:
                    # Once ready, any failure in the probe means vLLM is dead.
                    num_unhealthy = sum(1 for r in results if r is not True)
                    logger.info(
                        "DPSupervisor probe found %s unhealthy DP Servers.",
                        num_unhealthy,
                    )
                    self._is_ready = False
                    self._shutdown_event.set()
                    return

                with contextlib.suppress(asyncio.TimeoutError):
                    logger.debug(
                        "Waiting for %s seconds before next probe",
                        self.args.dp_supervisor_probe_interval_s,
                    )
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.args.dp_supervisor_probe_interval_s,
                    )

    async def _monitor_children(self) -> None:
        """
        Main coroutine task that monitors the children vLLM servers.

        Before the vLLM servers are /ready:
        - if the pid is dead, we will shut down
        - if the probe fails, we try again after dp_supervisor_probe_interval_s

        After the vLLM servers are /ready:
        - if the pid is dead, we will shut down
        - if the probe fails, we will shut down
        """
        probe_task = asyncio.create_task(
            self._probe_all_children(), name="dp-health-probe"
        )

        try:
            while not self._shutdown_event.is_set():
                # 1. Check for dead processes
                n_failed = len([p for p in self._processes if not p.is_alive()])
                if n_failed > 0:
                    logger.info("DPSupervisor found %s exited DP Servers.", n_failed)
                    break

                # 2. Check if the probe background task crashed or failed.
                if probe_task.done():
                    # Extract exception if it crashed, or log failure
                    exc = probe_task.exception() if not probe_task.cancelled() else None
                    if exc is not None:
                        logger.error(
                            "DPSupervisor probe task failed with exception: %s", exc
                        )
                        raise exc
                    logger.info("DPSupervisor probe task stopped clean/cancelled.")
                    break

                # Sleep for probe_interval seconds or until a shutdown.
                with contextlib.suppress(asyncio.TimeoutError):
                    logger.debug(
                        "Waiting for %s seconds before next monitor",
                        self.args.dp_supervisor_probe_interval_s,
                    )
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.args.dp_supervisor_probe_interval_s,
                    )

        finally:
            # Cleanup probe task if needed.
            if not probe_task.done():
                probe_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await probe_task

    async def _shutdown_children(self) -> None:
        """Terminate the vLLM DP servers."""
        timeout = self.args.shutdown_timeout + CHILD_EXIT_GRACE_S

        try:
            logger.info(
                "DPSupervisor forwarding %s to DP Servers.",
                self._shutdown_signal.name,
            )
            for process in self._processes:
                pid = process.pid
                if not process.is_alive() or pid is None:
                    continue
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.kill(pid, self._shutdown_signal)

            try:
                await asyncio.to_thread(
                    _join_processes_with_timeout, self._processes, timeout
                )
            except asyncio.CancelledError:
                logger.warning("Shutdown await cancelled")
                raise
        finally:
            for process in self._processes:
                pid = process.pid
                if not process.is_alive() or pid is None:
                    continue
                logger.warning(
                    "DP server %s did not exit within %.1fs; force killing.",
                    process.name,
                    timeout,
                )
                with contextlib.suppress(
                    ProcessLookupError,
                    OSError,
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                ):
                    kill_process_tree(pid)


def run_dp_supervisor(args: argparse.Namespace) -> None:
    uvloop.run(DPSupervisor(args).run())
