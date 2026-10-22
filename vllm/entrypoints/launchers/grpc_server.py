#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""vLLM gRPC Server

Starts a gRPC server backed by AsyncLLM, using the VllmEngineServicer
from the smg-grpc-servicer package.

Usage:
    python -m vllm.entrypoints.launchers.grpc_server --model <model_path>

Example:
    python -m vllm.entrypoints.launchers.grpc_server \
        --model meta-llama/Llama-2-7b-hf \
        --host 0.0.0.0 \
        --port 50051

"""

import argparse
import asyncio
import signal
import sys
import time
from functools import partial

try:
    import grpc
    from grpc_health.v1 import health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
    from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer
    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
except ImportError as e:
    raise ImportError(
        "gRPC mode requires smg-grpc-servicer. "
        "If not installed, run: pip install vllm[grpc]. "
        "If already installed, there may be a broken import due to a "
        "version mismatch — see the chained exception above for details."
    ) from e

import uvloop

from vllm import envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.common.startup import log_version_and_model
from vllm.logger import configure_logging_from_args, init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

# Teardown time EngineCore keeps past the drain budget (NIXL, device memory,
# shm), as the default 5 s join gave it before.
_ENGINE_TEARDOWN_S = 5.0
# Time RPCs get to send their last responses once EngineCore has stopped;
# whatever is still open after that (e.g. KV-event streams) is cancelled.
_RPC_FLUSH_S = 5.0
# How often to check whether the in-flight RPCs still have requests.
_IDLE_POLL_S = 0.1


async def _wait_for_requests(rpcs: asyncio.Future, async_llm: AsyncLLM) -> None:
    """Return once ``rpcs`` is done or the RPCs have no request in flight.

    EngineCore rejects every request it reads once its shutdown has started,
    even one an RPC forwarded before SIGTERM, so it must keep running until
    the requests are done. RPCs that carry none, such as KV-event streams, do
    not hold it back. Idle has to hold on two polls in a row, which gives an
    RPC accepted just before SIGTERM a poll interval to add its request.
    """
    idle = False
    while not rpcs.done():
        was_idle, idle = idle, async_llm.get_num_unfinished_requests() == 0
        if was_idle and idle:
            return
        await asyncio.wait([rpcs], timeout=_IDLE_POLL_S)


async def _drain_and_stop(server, async_llm: AsyncLLM, shutdown_timeout: int) -> None:
    """Stop the gRPC server and AsyncLLM within ``--shutdown-timeout``.

    0 keeps the abort behavior: in-flight RPCs get 5 s, then EngineCore gets
    its default join. Otherwise the server refuses new RPCs at once and
    EngineCore keeps serving until the in-flight requests are done; RPCs still
    running at the deadline are cancelled and abort their requests. EngineCore
    then drains work that outlives an RPC, such as NIXL prefill blocks the
    decode side has not read yet, until the same deadline plus
    ``_ENGINE_TEARDOWN_S`` for teardown.
    """
    loop = asyncio.get_running_loop()
    if shutdown_timeout == 0:
        await server.stop(grace=5.0)
        logger.info("gRPC server stopped")
        await loop.run_in_executor(None, async_llm.shutdown)
        logger.info("AsyncLLM engine stopped")
        return

    deadline = loop.time() + shutdown_timeout
    logger.info("Draining in-flight requests (shutdown_timeout=%ds)", shutdown_timeout)
    rpcs = asyncio.ensure_future(server.stop(grace=shutdown_timeout))
    await _wait_for_requests(rpcs, async_llm)
    remaining = max(deadline - loop.time(), 0.0)
    await loop.run_in_executor(
        None, partial(async_llm.shutdown, timeout=remaining + _ENGINE_TEARDOWN_S)
    )
    logger.info("AsyncLLM engine stopped")
    await asyncio.wait([rpcs], timeout=_RPC_FLUSH_S)
    await server.stop(grace=None)
    await rpcs
    logger.info("gRPC server stopped")


async def serve_grpc(args: argparse.Namespace):
    """Main gRPC serving function.

    Args:
        args: Parsed command line arguments

    """
    log_version_and_model(logger, VLLM_VERSION, args.model)
    logger.info("vLLM gRPC server args: %s", args)

    start_time = time.time()

    # Create engine args
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Build vLLM config
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    # Create AsyncLLM
    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=args.enable_log_requests,
        disable_log_stats=args.disable_log_stats,
    )

    # Create servicer
    servicer = VllmEngineServicer(async_llm, start_time)

    # Create gRPC server
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            # A non-streaming Generate writes nothing while the engine
            # decodes, so client keepalive and BDP-probe PINGs arriving in
            # that window are strikes and eventually earn a
            # GOAWAY(ENHANCE_YOUR_CALM, "too_many_pings") that fails every
            # in-flight RPC. Disable the strike count to tolerate them.
            ("grpc.http2.max_ping_strikes", 0),
            # GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS expands to
            # a string with no "recv", and gRPC Core silently ignores unknown
            # keys, so the misspelt form left the 300s default in force.
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            # Without this, gRPC Core widens the floor to a fixed two hours
            # once no streams are open, striking keepalives between requests.
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    # Add servicer to server
    vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

    # Add standard gRPC health service for Kubernetes probes
    health_servicer = VllmHealthServicer(async_llm)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Enable reflection for grpcurl and other tools
    service_names = (
        vllm_engine_pb2.DESCRIPTOR.services_by_name["VllmEngine"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    # Bind to address
    host = args.host or "0.0.0.0"
    address = f"{host}:{args.port}"
    server.add_insecure_port(address)

    try:
        # Start server
        await server.start()
        logger.info("vLLM gRPC server started on %s", address)
        logger.info("Server is ready to accept requests")

        # Start periodic stats logging (mirrors the HTTP server's lifespan task)
        if not args.disable_log_stats:

            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await async_llm.do_log_stats()

            stats_task = asyncio.create_task(_force_log())
        else:
            stats_task = None

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        force_stops = []

        def signal_handler():
            if stop_event.is_set():
                # A second signal skips the drain: cancelling the RPCs aborts
                # their requests, so EngineCore runs out of work and exits.
                logger.info("Received another shutdown signal; cancelling RPCs")
                force_stops.append(loop.create_task(server.stop(grace=None)))
                return
            logger.info("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
    finally:
        logger.info("Shutting down vLLM gRPC server...")
        if stats_task is not None:
            stats_task.cancel()
        try:
            health_servicer.set_not_serving()
        except Exception:  # broad: must not prevent server.stop() / shutdown()
            logger.warning("Failed to set health status to NOT_SERVING", exc_info=True)
        await _drain_and_stop(server, async_llm, vllm_config.shutdown_timeout)
        logger.info("Shutdown complete")


def main():
    """Main entry point for python -m vllm.entrypoints.launchers.grpc_server."""
    parser = FlexibleArgumentParser(
        description="vLLM gRPC Server",
    )

    # Server args
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind gRPC server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind gRPC server to",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    configure_logging_from_args(args)

    # Run server
    try:
        uvloop.run(serve_grpc(args))
    except Exception as e:
        logger.exception("Server failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
