#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""
vLLM gRPC Server

Starts a gRPC server backed by AsyncLLM, using the VllmEngineServicer
from the smg-grpc-servicer package.

Usage:
    python -m vllm.entrypoints.grpc_server --model <model_path>

Example:
    python -m vllm.entrypoints.grpc_server \
        --model meta-llama/Llama-2-7b-hf \
        --host 0.0.0.0 \
        --port 50051
"""

import argparse
import asyncio
import signal
import sys
import time

try:
    import grpc
    from grpc_reflection.v1alpha import reflection
    from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
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
from vllm.entrypoints.utils import log_version_and_model
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


async def serve_grpc(args: argparse.Namespace):
    """
    Main gRPC serving function.

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
            # Tolerate client keepalive pings every 10s (default 300s is too
            # strict for non-streaming requests where no DATA frames flow
            # during generation)
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    # Add servicer to server
    vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

    # Enable reflection for grpcurl and other tools
    service_names = (
        vllm_engine_pb2.DESCRIPTOR.services_by_name["VllmEngine"].full_name,
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
        # First signal → graceful drain. Second signal → force immediate exit.
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        force_exit = False

        def signal_handler():
            nonlocal force_exit
            if stop_event.is_set():
                logger.warning("Received second shutdown signal, forcing exit.")
                force_exit = True
                # Shorten the grace period if server.stop is already running.
                loop.create_task(server.stop(0))
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

        # Stop gRPC server
        grace = 0.0 if force_exit else 5.0
        await server.stop(grace=grace)
        logger.info("gRPC server stopped")
        async_llm.shutdown()
        logger.info("AsyncLLM engine stopped")
        logger.info("Shutdown complete")


def main():
    """Main entry point for python -m vllm.entrypoints.grpc_server."""
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

    # Run server
    try:
        uvloop.run(serve_grpc(args))
    except Exception as e:
        logger.exception("Server failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
