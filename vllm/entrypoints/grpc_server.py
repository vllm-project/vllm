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
import hmac
import signal
import sys
import time

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
from vllm.entrypoints.serve.utils.api_utils import log_version_and_model
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

# Services exempt from API key authentication: health probes and reflection
_UNAUTHENTICATED_SERVICES = frozenset(
    {
        "grpc.health.v1.Health",
        "grpc.reflection.v1alpha.ServerReflection",
    }
)


class _APIKeyAuthInterceptor(grpc.aio.ServerInterceptor):
    """gRPC interceptor that enforces API key authentication.

    If no API keys are configured, all requests are allowed (backward
    compatible). If one or more keys are set, every RPC must carry an
    ``authorization`` metadata header whose value matches one of the keys.
    Health/reflection services are always allowed without authentication
    so that Kubernetes probes and debugging tools continue to work.
    """

    def __init__(self, api_keys: list[str]) -> None:
        self._api_keys = api_keys

    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        # Strip leading '/' from fully-qualified method name
        service_name = method.lstrip("/").rsplit("/", 1)[0]
        if service_name in _UNAUTHENTICATED_SERVICES:
            return await continuation(handler_call_details)

        if not self._api_keys:
            return await continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or {})
        auth_header = metadata.get("authorization", "")

        if not any(
            hmac.compare_digest(auth_header, key) for key in self._api_keys
        ):
            await handler_call_details.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "Invalid or missing API key",
            )
            # Return a no-op handler; abort() terminates the call.
            # We still need to return something for the protocol.

        return await continuation(handler_call_details)


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

    # Bind address
    host = args.host or "0.0.0.0"
    address = f"{host}:{args.port}"

    # Get API keys before building server (interceptors must be set at creation)
    api_keys = [k for k in (args.api_key or [envs.VLLM_API_KEY]) if k]

    interceptors: list[grpc.aio.ServerInterceptor] = []
    if api_keys:
        interceptors.append(_APIKeyAuthInterceptor(api_keys))
        logger.info(
            "gRPC API key authentication enabled (%d key%s configured)",
            len(api_keys),
            "s" if len(api_keys) != 1 else "",
        )
    else:
        logger.warning(
            "No --api-key set or VLLM_API_KEY env var found. gRPC server is "
            "running without authentication. Anyone who can reach this "
            "port (%s) has full access to the vLLM engine.",
            address,
        )

    # Create gRPC server
    server = grpc.aio.server(
        interceptors=interceptors,
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

        def signal_handler():
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
        await server.stop(grace=5.0)
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
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        action="append",
        help="API key(s) for authenticating gRPC requests. "
        "If provided, clients must include a valid "
        "'authorization' metadata header.",
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
