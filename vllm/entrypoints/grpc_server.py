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
import importlib
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any

try:
    import grpc
    from google.protobuf.symbol_database import Default as symbol_db_default
    from grpc_reflection.v1alpha import reflection
    from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
except ImportError:
    raise ImportError(
        "smg-grpc-servicer is required for gRPC mode. "
        "Install it with: pip install vllm[grpc]"
    ) from None

import uvloop

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.grpc_kv_events import GrpcKvEventStreamer
from vllm.entrypoints.utils import log_version_and_model
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def _get_vllm_engine_service_descriptor(pb2_module: Any) -> Any | None:
    return pb2_module.DESCRIPTOR.services_by_name.get("VllmEngine")


def _get_vllm_engine_service_full_name(pb2_module: Any) -> str:
    service_descriptor = _get_vllm_engine_service_descriptor(pb2_module)
    if service_descriptor is None:
        raise RuntimeError("VllmEngine service descriptor is missing")
    return service_descriptor.full_name


def _supports_subscribe_kv_events(pb2_module: Any) -> bool:
    service_descriptor = _get_vllm_engine_service_descriptor(pb2_module)
    if service_descriptor is None:
        return False
    return "SubscribeKvEvents" in service_descriptor.methods_by_name


def _resolve_kv_proto_bindings(default_pb2_module: Any) -> tuple[Any, Any, Any]:
    """Resolve request/response protobuf classes for SubscribeKvEvents.

    Handles both layouts:
    - vLLM local proto: messages are in `vllm_engine_pb2`.
    - SMG proto: service is in `vllm_engine_pb2`, messages are in `common_pb2`.
    """
    service_descriptor = _get_vllm_engine_service_descriptor(default_pb2_module)
    if (
        service_descriptor is not None
        and "SubscribeKvEvents" in service_descriptor.methods_by_name
    ):
        subscribe_desc = service_descriptor.methods_by_name["SubscribeKvEvents"]
        sym_db = symbol_db_default()
        request_cls = sym_db.GetSymbol(subscribe_desc.input_type.full_name)
        response_cls = sym_db.GetSymbol(subscribe_desc.output_type.full_name)

        response_pb2 = importlib.import_module(response_cls.__module__)
        pb2_bundle = SimpleNamespace(
            KvEventBatch=response_pb2.KvEventBatch,
            KvCacheEvent=response_pb2.KvCacheEvent,
            KvBlocksStored=response_pb2.KvBlocksStored,
            KvBlock=response_pb2.KvBlock,
            KvBlocksRemoved=response_pb2.KvBlocksRemoved,
            KvCacheCleared=response_pb2.KvCacheCleared,
        )
        return request_cls, response_cls, pb2_bundle

    from vllm.grpc import vllm_engine_pb2 as local_pb2

    return (
        local_pb2.SubscribeKvEventsRequest,
        local_pb2.KvEventBatch,
        local_pb2,
    )


def _register_subscribe_kv_events(
    server: grpc.aio.Server,
    servicer: Any,
    async_llm: AsyncLLM,
) -> None:
    request_cls, response_cls, kv_pb2_module = _resolve_kv_proto_bindings(
        vllm_engine_pb2
    )
    kv_streamer = GrpcKvEventStreamer(
        kv_events_config=async_llm.vllm_config.kv_events_config,
        data_parallel_size=async_llm.vllm_config.parallel_config.data_parallel_size,
        pb2_module=kv_pb2_module,
    )

    async def subscribe_handler(request, context):
        async for batch in kv_streamer.subscribe(request, context):
            yield batch

    if _supports_subscribe_kv_events(vllm_engine_pb2):
        # Override the method on the runtime instance before registration.
        servicer.SubscribeKvEvents = subscribe_handler
        logger.info("Registered SubscribeKvEvents using smg_grpc_proto service.")
        return

    # Fallback: register this single method directly if the imported service
    # descriptor is older and does not include SubscribeKvEvents yet.
    method_handler = grpc.unary_stream_rpc_method_handler(
        subscribe_handler,
        request_deserializer=request_cls.FromString,
        response_serializer=response_cls.SerializeToString,
    )
    service_full_name = _get_vllm_engine_service_full_name(vllm_engine_pb2)
    generic_handler = grpc.method_handlers_generic_handler(
        service_full_name,
        {"SubscribeKvEvents": method_handler},
    )
    server.add_generic_rpc_handlers((generic_handler,))
    logger.warning(
        "Registered SubscribeKvEvents as a fallback generic handler because "
        "the imported gRPC service descriptor does not include it."
    )


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

    # Register SubscribeKvEvents before adding the main servicer so method
    # overrides are picked up during handler construction.
    _register_subscribe_kv_events(server, servicer, async_llm)

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
