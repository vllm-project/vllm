#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""
vLLM gRPC Server

Starts a gRPC server for vLLM using the VllmEngine protocol.

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
from collections.abc import AsyncGenerator

import grpc
from grpc_reflection.v1alpha import reflection

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.grpc import vllm_engine_pb2, vllm_engine_pb2_grpc
from vllm.grpc.grpc_request_manager import (
    GrpcRequestManager,
    create_sampling_params_from_proto,
)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)


class VllmEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """
    gRPC servicer implementing the VllmEngine service.

    Handles 6 RPCs:
    - Generate: Streaming text generation
    - Embed: Embeddings (TODO)
    - HealthCheck: Health probe
    - Abort: Cancel a request
    - GetModelInfo: Model metadata
    - GetServerInfo: Server state
    """

    def __init__(self, request_manager: GrpcRequestManager):
        """
        Initialize the servicer.

        Args:
            request_manager: The GrpcRequestManager instance
        """
        self.request_manager = request_manager
        logger.info("VllmEngineServicer initialized")

    async def Generate(
        self,
        request: vllm_engine_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[vllm_engine_pb2.GenerateResponse, None]:
        """
        Handle streaming generation requests.

        Args:
            request: The GenerateRequest protobuf
            context: gRPC context

        Yields:
            GenerateResponse protobuf messages (streaming)
        """
        request_id = request.request_id
        logger.info("Generate request %s received.", request_id)

        try:
            # Extract tokenized input
            if not request.HasField("tokenized"):
                yield self._error_response(
                    request_id,
                    "Missing tokenized input",
                    "400",
                )
                return

            prompt_token_ids = list(request.tokenized.input_ids)

            # Build sampling params with detokenize=False
            sampling_params = create_sampling_params_from_proto(
                request.sampling_params,
                stream=request.stream,
            )

            # Submit to request manager and stream outputs
            arrival_time = time.time()

            async for output in self.request_manager.generate(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                arrival_time=arrival_time,
            ):
                # Check if client disconnected
                if context.cancelled():
                    logger.info("Client disconnected for %s.", request_id)
                    await self.request_manager.abort(request_id)
                    return

                # Convert vLLM output to protobuf
                # For streaming, always send chunks
                if request.stream:
                    yield self._chunk_response(request_id, output)

                # Send complete response when finished
                if output.finished:
                    yield self._complete_response(request_id, output)

        except Exception as e:
            logger.error("Error in Generate for %s: %s", request_id, e)
            yield self._error_response(
                request_id,
                str(e),
                "500",
            )

    async def Embed(
        self,
        request: vllm_engine_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.EmbedResponse:
        """
        Handle embedding requests.

        TODO: Implement in Phase 4

        Args:
            request: The EmbedRequest protobuf
            context: gRPC context

        Returns:
            EmbedResponse protobuf
        """
        logger.warning("Embed RPC not yet implemented")
        return vllm_engine_pb2.EmbedResponse(
            request_id=request.request_id,
            error=vllm_engine_pb2.EmbedError(
                message="Embed RPC not yet implemented",
                code="NOT_IMPLEMENTED",
            ),
        )

    async def HealthCheck(
        self,
        request: vllm_engine_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.HealthCheckResponse:
        """
        Handle health check requests.

        Args:
            request: The HealthCheckRequest protobuf
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf
        """
        is_healthy, message = await self.request_manager.health_check()

        logger.info("HealthCheck request: healthy=%s, message=%s", is_healthy, message)

        return vllm_engine_pb2.HealthCheckResponse(
            healthy=is_healthy,
            message=message,
        )

    async def Abort(
        self,
        request: vllm_engine_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.AbortResponse:
        """
        Handle abort requests.

        Args:
            request: The AbortRequest protobuf
            context: gRPC context

        Returns:
            AbortResponse protobuf
        """
        request_id = request.request_id
        logger.info("Abort request for %s.", request_id)

        success = await self.request_manager.abort(request_id)

        return vllm_engine_pb2.AbortResponse(
            success=success,
            message=f"Request {request_id} {'aborted' if success else 'not found'}",
        )

    async def GetModelInfo(
        self,
        request: vllm_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetModelInfoResponse:
        """
        Handle model info requests.

        Args:
            request: The GetModelInfoRequest protobuf
            context: gRPC context

        Returns:
            GetModelInfoResponse protobuf
        """
        model_config = self.request_manager.get_model_config()

        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=model_config.get("model_path", ""),
            is_generation=model_config.get("is_generation", True),
            max_context_length=model_config.get("max_context_length", 0),
            vocab_size=model_config.get("vocab_size", 0),
            supports_vision=model_config.get("supports_vision", False),
        )

    async def GetServerInfo(
        self,
        request: vllm_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetServerInfoResponse:
        """
        Handle server info requests.

        Args:
            request: The GetServerInfoRequest protobuf
            context: gRPC context

        Returns:
            GetServerInfoResponse protobuf
        """
        num_requests = self.request_manager.get_num_unfinished_requests()

        return vllm_engine_pb2.GetServerInfoResponse(
            active_requests=num_requests,
            is_paused=False,
            last_receive_timestamp=time.time(),
            uptime_seconds=0.0,  # TODO: track server start time
            server_type="vllm-grpc",
        )

    # ========== Helper methods ==========

    def _chunk_response(
        self,
        request_id: str,
        output,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a streaming chunk response from vLLM output.
        When output_kind=DELTA, vLLM returns only new tokens automatically.

        Args:
            request_id: The request ID
            output: vLLM RequestOutput (with delta tokens when output_kind=DELTA)

        Returns:
            GenerateResponse with chunk field set
        """
        # Get the completion output (first one if n > 1)
        completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty chunk
            return vllm_engine_pb2.GenerateResponse(
                request_id=request_id,
                chunk=vllm_engine_pb2.GenerateStreamChunk(
                    token_ids=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                ),
            )

        # When output_kind=DELTA, completion.token_ids contains only new tokens
        # vLLM handles the delta logic internally
        # completion_tokens = delta count (client will accumulate)
        return vllm_engine_pb2.GenerateResponse(
            request_id=request_id,
            chunk=vllm_engine_pb2.GenerateStreamChunk(
                token_ids=completion.token_ids,
                prompt_tokens=len(output.prompt_token_ids)
                if output.prompt_token_ids
                else 0,
                completion_tokens=len(completion.token_ids),  # Delta count
                cached_tokens=output.num_cached_tokens,
            ),
        )

    def _complete_response(
        self,
        request_id: str,
        output,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a final completion response from vLLM output.

        Args:
            request_id: The request ID
            output: vLLM RequestOutput (finished=True)

        Returns:
            GenerateResponse with complete field set
        """
        # Get the completion output (first one if n > 1)
        completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty completion
            return vllm_engine_pb2.GenerateResponse(
                request_id=request_id,
                complete=vllm_engine_pb2.GenerateComplete(
                    output_ids=[],
                    finish_reason="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                ),
            )

        # Build complete response
        # When streaming (DELTA mode): completion.token_ids will be empty/last delta
        # When non-streaming (CUMULATIVE mode): completion.token_ids has all tokens
        # Client will accumulate token counts for streaming
        return vllm_engine_pb2.GenerateResponse(
            request_id=request_id,
            complete=vllm_engine_pb2.GenerateComplete(
                output_ids=completion.token_ids,
                finish_reason=completion.finish_reason or "stop",
                prompt_tokens=len(output.prompt_token_ids)
                if output.prompt_token_ids
                else 0,
                completion_tokens=len(completion.token_ids),
                cached_tokens=output.num_cached_tokens,
            ),
        )

    def _error_response(
        self,
        request_id: str,
        message: str,
        status_code: str,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build an error response.

        Args:
            request_id: The request ID
            message: Error message
            status_code: HTTP-style status code

        Returns:
            GenerateResponse with error field set
        """
        return vllm_engine_pb2.GenerateResponse(
            request_id=request_id,
            error=vllm_engine_pb2.GenerateError(
                message=message,
                http_status_code=status_code,
                details="",
            ),
        )


async def serve_grpc(args: argparse.Namespace):
    """
    Main serving function.

    Args:
        args: Parsed command line arguments
    """
    logger.info("Initializing vLLM gRPC server...")

    # Create engine args
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Build vLLM config
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # Create AsyncLLM
    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=not args.disable_log_requests_server,
        disable_log_stats=args.disable_log_stats_server,
    )

    logger.info("Model: %s", vllm_config.model_config.model)
    logger.info("Max model len: %s", vllm_config.model_config.max_model_len)
    logger.info("Vocab size: %s", vllm_config.model_config.get_vocab_size())

    # Create request manager
    request_manager = GrpcRequestManager(async_llm)

    # Create servicer
    servicer = VllmEngineServicer(request_manager)

    # Create gRPC server
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
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
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)

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

    # Serve until shutdown signal
    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Shutting down vLLM gRPC server...")

        # Stop gRPC server
        await server.stop(grace=5.0)
        logger.info("gRPC server stopped")

        # Shutdown AsyncLLM
        await async_llm.shutdown()
        logger.info("AsyncLLM engine stopped")

        logger.info("Shutdown complete")


def main():
    """Main entry point."""
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
        "--disable-log-requests-server",
        action="store_true",
        help="Disable request logging on server side",
    )
    parser.add_argument(
        "--disable-log-stats-server",
        action="store_true",
        help="Disable stats logging on server side",
    )

    # Add vLLM engine args
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()

    # Run server
    try:
        asyncio.run(serve_grpc(args))
    except Exception as e:
        logger.exception("Server failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
