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
import uvloop
from grpc_reflection.v1alpha import reflection

from vllm import SamplingParams, TextPrompt, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.utils import log_version_and_model
from vllm.grpc import vllm_engine_pb2, vllm_engine_pb2_grpc
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class VllmEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """
    gRPC servicer implementing the VllmEngine service.

    Handles 6 RPCs:
    - Generate: Streaming text generation
    - Embed: Embeddings (TODO)
    - HealthCheck: Health probe
    - Abort: Cancel requests out-of-band
    - GetModelInfo: Model metadata
    - GetServerInfo: Server state
    """

    def __init__(self, async_llm: AsyncLLM, start_time: float):
        """
        Initialize the servicer.

        Args:
            async_llm: The AsyncLLM instance
            start_time: The server start time, in seconds since epoch
        """
        self.async_llm = async_llm
        self.start_time = start_time
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
        logger.debug("Generate request %s received.", request_id)

        try:
            # Extract tokenized input
            if request.WhichOneof("input") == "tokenized":
                prompt: TokensPrompt = {
                    "prompt_token_ids": list(request.tokenized.input_ids)
                }
                if request.tokenized.original_text:
                    prompt["prompt"] = request.tokenized.original_text
            else:
                prompt: TextPrompt = {"prompt": request.text}

            # Build sampling params with detokenize=False
            sampling_params = self._sampling_params_from_proto(
                request.sampling_params, stream=request.stream
            )

            async for output in self.async_llm.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                # Convert vLLM output to protobuf
                # For streaming, always send chunks
                if request.stream:
                    yield self._chunk_response(output)

                # Send complete response when finished
                if output.finished:
                    yield self._complete_response(output)

        except ValueError as e:
            # Invalid request error (equiv to 400).
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Error in Generate for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

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
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED, "Embed RPC not yet implemented"
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
        is_healthy = not self.async_llm.errored
        message = "Health" if is_healthy else "Engine is not alive"

        logger.debug("HealthCheck request: healthy=%s, message=%s", is_healthy, message)

        return vllm_engine_pb2.HealthCheckResponse(healthy=is_healthy, message=message)

    async def Abort(
        self,
        request: vllm_engine_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.AbortResponse:
        """
        Out-of-band abort requests.

        Args:
            request: The AbortRequest protobuf
            context: gRPC context

        Returns:
            AbortResponse protobuf
        """
        request_ids = request.request_ids
        logger.debug("Abort requests: %s", request_ids)

        await self.async_llm.abort(request_ids)
        return vllm_engine_pb2.AbortResponse()

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
        model_config = self.async_llm.model_config

        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=model_config.model,
            is_generation=model_config.runner_type == "generate",
            max_context_length=model_config.max_model_len,
            vocab_size=model_config.get_vocab_size(),
            supports_vision=model_config.is_multimodal_model,
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
        # TODO: last_receive_timestamp is meant to be the time of the last received
        # request, not the current time; track it when requests are received.
        last_receive_timestamp = time.time()
        num_requests = self.async_llm.output_processor.get_num_unfinished_requests()

        return vllm_engine_pb2.GetServerInfoResponse(
            active_requests=num_requests,
            is_paused=self.async_llm.is_paused,
            last_receive_timestamp=last_receive_timestamp,
            uptime_seconds=last_receive_timestamp - self.start_time,
            server_type="vllm-grpc",
        )

    # ========== Helper methods ==========

    @staticmethod
    def _sampling_params_from_proto(
        params: vllm_engine_pb2.SamplingParams, stream: bool = True
    ) -> SamplingParams:
        """
        Convert protobuf SamplingParams to vLLM SamplingParams.

        Args:
            params: Protobuf SamplingParams message
            stream: Whether streaming is enabled

        Returns:
            vLLM SamplingParams with detokenize=False and structured_outputs
        """
        # Build stop sequences
        stop = list(params.stop) if params.stop else None
        stop_token_ids = list(params.stop_token_ids) if params.stop_token_ids else None

        # Handle structured outputs constraints
        structured_outputs = None
        constraint_field = params.WhichOneof("constraint")
        if constraint_field:
            if constraint_field == "json_schema":
                structured_outputs = StructuredOutputsParams(json=params.json_schema)
            elif constraint_field == "regex":
                structured_outputs = StructuredOutputsParams(regex=params.regex)
            elif constraint_field == "grammar":
                structured_outputs = StructuredOutputsParams(grammar=params.grammar)
            elif constraint_field == "structural_tag":
                structured_outputs = StructuredOutputsParams(
                    structural_tag=params.structural_tag
                )
            elif constraint_field == "json_object":
                structured_outputs = StructuredOutputsParams(
                    json_object=params.json_object
                )
            elif constraint_field == "choice":
                structured_outputs = StructuredOutputsParams(
                    choice=list(params.choice.choices)
                )

        # Create SamplingParams
        # output_kind=DELTA: Return only new tokens in each chunk (for streaming)
        return SamplingParams(
            temperature=params.temperature if params.HasField("temperature") else 1.0,
            top_p=params.top_p if params.top_p != 0.0 else 1.0,
            top_k=params.top_k,
            min_p=params.min_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            repetition_penalty=params.repetition_penalty
            if params.repetition_penalty != 0.0
            else 1.0,
            max_tokens=params.max_tokens if params.HasField("max_tokens") else None,
            min_tokens=params.min_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
            ignore_eos=params.ignore_eos,
            n=params.n if params.n > 0 else 1,
            logprobs=params.logprobs if params.HasField("logprobs") else None,
            prompt_logprobs=params.prompt_logprobs
            if params.HasField("prompt_logprobs")
            else None,
            seed=params.seed if params.HasField("seed") else None,
            include_stop_str_in_output=params.include_stop_str_in_output,
            logit_bias=dict(params.logit_bias) if params.logit_bias else None,
            truncate_prompt_tokens=params.truncate_prompt_tokens
            if params.HasField("truncate_prompt_tokens")
            else None,
            structured_outputs=structured_outputs,
            # detokenize must be True if stop strings are used
            detokenize=bool(stop),
            output_kind=RequestOutputKind.DELTA
            if stream
            else RequestOutputKind.FINAL_ONLY,
        )

    @staticmethod
    def _chunk_response(output: RequestOutput) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a streaming chunk response from vLLM output.
        When output_kind=DELTA, vLLM returns only new tokens automatically.

        Args:
            output: vLLM RequestOutput (with delta tokens when output_kind=DELTA)

        Returns:
            GenerateResponse with chunk field set
        """
        # Get the completion output (first one if n > 1)
        completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty chunk
            return vllm_engine_pb2.GenerateResponse(
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
            chunk=vllm_engine_pb2.GenerateStreamChunk(
                token_ids=completion.token_ids,
                prompt_tokens=len(output.prompt_token_ids)
                if output.prompt_token_ids
                else 0,
                completion_tokens=len(completion.token_ids),  # Delta count
                cached_tokens=output.num_cached_tokens,
            ),
        )

    @staticmethod
    def _complete_response(output: RequestOutput) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a final completion response from vLLM output.

        Args:
            output: vLLM RequestOutput (finished=True)

        Returns:
            GenerateResponse with complete field set
        """
        # Get the completion output (first one if n > 1)
        completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty completion
            return vllm_engine_pb2.GenerateResponse(
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
        # When non-streaming (FINAL_ONLY mode): completion.token_ids has all tokens
        # Client will accumulate token counts for streaming
        return vllm_engine_pb2.GenerateResponse(
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


async def serve_grpc(args: argparse.Namespace):
    """
    Main serving function.

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
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # Create AsyncLLM
    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=args.enable_log_requests,
        disable_log_stats=args.disable_log_stats_server,
    )

    # Create servicer
    servicer = VllmEngineServicer(async_llm, start_time)

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
        async_llm.shutdown()
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
        "--disable-log-stats-server",
        action="store_true",
        help="Disable stats logging on server side",
    )

    # Add vLLM engine args
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
