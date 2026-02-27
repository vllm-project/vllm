# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
gRPC Render Server — lightweight chat template rendering + tokenization.

This server provides the RenderService gRPC API that maps 1:1 to the
vLLM renderer's internal Python types (ChatParams, TokenizeParams).
No GPU or LLM inference is required.
"""

import argparse
import asyncio
import signal
import time

import grpc
from grpc_reflection.v1alpha import reflection

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.grpc import render_pb2, render_pb2_grpc
from vllm.grpc.utils import (
    conversation_to_proto,
    messages_from_proto,
    params_from_proto,
)
from vllm.logger import init_logger
from vllm.renderers.params import ChatParams, TokenizeParams
from vllm.renderers.registry import renderer_from_config
from vllm.usage.usage_lib import UsageContext
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


# =====================
# gRPC Servicer
# =====================


class RenderServicer(render_pb2_grpc.RenderServiceServicer):
    """Implements the RenderService gRPC API."""

    def __init__(self, renderer, start_time: float):
        self.renderer = renderer
        self.start_time = start_time

    async def RenderChat(self, request, context):
        """Render chat messages and tokenize."""
        try:
            messages = messages_from_proto(request.messages)

            # Build ChatParams
            if request.HasField("chat_params"):
                chat_params = params_from_proto(
                    request.chat_params, ChatParams(), ChatParams
                )
            else:
                chat_params = ChatParams()

            # Build TokenizeParams (with renderer defaults)
            if request.HasField("tok_params"):
                tok_params = params_from_proto(
                    request.tok_params,
                    self.renderer.default_chat_tok_params,
                    TokenizeParams,
                )
            else:
                tok_params = self.renderer.default_chat_tok_params

            # Step 1: Render messages → (conversation, DictPrompt)
            conversation, dict_prompt = await self.renderer.render_messages_async(
                messages, chat_params
            )

            # Step 2: Tokenize → TokPrompt
            tok_prompt = await self.renderer.tokenize_prompt_async(
                dict_prompt, tok_params
            )

            prompt_token_ids = tok_prompt.get("prompt_token_ids", [])
            prompt_text = tok_prompt.get("prompt", "")

            return render_pb2.RenderChatResponse(
                prompt_token_ids=prompt_token_ids,
                prompt_text=prompt_text,
                num_tokens=len(prompt_token_ids),
                conversation=conversation_to_proto(conversation),
            )

        except Exception as e:
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"RenderChat failed: {e}",
            )

    async def RenderCompletion(self, request, context):
        """Tokenize a completion prompt."""
        try:
            # Build TokenizeParams (with renderer defaults)
            if request.HasField("tok_params"):
                tok_params = params_from_proto(
                    request.tok_params,
                    self.renderer.default_cmpl_tok_params,
                    TokenizeParams,
                )
            else:
                tok_params = self.renderer.default_cmpl_tok_params

            prompt_type = request.WhichOneof("prompt")
            if prompt_type == "text_prompt":
                dict_prompt = {"prompt": request.text_prompt}
            elif prompt_type == "token_ids_prompt":
                # Token passthrough — return as-is with validation
                token_ids = list(request.token_ids_prompt.token_ids)
                dict_prompt = {"prompt_token_ids": token_ids}
            else:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Either text_prompt or token_ids_prompt must be set",
                )
                return

            tok_prompt = await self.renderer.tokenize_prompt_async(
                dict_prompt, tok_params
            )

            prompt_token_ids = tok_prompt.get("prompt_token_ids", [])

            return render_pb2.RenderCompletionResponse(
                prompt_token_ids=prompt_token_ids,
                num_tokens=len(prompt_token_ids),
            )

        except Exception as e:
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"RenderCompletion failed: {e}",
            )

    async def HealthCheck(self, request, context):
        """Health check."""
        return render_pb2.HealthCheckResponse(
            healthy=True,
            message="OK",
        )


# =====================
# Server entrypoint
# =====================


async def serve_render(args: argparse.Namespace):
    """Start the gRPC render server."""
    logger.info("vLLM Render Server v%s", VLLM_VERSION)
    logger.info("Model: %s", args.model)

    start_time = time.time()

    # Build minimal engine config (no GPU needed)
    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=getattr(args, "tokenizer", None),
        tokenizer_mode=getattr(args, "tokenizer_mode", "auto"),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        revision=getattr(args, "revision", None),
        max_model_len=getattr(args, "max_model_len", None),
        enforce_eager=True,
        load_format="dummy",
    )
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # Create renderer (tokenizer only, no GPU)
    renderer = renderer_from_config(vllm_config)

    logger.info(
        "Renderer initialized in %.2fs (model_len=%d)",
        time.time() - start_time,
        vllm_config.model_config.max_model_len,
    )

    # Create servicer
    servicer = RenderServicer(renderer, start_time)

    # Create gRPC server
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )

    # Register servicer
    render_pb2_grpc.add_RenderServiceServicer_to_server(servicer, server)

    # Enable reflection for grpcurl and other tools
    service_names = (
        render_pb2.DESCRIPTOR.services_by_name["RenderService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    # Bind
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 50052)
    address = f"{host}:{port}"
    server.add_insecure_port(address)

    # Start server
    await server.start()
    logger.info("vLLM Render gRPC server started on %s", address)

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
        logger.info("Shutting down vLLM Render server...")
        await server.stop(grace=5.0)
        logger.info("Shutdown complete")
