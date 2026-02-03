#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""
vLLM Render gRPC Server

A lightweight gRPC server for tokenization and chat message rendering.
This server does not run LLM inference - it only handles input preprocessing.

Usage:
    python -m vllm.entrypoints.render_server --model <model_path>

Example:
    python -m vllm.entrypoints.render_server \
        --model meta-llama/Llama-2-7b-hf \
        --host 0.0.0.0 \
        --port 50052
"""

import argparse
import asyncio
import signal
import sys
import time
from typing import Any

import grpc
import uvloop
from grpc_reflection.v1alpha import reflection

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.preprocess import preprocess_chat, preprocess_completion
from vllm.grpc import render_pb2, render_pb2_grpc
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.tokenizers.mistral import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def grpc_to_chat_request(
    grpc_request: render_pb2.RenderChatRequest,
    messages: list[dict[str, Any]],
) -> ChatCompletionRequest:
    """
    Convert gRPC RenderChatRequest to ChatCompletionRequest.

    This allows the gRPC server to use the same preprocessing logic
    (build_tok_params, build_chat_params) as the OpenAI HTTP serving layer.
    """
    # Convert documents
    documents = None
    if grpc_request.documents:
        documents = [
            {"title": doc.title, "text": doc.text} for doc in grpc_request.documents
        ]

    # Convert reasoning_effort
    reasoning_effort = None
    if grpc_request.HasField("reasoning_effort"):
        reasoning_effort = grpc_request.reasoning_effort  # type: ignore[assignment]

    return ChatCompletionRequest(
        messages=messages,  # type: ignore[arg-type]
        chat_template=grpc_request.chat_template
        if grpc_request.HasField("chat_template")
        else None,
        chat_template_kwargs=dict(grpc_request.chat_template_kwargs)
        if grpc_request.chat_template_kwargs
        else None,
        add_generation_prompt=grpc_request.add_generation_prompt,
        continue_final_message=grpc_request.continue_final_message,
        add_special_tokens=grpc_request.add_special_tokens,
        max_completion_tokens=grpc_request.max_completion_tokens
        if grpc_request.HasField("max_completion_tokens")
        else None,
        truncate_prompt_tokens=grpc_request.truncate_prompt_tokens
        if grpc_request.HasField("truncate_prompt_tokens")
        else None,
        documents=documents,
        reasoning_effort=reasoning_effort,
    )


def grpc_to_completion_request(
    grpc_request: render_pb2.RenderCompletionRequest,
    prompt: str | list[int],
) -> CompletionRequest:
    """
    Convert gRPC RenderCompletionRequest to CompletionRequest.

    This allows the gRPC server to use the same preprocessing logic
    (build_tok_params) as the OpenAI HTTP serving layer.
    """
    return CompletionRequest(
        prompt=prompt,  # type: ignore[arg-type]
        max_tokens=grpc_request.max_output_tokens or None,
        truncate_prompt_tokens=grpc_request.truncate_prompt_tokens
        if grpc_request.HasField("truncate_prompt_tokens")
        else None,
        add_special_tokens=grpc_request.add_special_tokens
        if grpc_request.HasField("add_special_tokens")
        else True,
    )


class RenderServicer(render_pb2_grpc.RenderServiceServicer):
    """
    gRPC servicer implementing the RenderService.

    Handles 6 RPCs:
    - RenderChat: Render chat messages to tokenized prompt
    - RenderCompletion: Render completion input to tokenized prompt
    - Tokenize: Tokenize text
    - Detokenize: Detokenize token IDs
    - GetInfo: Get renderer information
    - HealthCheck: Health probe
    """

    def __init__(
        self,
        renderer: BaseRenderer,
        model_config: ModelConfig,
        trust_request_chat_template: bool = False,
    ):
        """
        Initialize the servicer.

        Args:
            renderer: The BaseRenderer instance
            model_config: The model configuration
            trust_request_chat_template: Whether to trust chat templates from requests
        """
        self.renderer = renderer
        self.model_config = model_config
        self.trust_request_chat_template = trust_request_chat_template
        logger.info("RenderServicer initialized")

    async def RenderChat(
        self,
        request: render_pb2.RenderChatRequest,
        context: grpc.aio.ServicerContext,
    ) -> render_pb2.RenderChatResponse:
        """
        Render chat messages to tokenized prompt.

        Args:
            request: The RenderChatRequest protobuf
            context: gRPC context

        Returns:
            RenderChatResponse protobuf
        """
        logger.debug("RenderChat request received")

        try:
            # Convert protobuf messages to dict format
            messages = self._convert_messages(request.messages)
            tool_dicts = self._convert_tools(request.tools) if request.tools else None

            # Convert gRPC request to ChatCompletionRequest
            chat_request = grpc_to_chat_request(request, messages)

            # Preprocess for specific tokenizers
            tokenizer = self.renderer.tokenizer
            if isinstance(tokenizer, MistralTokenizer):
                maybe_serialize_tool_calls(chat_request)
                truncate_tool_call_ids(chat_request)
                validate_request_params(chat_request)

            if self.renderer.uses_harmony:
                maybe_serialize_tool_calls(chat_request)
            conversation, engine_prompts = await preprocess_chat(
                self.renderer,
                self.model_config,
                chat_request,
                messages,
                default_template=None,
                default_template_content_format="auto",
                default_template_kwargs={"tools": tool_dicts},
            )
            engine_prompt = engine_prompts[0]

            # Get token IDs
            prompt_token_ids = engine_prompt.get("prompt_token_ids", [])
            prompt_text = engine_prompt.get("prompt", "")

            # Convert conversation to protobuf
            conv_messages = [
                render_pb2.ConversationMessage(
                    role=msg.get("role", ""),
                    content=str(msg.get("content", "")),
                )
                for msg in conversation
            ]

            return render_pb2.RenderChatResponse(
                prompt_token_ids=prompt_token_ids,
                prompt_text=prompt_text,
                num_tokens=len(prompt_token_ids),
                conversation=conv_messages,
            )

        except Exception as e:
            logger.exception("Error in RenderChat")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RenderCompletion(
        self,
        request: render_pb2.RenderCompletionRequest,
        context: grpc.aio.ServicerContext,
    ) -> render_pb2.RenderCompletionResponse:
        """
        Render completion input to tokenized prompt.

        Args:
            request: The RenderCompletionRequest protobuf
            context: gRPC context

        Returns:
            RenderCompletionResponse protobuf
        """
        logger.debug("RenderCompletion request received")

        try:
            # Get input
            input_type = request.WhichOneof("input")
            if input_type == "text":
                prompt_input: str | list[int] = request.text
            elif input_type == "tokens":
                prompt_input = list(request.tokens.token_ids)
            else:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "No input provided"
                )
                return

            # Convert gRPC request to CompletionRequest and preprocess
            completion_request = grpc_to_completion_request(request, prompt_input)
            engine_prompts = await preprocess_completion(
                self.renderer,
                self.model_config,
                completion_request,
                prompt_input,
                None,  # prompt_embeds
            )
            engine_prompt = engine_prompts[0]
            prompt_token_ids = engine_prompt.get("prompt_token_ids", [])

            return render_pb2.RenderCompletionResponse(
                prompt_token_ids=prompt_token_ids,
                num_tokens=len(prompt_token_ids),
            )

        except Exception as e:
            logger.exception("Error in RenderCompletion")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Tokenize(
        self,
        request: render_pb2.TokenizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> render_pb2.TokenizeResponse:
        """
        Tokenize text.

        Args:
            request: The TokenizeRequest protobuf
            context: gRPC context

        Returns:
            TokenizeResponse protobuf
        """
        logger.debug("Tokenize request received")

        try:
            tokenizer = self.renderer.get_tokenizer()

            # Build encode kwargs
            encode_kwargs: dict[str, Any] = {
                "add_special_tokens": request.add_special_tokens,
            }
            if request.HasField("truncation"):
                encode_kwargs["truncation"] = True
                encode_kwargs["max_length"] = request.truncation

            token_ids = tokenizer.encode(request.text, **encode_kwargs)

            return render_pb2.TokenizeResponse(
                token_ids=token_ids,
                num_tokens=len(token_ids),
            )

        except Exception as e:
            logger.exception("Error in Tokenize")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Detokenize(
        self,
        request: render_pb2.DetokenizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> render_pb2.DetokenizeResponse:
        """
        Detokenize token IDs to text.

        Args:
            request: The DetokenizeRequest protobuf
            context: gRPC context

        Returns:
            DetokenizeResponse protobuf
        """
        logger.debug("Detokenize request received")

        try:
            tokenizer = self.renderer.get_tokenizer()
            text = tokenizer.decode(
                list(request.token_ids),
                skip_special_tokens=request.skip_special_tokens,
            )

            return render_pb2.DetokenizeResponse(text=text)

        except Exception as e:
            logger.exception("Error in Detokenize")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetInfo(
        self,
        request: render_pb2.GetInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> render_pb2.GetInfoResponse:
        """
        Get renderer information.

        Args:
            request: The GetInfoRequest protobuf
            context: gRPC context

        Returns:
            GetInfoResponse protobuf
        """
        logger.debug("GetInfo request received")

        tokenizer = self.renderer.tokenizer
        has_chat_template = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        )

        # Get special tokens
        special_tokens = []
        if tokenizer is not None:
            for attr in ["bos_token", "eos_token", "pad_token", "unk_token"]:
                token = getattr(tokenizer, attr, None)
                if token is not None:
                    special_tokens.append(token)

        return render_pb2.GetInfoResponse(
            model_path=self.model_config.model,
            max_model_len=self.model_config.max_model_len,
            vocab_size=self.model_config.get_vocab_size(),
            tokenizer_mode=self.model_config.tokenizer_mode,
            has_chat_template=has_chat_template,
            special_tokens=special_tokens,
            uses_harmony=self.renderer.uses_harmony,
        )

    async def HealthCheck(
        self,
        request: render_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> render_pb2.HealthCheckResponse:
        """
        Handle health check requests.

        Args:
            request: The HealthCheckRequest protobuf
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf
        """
        logger.debug("HealthCheck request received")

        return render_pb2.HealthCheckResponse(
            healthy=True,
            message="Healthy",
        )

    # ========== Helper methods ==========

    @staticmethod
    def _convert_messages(
        messages: list[render_pb2.ChatMessage],
    ) -> list[dict[str, Any]]:
        """Convert protobuf messages to dict format."""
        result = []
        for msg in messages:
            m: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.HasField("name"):
                m["name"] = msg.name
            if msg.HasField("tool_call_id"):
                m["tool_call_id"] = msg.tool_call_id
            if msg.HasField("reasoning"):
                m["reasoning"] = msg.reasoning
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(m)
        return result

    @staticmethod
    def _convert_tools(
        tools: list[render_pb2.ToolDefinition],
    ) -> list[dict[str, Any]]:
        """Convert protobuf tools to dict format."""
        result = []
        for tool in tools:
            t: dict[str, Any] = {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                },
            }
            if tool.function.HasField("description"):
                t["function"]["description"] = tool.function.description
            if tool.function.HasField("parameters"):
                import json

                t["function"]["parameters"] = json.loads(tool.function.parameters)
            result.append(t)
        return result


async def serve_render(args: argparse.Namespace):
    """
    Main serving function.

    Args:
        args: Parsed command line arguments
    """
    logger.info("vLLM Render gRPC server version: %s", VLLM_VERSION)
    logger.info("Model: %s", args.model)
    logger.info("vLLM Render gRPC server args: %s", args)

    start_time = time.time()

    # Create engine args for model config
    engine_args = EngineArgs(
        model=args.model,
        tokenizer=getattr(args, "tokenizer", None),
        tokenizer_mode=getattr(args, "tokenizer_mode", "auto"),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        revision=getattr(args, "revision", None),
        max_model_len=getattr(args, "max_model_len", None),
    )

    # Create vLLM config
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    model_config = vllm_config.model_config

    # Create renderer
    renderer = renderer_from_config(model_config)

    # Create servicer
    servicer = RenderServicer(
        renderer=renderer,
        model_config=model_config,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    # Create gRPC server
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )

    # Add servicer to server
    render_pb2_grpc.add_RenderServiceServicer_to_server(servicer, server)

    # Enable reflection for grpcurl and other tools
    service_names = (
        render_pb2.DESCRIPTOR.services_by_name["RenderService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    # Bind to address
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)

    # Start server
    await server.start()
    logger.info("vLLM Render gRPC server started on %s", address)
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
        logger.info("Shutting down vLLM Render gRPC server...")

        # Stop gRPC server
        await server.stop(grace=5.0)
        logger.info("gRPC server stopped")

        elapsed = time.time() - start_time
        logger.info("Shutdown complete (uptime: %.1f seconds)", elapsed)


def main():
    """Main entry point."""
    parser = FlexibleArgumentParser(
        description="vLLM Render gRPC Server",
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
        default=50052,
        help="Port to bind gRPC server to",
    )

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (defaults to model path)",
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "hf", "mistral"],
        help="Tokenizer mode",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision",
    )
    parser.add_argument(
        "--tokenizer-revision",
        type=str,
        default=None,
        help="Tokenizer revision",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--trust-request-chat-template",
        action="store_true",
        help="Trust chat templates from requests",
    )

    args = parser.parse_args()

    # Run server
    try:
        uvloop.run(serve_render(args))
    except Exception as e:
        logger.exception("Server failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
