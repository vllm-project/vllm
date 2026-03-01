# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Render Server — lightweight chat template rendering + tokenization.

This server provides gRPC or HTTP APIs that map 1:1 to the vLLM
renderer's internal Python types (ChatParams, TokenizeParams).
No GPU or LLM inference is required.
"""

import argparse
import asyncio
import signal
import time
from typing import Any

import grpc
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from grpc_reflection.v1alpha import reflection

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.grpc import render_pb2, render_pb2_grpc  # type: ignore[attr-defined]
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
# Shared render logic
# =====================


async def _render_chat(renderer, body: dict) -> dict:
    """Shared logic for RenderChat (used by both gRPC and HTTP)."""
    messages = body.get("messages", [])

    # Build ChatParams
    chat_kwargs = body.get("chat_params") or {}
    chat_params = ChatParams(**chat_kwargs) if chat_kwargs else ChatParams()

    # Build TokenizeParams (with renderer defaults)
    tok_kwargs = body.get("tok_params") or {}
    if tok_kwargs:
        defaults = renderer.default_chat_tok_params
        merged = {
            f.name: tok_kwargs.get(f.name, getattr(defaults, f.name))
            for f in TokenizeParams.__dataclass_fields__.values()
        }
        tok_params = TokenizeParams(**merged)
    else:
        tok_params = renderer.default_chat_tok_params

    # Step 1: Render messages → (conversation, DictPrompt)
    conversation, dict_prompt = await renderer.render_messages_async(
        messages, chat_params
    )

    # Step 2: Tokenize → TokPrompt
    tok_prompt = await renderer.tokenize_prompt_async(dict_prompt, tok_params)

    prompt_token_ids = tok_prompt.get("prompt_token_ids", [])
    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_text": tok_prompt.get("prompt", ""),
        "num_tokens": len(prompt_token_ids),
        "conversation": conversation,
    }


async def _render_completion(renderer, body: dict) -> dict:
    """Shared logic for RenderCompletion (used by both gRPC and HTTP)."""
    # Build TokenizeParams (with renderer defaults)
    tok_kwargs = body.get("tok_params") or {}
    if tok_kwargs:
        defaults = renderer.default_cmpl_tok_params
        merged = {
            f.name: tok_kwargs.get(f.name, getattr(defaults, f.name))
            for f in TokenizeParams.__dataclass_fields__.values()
        }
        tok_params = TokenizeParams(**merged)
    else:
        tok_params = renderer.default_cmpl_tok_params

    # Determine prompt type
    if "text_prompt" in body:
        dict_prompt: dict[str, Any] = {"prompt": body["text_prompt"]}
    elif "token_ids_prompt" in body:
        dict_prompt = {"prompt_token_ids": body["token_ids_prompt"]}
    elif "prompt" in body:
        # Convenience: accept "prompt" as alias for "text_prompt"
        dict_prompt = {"prompt": body["prompt"]}
    else:
        raise ValueError("Either text_prompt, token_ids_prompt, or prompt must be set")

    tok_prompt = await renderer.tokenize_prompt_async(dict_prompt, tok_params)

    prompt_token_ids = tok_prompt.get("prompt_token_ids", [])
    return {
        "prompt_token_ids": prompt_token_ids,
        "num_tokens": len(prompt_token_ids),
    }


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
            logger.exception("RenderChat failed")
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
            logger.exception("RenderCompletion failed")
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
# HTTP App
# =====================


def build_http_app(renderer) -> FastAPI:
    """Build a minimal FastAPI app for the render server."""
    app = FastAPI(
        title="vLLM Render Server",
        version=VLLM_VERSION,
    )

    @app.get("/health")
    async def health():
        return {"healthy": True, "message": "OK"}

    @app.post("/render/chat")
    async def render_chat(request: dict):
        try:
            result = await _render_chat(renderer, request)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"RenderChat failed: {e}"},
            )

    @app.post("/render/completion")
    async def render_completion(request: dict):
        try:
            result = await _render_completion(renderer, request)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"RenderCompletion failed: {e}"},
            )

    return app


# =====================
# Server entrypoint
# =====================


async def serve_render(args: argparse.Namespace):
    """Start the gRPC or HTTP render server."""
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
        max_model_len=getattr(args, "max_model_len", None),  # type: ignore[arg-type]
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

    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 50052)
    server_type = getattr(args, "server", "grpc")

    if server_type == "grpc":
        await _serve_grpc(renderer, host, port, start_time)
    else:
        await _serve_http(renderer, host, port)


async def _serve_grpc(renderer, host: str, port: int, start_time: float):
    """Run the gRPC render server."""
    grpc_server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    servicer = RenderServicer(renderer, start_time)
    render_pb2_grpc.add_RenderServiceServicer_to_server(servicer, grpc_server)

    service_names = (
        render_pb2.DESCRIPTOR.services_by_name["RenderService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, grpc_server)

    address = f"{host}:{port}"
    grpc_server.add_insecure_port(address)
    await grpc_server.start()
    logger.info("gRPC server started on %s", address)

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
        await grpc_server.stop(grace=5.0)
        logger.info("Shutdown complete")


async def _serve_http(renderer, host: str, port: int):
    """Run the HTTP render server."""
    http_app = build_http_app(renderer)
    uvicorn_config = uvicorn.Config(
        http_app,
        host=host,
        port=port,
        log_level="info",
    )
    http_server = uvicorn.Server(uvicorn_config)

    logger.info("HTTP server started on %s:%d", host, port)
    await http_server.serve()
