# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
gRPC servicer for GPU-less render serving.

Implements the VllmRender service with management and rendering RPCs.
"""

import time

import grpc
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.grpc import vllm_engine_pb2, vllm_render_pb2  # type: ignore[attr-defined]
from vllm.grpc.field_transforms import FIELD_TRANSFORMS
from vllm.grpc.proto_utils import from_proto, to_proto


class RenderGrpcServicer:
    """gRPC servicer for GPU-less render serving.

    Implements the VllmRender service with management RPCs
    (HealthCheck, GetModelInfo, GetServerInfo) and rendering RPCs
    (RenderChat, RenderCompletion).
    """

    def __init__(self, state: State, start_time: float):
        self.state = state
        self.engine_client: EngineClient = state.engine_client
        self.start_time = start_time

    async def HealthCheck(self, request, context):
        return vllm_engine_pb2.HealthCheckResponse(healthy=True, message="Healthy")

    async def GetModelInfo(self, request, context):
        model_config = self.engine_client.model_config
        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=model_config.model,
            is_generation=model_config.runner_type == "generate",
            max_context_length=model_config.max_model_len,
            vocab_size=model_config.get_vocab_size(),
            supports_vision=model_config.is_multimodal_model,
        )

    async def GetServerInfo(self, request, context):
        return vllm_engine_pb2.GetServerInfoResponse(
            last_receive_timestamp=time.time(),
            uptime_seconds=time.time() - self.start_time,
            server_type="vllm-render-grpc",
        )

    async def RenderChat(self, request, context):
        if self.state.openai_serving_chat is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "RenderChat is not configured on this server.",
            )
            return

        try:
            pydantic_request = from_proto(
                request, ChatCompletionRequest, FIELD_TRANSFORMS
            )

            result = await self.state.openai_serving_chat.render_chat_request(
                pydantic_request
            )

            if isinstance(result, ErrorResponse):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    result.error.message,
                )
                return

            conversation, prompts = result
            # Serialize Pydantic models (Harmony) or pass through dicts
            conversation = [
                getattr(msg, "model_dump", lambda m=msg: m)() for msg in conversation
            ]
            return to_proto(
                (conversation, prompts),
                vllm_render_pb2.RenderChatResponse,
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RenderCompletion(self, request, context):
        if self.state.openai_serving_completion is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "RenderCompletion is not configured on this server.",
            )
            return

        try:
            pydantic_request = from_proto(request, CompletionRequest, FIELD_TRANSFORMS)

            result = (
                await self.state.openai_serving_completion.render_completion_request(
                    pydantic_request
                )
            )

            if isinstance(result, ErrorResponse):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    result.error.message,
                )
                return

            return to_proto((result,), vllm_render_pb2.RenderCompletionResponse)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
