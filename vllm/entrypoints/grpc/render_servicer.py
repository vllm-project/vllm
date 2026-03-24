# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
gRPC servicer for GPU-less render serving.

Implements the VllmRender service with management and rendering RPCs.
"""

import time

import grpc
from smg_grpc_proto import vllm_engine_pb2  # type: ignore[import-untyped]
from starlette.datastructures import State

from vllm.entrypoints.grpc import vllm_render_pb2  # type: ignore[attr-defined]
from vllm.entrypoints.grpc.field_transforms import FIELD_TRANSFORMS
from vllm.entrypoints.grpc.proto_utils import from_proto, pydantic_to_proto
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse


class RenderGrpcServicer:
    """gRPC servicer for GPU-less render serving.

    Implements the VllmRender service with management RPCs
    (HealthCheck, GetModelInfo, GetServerInfo) and rendering RPCs
    (RenderChat, RenderCompletion).
    """

    def __init__(self, state: State, start_time: float):
        self.state = state
        self.start_time = start_time

    async def HealthCheck(self, request, context):
        return vllm_engine_pb2.HealthCheckResponse(healthy=True, message="Healthy")

    async def GetModelInfo(self, request, context):
        model_config = self.state.vllm_config.model_config
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
        if self.state.openai_serving_render is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "RenderChat is not configured on this server.",
            )
            return

        try:
            pydantic_request = from_proto(
                request, ChatCompletionRequest, FIELD_TRANSFORMS
            )

            result = await self.state.openai_serving_render.render_chat_request(
                pydantic_request
            )

            if isinstance(result, ErrorResponse):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    result.error.message,
                )
                return

            return vllm_render_pb2.RenderChatResponse(
                generate_request=pydantic_to_proto(
                    result, vllm_render_pb2.GenerateRequestProto
                ),
            )
        except grpc.aio.AbortError:
            raise
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RenderCompletion(self, request, context):
        if self.state.openai_serving_render is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "RenderCompletion is not configured on this server.",
            )
            return

        try:
            pydantic_request = from_proto(request, CompletionRequest, FIELD_TRANSFORMS)

            result = await self.state.openai_serving_render.render_completion_request(
                pydantic_request
            )

            if isinstance(result, ErrorResponse):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    result.error.message,
                )
                return

            return vllm_render_pb2.RenderCompletionResponse(
                generate_requests=[
                    pydantic_to_proto(gr, vllm_render_pb2.GenerateRequestProto)
                    for gr in result
                ],
            )
        except grpc.aio.AbortError:
            raise
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
