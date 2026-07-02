# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
gRPC servicer for GPU-less render serving.

Implements the VllmRender service with management and rendering RPCs.
"""

import logging
import time

import grpc
from smg_grpc_proto import vllm_engine_pb2  # type: ignore[import-untyped]
from starlette.datastructures import State

from vllm.entrypoints.grpc import (  # type: ignore[attr-defined]
    vllm_render_pb2,
    vllm_render_pb2_grpc,
)
from vllm.entrypoints.grpc.field_transforms import FIELD_TRANSFORMS
from vllm.entrypoints.grpc.proto_utils import from_proto, pydantic_to_proto
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse

logger = logging.getLogger(__name__)

_HTTP_TO_GRPC = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    404: grpc.StatusCode.NOT_FOUND,
    409: grpc.StatusCode.ALREADY_EXISTS,
    422: grpc.StatusCode.INVALID_ARGUMENT,
    429: grpc.StatusCode.RESOURCE_EXHAUSTED,
    503: grpc.StatusCode.UNAVAILABLE,
}


def _http_to_grpc_status(http_code: int) -> grpc.StatusCode:
    return _HTTP_TO_GRPC.get(http_code, grpc.StatusCode.INTERNAL)


class RenderGrpcServicer(vllm_render_pb2_grpc.VllmRenderServicer):
    """gRPC servicer for GPU-less render serving.

    Implements the VllmRender service with management RPCs
    (HealthCheck, GetModelInfo, GetServerInfo) and rendering RPCs
    (RenderChat, RenderCompletion).
    """

    def __init__(self, state: State, start_time: float):
        self.state = state
        self.start_time = start_time

    async def HealthCheck(self, request, context):
        healthy = self.state.openai_serving_render is not None
        msg = "Healthy" if healthy else "Render service not initialized"
        return vllm_engine_pb2.HealthCheckResponse(healthy=healthy, message=msg)

    async def GetModelInfo(self, request, context):
        model_config = self.state.vllm_config.model_config
        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=model_config.model,
            is_generation=model_config.runner_type == "generate",
            max_context_length=model_config.max_model_len,
            vocab_size=model_config.get_vocab_size(),
            supports_vision=model_config.is_multimodal_model,
            served_model_name=model_config.served_model_name or model_config.model,
        )

    async def GetServerInfo(self, request, context):
        return vllm_engine_pb2.GetServerInfoResponse(
            active_requests=0,
            is_paused=False,
            last_receive_timestamp=time.time(),
            uptime_seconds=time.time() - self.start_time,
            server_type="vllm-render-grpc",
            kv_connector="",
            kv_role="",
        )

    async def RenderChat(self, request, context):
        if self.state.openai_serving_render is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "RenderChat is not configured on this server.",
            )

        try:
            pydantic_request = from_proto(
                request, ChatCompletionRequest, FIELD_TRANSFORMS
            )
            result = await self.state.openai_serving_render.render_chat_request(
                pydantic_request
            )
            if isinstance(result, ErrorResponse):
                await context.abort(
                    _http_to_grpc_status(result.error.code), result.error.message
                )
        except grpc.aio.AbortError:
            raise
        except (ValueError, TypeError) as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("RenderChat failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        try:
            return vllm_render_pb2.RenderChatResponse(
                generate_request=pydantic_to_proto(
                    result, vllm_render_pb2.GenerateRequestProto
                ),
            )
        except Exception as e:
            logger.exception("RenderChat response serialization failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RenderCompletion(self, request, context):
        if self.state.openai_serving_render is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "RenderCompletion is not configured on this server.",
            )

        try:
            pydantic_request = from_proto(request, CompletionRequest, FIELD_TRANSFORMS)
            result = await self.state.openai_serving_render.render_completion_request(
                pydantic_request
            )
            if isinstance(result, ErrorResponse):
                await context.abort(
                    _http_to_grpc_status(result.error.code), result.error.message
                )
        except grpc.aio.AbortError:
            raise
        except (ValueError, TypeError) as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("RenderCompletion failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        try:
            return vllm_render_pb2.RenderCompletionResponse(
                generate_requests=[
                    pydantic_to_proto(gr, vllm_render_pb2.GenerateRequestProto)
                    for gr in result
                ],
            )
        except Exception as e:
            logger.exception("RenderCompletion response serialization failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
