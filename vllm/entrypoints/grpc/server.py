# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""gRPC server implementation for vLLM OpenAI-style API."""

import json
from argparse import Namespace
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
from grpc import aio

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.entrypoints.grpc.proto import openai_pb2, openai_pb2_grpc

logger = init_logger(__name__)


class OpenAIServicer:
    """
    gRPC servicer 实现。

    设计理念：
    - 薄转换层：只做 protobuf ↔ Pydantic 转换
    - 复用逻辑：调用 OpenAIServingChat 处理业务逻辑
    - 错误处理：统一的异常处理
    """

    def __init__(
        self,
        engine_client: EngineClient,
        serving_chat: OpenAIServingChat,
        args: Namespace,
    ):
        self.engine_client = engine_client
        self.serving_chat = serving_chat
        self.args = args

    async def ChatCompletion(
        self,
        request,
        context: grpc.aio.ServicerContext,
    ):
        """
        非流式 chat completion。

        流程：
        1. protobuf → Pydantic
        2. 调用 OpenAIServingChat.create_chat_completion()
        3. Pydantic → protobuf
        """
        from vllm.entrypoints.grpc.converters import (
            grpc_to_pydantic_request,
            pydantic_to_grpc_response,
        )
        from vllm.entrypoints.grpc.proto import openai_pb2

        try:
            # Step 1: 转换请求
            pydantic_request = grpc_to_pydantic_request(request)
            # 确保非流式
            pydantic_request.stream = False

            # Step 2: 调用现有逻辑
            result = await self.serving_chat.create_chat_completion(
                pydantic_request,
                raw_request=None,  # gRPC 没有 FastAPI Request 对象
            )

            # Step 3: 转换响应
            if isinstance(result, ErrorResponse):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(result.error.message)
                raise grpc.RpcError(
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    details=result.error.message,
                )

            if isinstance(result, ChatCompletionResponse):
                grpc_response = pydantic_to_grpc_response(result)
                return grpc_response
            else:
                # 如果返回 generator，需要收集所有 chunks
                # （非流式请求不应该走到这里）
                raise ValueError("Unexpected generator for non-streaming request")

        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception(f"Error in ChatCompletion: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise grpc.RpcError(code=grpc.StatusCode.INTERNAL, details=str(e)) from e

    async def ChatCompletionStream(
        self,
        request,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator:
        """
        流式 chat completion。

        流程：
        1. protobuf → Pydantic
        2. 调用 OpenAIServingChat.create_chat_completion()（stream=True）
        3. 解析 SSE 格式的响应
        4. 逐个 yield Pydantic → protobuf
        """
        from vllm.entrypoints.grpc.converters import (
            grpc_to_pydantic_request,
            parse_sse_chunk,
            pydantic_stream_to_grpc_stream_response,
        )

        try:
            # Step 1: 转换请求（强制 stream=True）
            pydantic_request = grpc_to_pydantic_request(request)
            pydantic_request.stream = True

            # Step 2: 获取流式 generator
            result_generator = await self.serving_chat.create_chat_completion(
                pydantic_request,
                raw_request=None,
            )

            # Step 3: 解析 SSE 格式并逐个 yield
            if isinstance(result_generator, ErrorResponse):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(result_generator.error.message)
                raise grpc.RpcError(
                    code=grpc.StatusCode.INVALID_ARGUMENT,
                    details=result_generator.error.message,
                )

            # result_generator 是 AsyncGenerator[str, None]，返回 SSE 格式字符串
            buffer = ""
            async for sse_chunk in result_generator:
                buffer += sse_chunk

                # 处理完整的 SSE 行
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if not line.strip():
                        continue

                    # 解析 SSE chunk
                    chunk_data = parse_sse_chunk(line)
                    if chunk_data is None:
                        # [DONE] 或其他非数据行
                        continue

                    # 转换为 Pydantic 对象
                    try:
                        pydantic_chunk = ChatCompletionStreamResponse(**chunk_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse SSE chunk: {e}, data: {chunk_data}")
                        continue

                    # 转换为 gRPC 响应并 yield
                    grpc_chunk = pydantic_stream_to_grpc_stream_response(pydantic_chunk)
                    yield grpc_chunk

        except grpc.RpcError:
            raise
        except Exception as e:
            logger.exception(f"Error in ChatCompletionStream: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise grpc.RpcError(code=grpc.StatusCode.INTERNAL, details=str(e)) from e

    async def HealthCheck(
        self,
        request,
        context: grpc.aio.ServicerContext,
    ):
        """
        健康检查。
        """
        from vllm.entrypoints.grpc.proto import openai_pb2

        try:
            # 检查引擎状态
            healthy = self.engine_client.is_running if hasattr(self.engine_client, "is_running") else True

            return openai_pb2.HealthCheckResponse(
                healthy=healthy,
                message="OK" if healthy else "Engine not running",
            )
        except Exception as e:
            return openai_pb2.HealthCheckResponse(
                healthy=False,
                message=f"Error: {str(e)}",
            )


async def run_grpc_server(
    engine_client: EngineClient,
    serving_chat: OpenAIServingChat,
    args: Namespace,
    host: str = "0.0.0.0",
    port: int = 8033,
) -> None:
    """
    启动 gRPC 服务器。
    """
    from vllm.entrypoints.grpc.proto import openai_pb2_grpc

    # 创建 gRPC server
    server = aio.server()

    # 注册 servicer
    servicer = OpenAIServicer(engine_client, serving_chat, args)
    openai_pb2_grpc.add_OpenAIServiceServicer_to_server(servicer, server)

    # 添加端口
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting gRPC server on {listen_addr}")

    # 启动服务器
    await server.start()

    # 等待终止信号
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(grace=5.0)
