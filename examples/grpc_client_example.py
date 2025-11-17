#!/usr/bin/env python3
"""
示例 gRPC 客户端，用于测试 vLLM gRPC API。

使用方法：
    python examples/grpc_client_example.py
"""

import asyncio

import grpc
from grpc import aio

from vllm.entrypoints.grpc.proto import openai_pb2, openai_pb2_grpc


async def test_non_streaming():
    """测试非流式请求"""
    async with aio.insecure_channel("localhost:8033") as channel:
        stub = openai_pb2_grpc.OpenAIServiceStub(channel)

        request = openai_pb2.ChatCompletionRequest(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[
                openai_pb2.ChatMessage(
                    role="user",
                    content="What is the capital of France?",
                )
            ],
            temperature=0.7,
            max_tokens=100,
        )

        print("发送非流式请求...")
        response = await stub.ChatCompletion(request)

        print(f"响应 ID: {response.id}")
        print(f"模型: {response.model}")
        print(f"内容: {response.choices[0].message.content}")
        print(f"完成原因: {response.choices[0].finish_reason}")
        print(f"使用情况: prompt_tokens={response.usage.prompt_tokens}, "
              f"completion_tokens={response.usage.completion_tokens}, "
              f"total_tokens={response.usage.total_tokens}")


async def test_streaming():
    """测试流式请求"""
    async with aio.insecure_channel("localhost:8033") as channel:
        stub = openai_pb2_grpc.OpenAIServiceStub(channel)

        request = openai_pb2.ChatCompletionRequest(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[
                openai_pb2.ChatMessage(
                    role="user",
                    content="Write a short poem about AI",
                )
            ],
            temperature=0.8,
            max_tokens=200,
            stream=True,
        )

        print("\n流式响应:")
        async for chunk in stub.ChatCompletionStream(request):
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n\n流式响应完成!")


async def test_health_check():
    """测试健康检查"""
    async with aio.insecure_channel("localhost:8033") as channel:
        stub = openai_pb2_grpc.OpenAIServiceStub(channel)

        request = openai_pb2.HealthCheckRequest()
        response = await stub.HealthCheck(request)

        print(f"\n健康检查: healthy={response.healthy}, message={response.message}")


async def main():
    """主函数"""
    try:
        print("=== 健康检查测试 ===")
        await test_health_check()

        print("\n=== 非流式测试 ===")
        await test_non_streaming()

        print("\n=== 流式测试 ===")
        await test_streaming()
    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
