# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Protocol conversion between gRPC protobuf and vLLM Pydantic models."""

import json

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest as PydanticChatCompletionRequest,
    ChatCompletionResponse as PydanticChatCompletionResponse,
    ChatCompletionStreamResponse as PydanticChatCompletionStreamResponse,
    ChatMessage as PydanticChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    UsageInfo,
)


def grpc_to_pydantic_request(
    grpc_request,
) -> PydanticChatCompletionRequest:
    """
    将 gRPC protobuf 请求转换为 vLLM Pydantic 请求。
    """
    # 转换 messages
    messages = []
    for msg in grpc_request.messages:
        message_dict: dict[str, str | None] = {"role": msg.role}
        if msg.HasField("content"):
            message_dict["content"] = msg.content
        else:
            message_dict["content"] = None
        messages.append(message_dict)

    # 构建 Pydantic 请求
    request_dict: dict[str, any] = {
        "model": grpc_request.model,
        "messages": messages,
    }

    # 添加可选字段
    if grpc_request.HasField("temperature"):
        request_dict["temperature"] = grpc_request.temperature
    if grpc_request.HasField("top_p"):
        request_dict["top_p"] = grpc_request.top_p
    if grpc_request.HasField("max_tokens"):
        request_dict["max_tokens"] = grpc_request.max_tokens
    if grpc_request.HasField("max_completion_tokens"):
        request_dict["max_completion_tokens"] = grpc_request.max_completion_tokens
    if grpc_request.HasField("stream"):
        request_dict["stream"] = grpc_request.stream
    if grpc_request.stop:
        request_dict["stop"] = list(grpc_request.stop)
    if grpc_request.HasField("seed"):
        request_dict["seed"] = grpc_request.seed
    if grpc_request.HasField("frequency_penalty"):
        request_dict["frequency_penalty"] = grpc_request.frequency_penalty
    if grpc_request.HasField("presence_penalty"):
        request_dict["presence_penalty"] = grpc_request.presence_penalty
    if grpc_request.HasField("top_k"):
        request_dict["top_k"] = grpc_request.top_k
    if grpc_request.HasField("min_p"):
        request_dict["min_p"] = grpc_request.min_p
    if grpc_request.HasField("repetition_penalty"):
        request_dict["repetition_penalty"] = grpc_request.repetition_penalty
    if grpc_request.HasField("n"):
        request_dict["n"] = grpc_request.n
    if grpc_request.HasField("user"):
        request_dict["user"] = grpc_request.user

    return PydanticChatCompletionRequest(**request_dict)


def pydantic_to_grpc_response(
    pydantic_response: PydanticChatCompletionResponse,
):
    """
    将 vLLM Pydantic 响应转换为 gRPC protobuf 响应。
    """
    from vllm.entrypoints.grpc.proto import openai_pb2

    # 转换 choices
    choices = []
    for choice in pydantic_response.choices:
        grpc_choice = openai_pb2.Choice(
            index=choice.index,
            message=openai_pb2.ChatMessage(
                role=choice.message.role,
                content=choice.message.content or "",
            ),
            finish_reason=choice.finish_reason or "stop",
        )
        choices.append(grpc_choice)

    # 转换 usage
    usage = openai_pb2.Usage(
        prompt_tokens=pydantic_response.usage.prompt_tokens,
        completion_tokens=pydantic_response.usage.completion_tokens or 0,
        total_tokens=pydantic_response.usage.total_tokens,
    )

    return openai_pb2.ChatCompletionResponse(
        id=pydantic_response.id,
        object=pydantic_response.object,
        created=pydantic_response.created,
        model=pydantic_response.model,
        choices=choices,
        usage=usage,
    )


def parse_sse_chunk(sse_str: str) -> dict | None:
    """
    解析 SSE 格式的 chunk。

    SSE 格式示例：
    data: {"id":"chatcmpl-123","object":"chat.completion.chunk",...}
    data: [DONE]
    """
    if not sse_str.startswith("data: "):
        return None

    json_str = sse_str[6:].strip()
    if json_str == "[DONE]":
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def pydantic_stream_to_grpc_stream_response(
    pydantic_chunk: PydanticChatCompletionStreamResponse,
):
    """
    将流式 chunk 转换为 gRPC 响应。
    """
    from vllm.entrypoints.grpc.proto import openai_pb2

    # 转换 choices
    choices = []
    for choice in pydantic_chunk.choices:
        delta = openai_pb2.Delta()
        if choice.delta.role is not None:
            delta.role = choice.delta.role
        if choice.delta.content is not None:
            delta.content = choice.delta.content

        grpc_choice = openai_pb2.StreamChoice(
            index=choice.index,
            delta=delta,
        )
        if choice.finish_reason is not None:
            grpc_choice.finish_reason = choice.finish_reason
        choices.append(grpc_choice)

    # 转换 usage（如果存在）
    usage = None
    if pydantic_chunk.usage is not None:
        usage = openai_pb2.Usage(
            prompt_tokens=pydantic_chunk.usage.prompt_tokens,
            completion_tokens=pydantic_chunk.usage.completion_tokens or 0,
            total_tokens=pydantic_chunk.usage.total_tokens,
        )

    grpc_response = openai_pb2.ChatCompletionStreamResponse(
        id=pydantic_chunk.id,
        object=pydantic_chunk.object,
        created=pydantic_chunk.created,
        model=pydantic_chunk.model,
        choices=choices,
    )
    if usage is not None:
        grpc_response.usage.CopyFrom(usage)

    return grpc_response
