# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proto ↔ Python conversion utilities for the render gRPC service.

These helpers use proto field descriptors for automatic field discovery,
so that adding new fields to render.proto + the corresponding Python types
requires no changes here.
"""

from typing import TypeVar

from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
)
from vllm.grpc import render_pb2

T = TypeVar("T")

# =====================
# Params (proto → Python dataclass)
# =====================


def params_from_proto(proto, defaults: T, cls: type[T]) -> T:
    """Convert a proto params message to its Python dataclass.

    Uses proto field descriptors + HasField() so that new fields added to
    both render.proto and the Python dataclass are picked up automatically
    without changing this function.
    """
    kwargs = {}
    for field in proto.DESCRIPTOR.fields:
        if proto.HasField(field.name):
            val = getattr(proto, field.name)
            # google.protobuf.Struct → dict
            if (
                field.message_type
                and field.message_type.full_name == "google.protobuf.Struct"
            ):
                val = dict(val)
            kwargs[field.name] = val
        else:
            kwargs[field.name] = getattr(defaults, field.name)
    return cls(**kwargs)


# =====================
# Messages (proto ↔ ChatCompletionMessageParam / ConversationMessage)
# =====================

# Fields needing special conversion (oneof, repeated nested).
# All other optional scalar fields are handled via reflection.
_MSG_SPECIAL_FIELDS = frozenset(
    {
        "role",
        "text_content",
        "parts",
        "tool_calls",
    }
)


def messages_from_proto(
    messages: list[render_pb2.ChatMessage],
) -> list[ChatCompletionMessageParam]:
    """Convert ChatMessage protos to ChatCompletionMessageParam dicts.

    Simple optional scalar fields (name, tool_call_id, reasoning, …) are
    picked up automatically via proto field descriptors so that new fields
    added to both render.proto and ChatCompletionMessageParam work without
    changing this function.
    """
    result = []
    for msg in messages:
        d: dict = {"role": msg.role}

        # content (oneof)
        content_type = msg.WhichOneof("content_type")
        if content_type == "text_content":
            d["content"] = msg.text_content
        elif content_type == "parts":
            parts = []
            for part in msg.parts.parts:
                p: dict = {"type": part.type}
                if part.HasField("text"):
                    p["text"] = part.text
                if part.HasField("image_url"):
                    p["image_url"] = {"url": part.image_url}
                parts.append(p)
            d["content"] = parts

        # tool_calls (repeated nested)
        if msg.tool_calls:
            d["tool_calls"] = [
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

        # Simple optional scalars (name, tool_call_id, reasoning, …)
        for field in msg.DESCRIPTOR.fields:
            if field.name in _MSG_SPECIAL_FIELDS:
                continue
            if field.has_presence and msg.HasField(field.name):
                d[field.name] = getattr(msg, field.name)

        result.append(d)
    return result


def conversation_to_proto(
    conversation: list[ConversationMessage],
) -> list[render_pb2.ConversationMessage]:
    """Convert ConversationMessage dicts to proto messages.

    Simple optional scalar fields are picked up automatically via proto
    field descriptors, matching messages_from_proto.
    """
    result = []
    for conv in conversation:
        proto_msg = render_pb2.ConversationMessage(role=conv["role"])

        # content (oneof)
        content = conv.get("content")
        if isinstance(content, str):
            proto_msg.text_content = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                p = render_pb2.ContentPart(type=part.get("type", "text"))
                if "text" in part:
                    p.text = part["text"]
                if "image_url" in part:
                    url = part["image_url"]
                    if isinstance(url, dict):
                        p.image_url = url.get("url", "")
                    else:
                        p.image_url = str(url)
                parts.append(p)
            proto_msg.parts.CopyFrom(render_pb2.ContentPartList(parts=parts))

        # tool_calls (repeated nested)
        if "tool_calls" in conv:
            for tc in conv["tool_calls"]:
                func = tc.get("function", {})
                proto_msg.tool_calls.append(
                    render_pb2.ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=render_pb2.ToolCallFunction(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", ""),
                        ),
                    )
                )

        # Simple optional scalars (name, tool_call_id, reasoning, …)
        for field in proto_msg.DESCRIPTOR.fields:
            if field.name in _MSG_SPECIAL_FIELDS:
                continue
            if field.has_presence and field.name in conv:
                setattr(proto_msg, field.name, conv[field.name])

        result.append(proto_msg)
    return result
