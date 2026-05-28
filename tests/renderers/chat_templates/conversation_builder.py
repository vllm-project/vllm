# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class _Sentinel(Enum):
    AUTO = auto()
    MISSING = auto()


AUTO = _Sentinel.AUTO
MISSING = _Sentinel.MISSING


@dataclass
class _Context:
    values: dict[str, int] = field(default_factory=dict)
    auto_tools: list[Tool] | None = None

    def __post_init__(self):
        self.auto_tool_index = 0

    def next_auto_value(self, base: str) -> str:
        count = self.values.get(base, 0)
        self.values[base] = count + 1
        return f"{base}_{count}"

    def next_auto_tool(self) -> Tool:
        if self.auto_tools is None:
            return Tool(
                name=self.next_auto_value("TOOL"),
                description=self.next_auto_value("TOOL_DESCRIPTION"),
            )
        elif self.auto_tools:
            tool = self.auto_tools[self.auto_tool_index]
            self.auto_tool_index = (self.auto_tool_index + 1) % len(self.auto_tools)
            return tool
        else:
            raise ValueError("auto_tools is empty but AUTO tool is requested")

    def render(self, value: Any, auto_label: str) -> Any:
        return self.next_auto_value(auto_label) if value is AUTO else value


@dataclass(frozen=True)
class Message:
    role: str
    content: Any = AUTO

    def render(self, context: _Context) -> list[dict[str, Any]]:
        message = {"role": self.role}
        if self.content is not MISSING:
            message["content"] = context.render(
                self.content, f"{self.role.upper()}_MESSAGE"
            )
        return [message]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    # TODO: Support tool parameters
    # TODO: Support richer tool responses
    result: Any = AUTO

    def render_tool_def(self, context: _Context) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {},
            "required": [],
        }

    def render_tool_call(self, tool_call_id: str, context: _Context) -> dict[str, Any]:
        return {
            "id": tool_call_id,
            "function": {
                "name": self.name,
                "arguments": {},
            },
        }

    def render_tool_response(
        self, tool_call_id: str, context: _Context
    ) -> dict[str, Any]:
        response = {
            "role": "tool",
            "tool_call_id": tool_call_id,
        }
        if self.result is not MISSING:
            response["content"] = context.render(self.result, "TOOL_RESULT")
        return response


@dataclass(frozen=True)
class Assistant:
    content: Any = AUTO
    reasoning: Any = MISSING
    reasoning_content: Any = MISSING
    tool_uses: Sequence[Any | tuple[Any, Any]] = ()

    def render(self, context: _Context) -> list[dict[str, Any]]:
        assistant: dict[str, Any] = {"role": "assistant"}
        for field_name, value, auto_label in (
            ("content", self.content, "ASSISTANT_MESSAGE"),
            ("reasoning", self.reasoning, "ASSISTANT_REASONING"),
            (
                "reasoning_content",
                self.reasoning_content,
                "ASSISTANT_REASONING_CONTENT",
            ),
        ):
            if value is not MISSING:
                assistant[field_name] = context.render(value, auto_label)

        tool_calls: list[dict[str, Any]] = []
        tool_responses: list[dict[str, Any]] = []
        for tool_use in self.tool_uses:
            # Normalize tool_use to a (call, response) tuple
            if not isinstance(tool_use, tuple):
                call, response = tool_use, tool_use
            else:
                call, response = tool_use

            if call is AUTO:
                call = context.next_auto_tool()
                if response is AUTO:
                    response = call
            elif response is AUTO:
                response = context.next_auto_tool()

            # Render tool call and response
            tool_call_id = None
            if isinstance(call, Tool):
                tool_call_id = context.next_auto_value("TOOL_CALL_ID")
                tool_calls.append(call.render_tool_call(tool_call_id, context))
            else:
                # Preserve MISSING calls
                tool_calls.append(call)

            if isinstance(response, Tool):
                if tool_call_id is None:
                    tool_call_id = context.next_auto_value("TOOL_CALL_ID")
                tool_responses.append(
                    response.render_tool_response(tool_call_id, context)
                )
            elif response is not MISSING:
                tool_responses.append(response)

        if tool_calls:
            # Filter out MISSING calls here so that MISSING tool call still
            # result in a "tool_calls" entry in the assistant message.
            assistant["tool_calls"] = list(
                tool_call for tool_call in tool_calls if tool_call is not MISSING
            )
        return [assistant, *tool_responses]


def create_conversation(
    *parts: Any, auto_tool_restrictions: list[Tool] | None = None
) -> list[dict[str, Any]]:
    context: _Context = _Context(auto_tools=auto_tool_restrictions)
    conversation: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, (Message, Assistant)):
            conversation.extend(part.render(context))
        else:
            conversation.append(part)
    return conversation


SYSTEM = Message("system")
DEVELOPER = Message("developer")
USER = Message("user")
