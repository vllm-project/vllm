# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Demo and regression tests for conversation builder."""

import pytest

from tests.renderers.chat_templates.conversation_builder import (
    AUTO,
    DEVELOPER,
    MISSING,
    SYSTEM,
    USER,
    Assistant,
    Message,
    Tool,
    create_conversation,
)

# Simple conversation, Assistant() by default comes with only content
SIMPLE = [SYSTEM, DEVELOPER, USER, Assistant()]
# To add reasoning, set reasoning=AUTO or reasoning_content=AUTO
REASONING = [Assistant(reasoning=AUTO, reasoning_content=AUTO)]
# For a tool call + tool response, set tool_uses=[AUTO]
TOOL_USE = [Assistant(content=MISSING, tool_uses=[AUTO])]
# For parallel tool calls and tool responses, add multiple items to tool_uses
PARALLEL_TOOL_USE = [Assistant(content=MISSING, tool_uses=[AUTO, AUTO])]
# For tool call without response, add (call, response) items to tool_uses
TOOL_CALL = [Assistant(content=MISSING, tool_uses=[(AUTO, MISSING)])]
# Works for parallel tool calls too
PARALLEL_TOOL_CALL_PARTIAL_RESPONSE = [
    Assistant(content=MISSING, tool_uses=[(AUTO, MISSING), (AUTO, AUTO)])
]
# Anywhere AUTO / MISSING is used, you can also pass in a custom value
# Custom messages are also supported.
CUSTOM_MESSAGE = [
    Message("CUSTOM_ROLE", "CUSTOM_MESSAGE"),
    Assistant(
        reasoning="CUSTOM_ASSISTANT_REASONING", content="CUSTOM_ASSISTANT_MESSAGE"
    ),
    {"CUSTOM_FIELD": "CUSTOM_VALUE"},
]
# tool_uses also accept a Tool object that you can re-use across calls
CUSTOM_TOOL = Tool("CUSTOM_TOOL_NAME", "CUSTOM_TOOL_DESCRIPTION")
CUSTOM_TOOL_CALL_PARTIAL_RESPONSE = [
    Assistant(content=MISSING, tool_uses=[CUSTOM_TOOL, (CUSTOM_TOOL, MISSING)])
]
# tool_uses also accept custom tool call and response as dicts
CUSTOM_TOOL_USE_W_CUSTOM_ID = [
    Assistant(
        content=MISSING,
        tool_uses=[
            (
                {
                    "id": "CUSTOM_ID",
                    "function": {
                        "name": "CUSTOM_TOOL",
                        "arguments": "CUSTOM_ARGUMENTS",
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": "CUSTOM_ID_MISMATCH",
                    "content": "CUSTOM_TOOL_RESULT",
                },
            )
        ],
    )
]


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        pytest.param(
            SIMPLE,
            [
                {"role": "system", "content": "SYSTEM_MESSAGE_0"},
                {"role": "developer", "content": "DEVELOPER_MESSAGE_0"},
                {"role": "user", "content": "USER_MESSAGE_0"},
                {"role": "assistant", "content": "ASSISTANT_MESSAGE_0"},
            ],
            id="simple",
        ),
        pytest.param(
            REASONING,
            [
                {
                    "role": "assistant",
                    "content": "ASSISTANT_MESSAGE_0",
                    "reasoning": "ASSISTANT_REASONING_0",
                    "reasoning_content": "ASSISTANT_REASONING_CONTENT_0",
                }
            ],
            id="reasoning",
        ),
        pytest.param(
            TOOL_USE,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "TOOL_CALL_ID_0",
                            "function": {
                                "name": "TOOL_0",
                                "arguments": {},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "TOOL_CALL_ID_0",
                    "content": "TOOL_RESULT_0",
                },
            ],
            id="tool_use",
        ),
        pytest.param(
            PARALLEL_TOOL_USE,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "TOOL_CALL_ID_0",
                            "function": {
                                "name": "TOOL_0",
                                "arguments": {},
                            },
                        },
                        {
                            "id": "TOOL_CALL_ID_1",
                            "function": {
                                "name": "TOOL_1",
                                "arguments": {},
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "TOOL_CALL_ID_0",
                    "content": "TOOL_RESULT_0",
                },
                {
                    "role": "tool",
                    "tool_call_id": "TOOL_CALL_ID_1",
                    "content": "TOOL_RESULT_1",
                },
            ],
            id="parallel_tool_use",
        ),
        pytest.param(
            TOOL_CALL,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "TOOL_CALL_ID_0",
                            "function": {
                                "name": "TOOL_0",
                                "arguments": {},
                            },
                        }
                    ],
                }
            ],
            id="tool_call",
        ),
        pytest.param(
            PARALLEL_TOOL_CALL_PARTIAL_RESPONSE,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "TOOL_CALL_ID_0",
                            "function": {
                                "name": "TOOL_0",
                                "arguments": {},
                            },
                        },
                        {
                            "id": "TOOL_CALL_ID_1",
                            "function": {
                                "name": "TOOL_1",
                                "arguments": {},
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "TOOL_CALL_ID_1",
                    "content": "TOOL_RESULT_0",
                },
            ],
            id="parallel_tool_call_partial_response",
        ),
        pytest.param(
            CUSTOM_MESSAGE,
            [
                {"role": "CUSTOM_ROLE", "content": "CUSTOM_MESSAGE"},
                {
                    "role": "assistant",
                    "content": "CUSTOM_ASSISTANT_MESSAGE",
                    "reasoning": "CUSTOM_ASSISTANT_REASONING",
                },
                {"CUSTOM_FIELD": "CUSTOM_VALUE"},
            ],
            id="custom_message",
        ),
        pytest.param(
            CUSTOM_TOOL_CALL_PARTIAL_RESPONSE,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "TOOL_CALL_ID_0",
                            "function": {
                                "name": "CUSTOM_TOOL_NAME",
                                "arguments": {},
                            },
                        },
                        {
                            "id": "TOOL_CALL_ID_1",
                            "function": {
                                "name": "CUSTOM_TOOL_NAME",
                                "arguments": {},
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "TOOL_CALL_ID_0",
                    "content": "TOOL_RESULT_0",
                },
            ],
            id="custom_tool_call_partial_response",
        ),
        pytest.param(
            CUSTOM_TOOL_USE_W_CUSTOM_ID,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "CUSTOM_ID",
                            "function": {
                                "name": "CUSTOM_TOOL",
                                "arguments": "CUSTOM_ARGUMENTS",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "CUSTOM_ID_MISMATCH",
                    "content": "CUSTOM_TOOL_RESULT",
                },
            ],
            id="custom_tool_use_w_custom_id",
        ),
    ],
)
def test_create_conversation_cases(parts, expected):
    assert create_conversation(*parts) == expected
