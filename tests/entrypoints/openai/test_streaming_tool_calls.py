# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir": {"type": "string", "description": "Directory path"}
                },
                "required": ["dir"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def server():
    """Start vLLM server with tool calling enabled."""
    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--max-model-len",
        "2048",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize(
    "stream_interval", [1, 8, 9, 10, 18, 19, 20, 100, 1000, 1_000_000]
)
def test_streaming_tool_calls_with_different_intervals(stream_interval: int):
    """
    Test that streaming tool calls work correctly with different stream intervals.

    Regression test for issue #31501 where tool calls failed when stream_interval > 1.
    """
    # Restart server with the specific stream_interval for this test
    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--max-model-len",
        "2048",
        "--stream-interval",
        str(stream_interval),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as test_server:
        client = test_server.get_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Use the list_directory tool when asked about files.",
                },
                {"role": "user", "content": "List files in the src folder"},
            ],
            tools=TOOLS,
            stream=True,
        )

        # Accumulate streamed tool call
        accumulated_args = ""
        tool_name = None

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function.name:
                        tool_name = tc.function.name
                    if tc.function.arguments:
                        accumulated_args += tc.function.arguments

        # Verify tool call was correctly accumulated
        assert tool_name == "list_directory", (
            f"Expected tool name 'list_directory', got {tool_name}"
        )
        assert accumulated_args, (
            f"Expected non-empty arguments, got {accumulated_args!r}"
        )

        # Verify the arguments contain the expected directory
        assert '{"dir": "/src"}' in accumulated_args, (
            f'Expected {{"dir": "/src"}} in arguments, got {accumulated_args!r}'
        )


def test_streaming_tool_calls_basic(server):
    """Basic test that streaming tool calls work with default settings."""
    client = server.get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Use the list_directory tool when asked about files.",
            },
            {"role": "user", "content": "List files in the src folder"},
        ],
        tools=TOOLS,
        stream=True,
    )

    # Accumulate streamed tool call
    accumulated_args = ""
    tool_name = None

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function.name:
                    tool_name = tc.function.name
                if tc.function.arguments:
                    accumulated_args += tc.function.arguments

    # Verify results
    assert tool_name == "list_directory"
    assert accumulated_args
    assert '{"dir": "/src"}' in accumulated_args
