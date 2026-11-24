# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.utils import RemoteOpenAIServer

MODEL = "google/gemma-4-E2B-it"
REVISION = "4adf15748de9e1a0097b6274cee9c15836c41c37"
EXPECTED = {"sample": {"count": 7, "flag": False, "empty": None, "label": "7"}}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_values",
            "description": "Record a sample of JSON values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample": {"type": "object", "additionalProperties": True},
                },
                "required": ["sample"],
                "additionalProperties": False,
            },
        },
    }
]


@pytest.fixture(scope="module")
def gemma4_server():
    args = [
        "--revision",
        REVISION,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        "--reasoning-parser",
        "gemma4",
        "--language-model-only",
        "--max-model-len",
        "1024",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL, args) as server:
        yield server


@pytest.mark.parametrize("stream", [False, True])
def test_gemma4_bare_tool_argument_types(gemma4_server, stream):
    """Preserve the types of nested tool values with no declared leaf types."""
    with gemma4_server.get_client() as client:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Call record_values once with exactly these arguments: "
                    + json.dumps(EXPECTED),
                }
            ],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            seed=42,
            max_completion_tokens=192,
            stream=stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if stream:
            names = []
            arguments = []
            for chunk in response:
                for choice in chunk.choices:
                    for call in choice.delta.tool_calls or []:
                        assert call.index == 0
                        if call.function:
                            names.append(call.function.name or "")
                            arguments.append(call.function.arguments or "")
            assert "".join(names) == "record_values"
            received = "".join(arguments)
        else:
            calls = response.choices[0].message.tool_calls
            assert calls and len(calls) == 1
            assert calls[0].function.name == "record_values"
            received = calls[0].function.arguments

    assert json.dumps(json.loads(received), sort_keys=True) == json.dumps(
        EXPECTED, sort_keys=True
    )
