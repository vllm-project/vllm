# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import regex as re

from tests.utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen2-0.5B-Instruct"


@pytest.fixture(scope="module")
def server():
    # Constrain a small public model to the FunctionGemma wire format. This
    # isolates serving/parser behavior from model tool-selection accuracy.
    args = [
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "functiongemma",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.1",
    ]
    with RemoteOpenAIServer(MODEL, args, max_wait_seconds=480) as server:
        yield server


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "value", ["123", "true", "null", "[1,2]", '{"x":1}', '"quoted"']
)
def test_escape_delimited_strings_over_http(server, stream, value):
    output = (
        "<start_function_call>call:lookup{value:<escape>"
        f"{value}<escape>}}<end_function_call>"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
    ]
    with server.get_client() as client:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Call lookup."}],
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=128,
            stream=stream,
            extra_body={"structured_outputs": {"regex": re.escape(output)}},
        )
        if stream:
            arguments = ""
            names = []
            for chunk in response:
                for call in chunk.choices[0].delta.tool_calls or []:
                    assert call.index == 0
                    if call.function.name:
                        names.append(call.function.name)
                    arguments += call.function.arguments or ""
            assert names == ["lookup"]
        else:
            calls = response.choices[0].message.tool_calls
            assert calls is not None and len(calls) == 1
            assert calls[0].function.name == "lookup"
            arguments = calls[0].function.arguments
    assert json.loads(arguments) == {"value": value}
