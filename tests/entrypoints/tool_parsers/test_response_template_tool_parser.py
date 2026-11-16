# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from typing import Any

import pytest
import regex as re
from transformers import AutoConfig, AutoTokenizer

from tests.parser.test_chat_parsing import qwen3_template
from tests.utils import RemoteOpenAIServer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    template: dict[str, Any] = copy.deepcopy(qwen3_template)
    template["fields"]["tool_calls"]["content_args"]["merge_duplicates"] = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, response_template=template)
    tokenizer_dir = tmp_path_factory.mktemp("xml-response-template")
    tokenizer.save_pretrained(tokenizer_dir)
    AutoConfig.from_pretrained(MODEL).save_pretrained(tokenizer_dir)
    args = [
        "--host",
        "127.0.0.1",
        "--tokenizer",
        str(tokenizer_dir),
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hf",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.15",
    ]
    with RemoteOpenAIServer(MODEL, args, max_wait_seconds=480) as server:
        yield server


@pytest.mark.parametrize("stream", [False, True])
def test_duplicate_array_arguments_reach_the_client(server, stream):
    output = (
        "<tool_call><function=set_alarm>"
        "<parameter=groups>[1,2]</parameter>"
        "<parameter=groups>[3,4]</parameter>"
        "</function></tool_call>"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_alarm",
                "parameters": {
                    "type": "object",
                    "properties": {"groups": {"type": "array"}},
                },
            },
        }
    ]
    with server.get_client() as client:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Call set_alarm."}],
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=128,
            stream=stream,
            # Pin the generated XML to test parsing through the real server.
            extra_body={"structured_outputs": {"regex": re.escape(output)}},
        )
        if stream:
            names = set()
            arguments = ""
            for chunk in response:
                for call in chunk.choices[0].delta.tool_calls or []:
                    assert call.index == 0
                    if call.function.name:
                        names.add(call.function.name)
                    arguments += call.function.arguments or ""
            assert names == {"set_alarm"}
        else:
            calls = response.choices[0].message.tool_calls
            assert calls is not None and len(calls) == 1
            assert calls[0].function.name == "set_alarm"
            arguments = calls[0].function.arguments
    assert json.loads(arguments) == {"groups": [[1, 2], [3, 4]]}
