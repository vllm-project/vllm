# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import jinja2.sandbox

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_nemotron_nano_v2.jinja"
)


def test_tool_call_name_is_json_escaped():
    template = jinja2.sandbox.ImmutableSandboxedEnvironment().from_string(
        TEMPLATE_PATH.read_text()
    )
    tool_name = 'search"quoted\\name'
    rendered = template.render(
        messages=[
            {"role": "user", "content": "Search docs"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": {"query": "vllm"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"result": "ok"}'},
        ],
        add_generation_prompt=False,
    )

    payload = rendered.split("<TOOLCALL>", 1)[1].split("</TOOLCALL>", 1)[0]
    tool_calls = json.loads(payload)

    assert tool_calls[0]["name"] == tool_name
    assert tool_calls[0]["arguments"] == {"query": "vllm"}
