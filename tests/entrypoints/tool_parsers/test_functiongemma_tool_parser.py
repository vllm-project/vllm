# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import regex as re

from tests.utils import RemoteOpenAIServer

MODEL = "unsloth/functiongemma-270m-it"
START = "<start_function_call>"
END = "<end_function_call>"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for name in ("set_alarm", "get_weather", "get_time")
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "functiongemma",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.05",
    ]
    with RemoteOpenAIServer(MODEL, args, max_wait_seconds=480) as server:
        yield server


def _tool_calls(server, output, stream, stream_interval=None):
    """Pin the model output with a regex, so only serving and parsing are tested."""
    extra_body = {"structured_outputs": {"regex": re.escape(output)}}
    if stream_interval is not None:
        extra_body["stream_interval"] = stream_interval
    with server.get_client() as client:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Call the tools."}],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=128,
            stream=stream,
            extra_body=extra_body,
        )
        if not stream:
            calls = response.choices[0].message.tool_calls or []
            return [(c.function.name, json.loads(c.function.arguments)) for c in calls]
        streamed: dict[int, dict] = {}
        for chunk in response:
            for call in chunk.choices[0].delta.tool_calls or []:
                entry = streamed.setdefault(call.index, {"name": None, "args": ""})
                entry["name"] = entry["name"] or call.function.name
                entry["args"] += call.function.arguments or ""
    return [(e["name"], json.loads(e["args"])) for _, e in sorted(streamed.items())]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            "days:[<escape>mon<escape>],enabled:true,hour:7",
            {"days": ["mon"], "enabled": True, "hour": 7},
        ),
        ("", {}),
    ],
    ids=["non_string_arguments", "no_arguments"],
)
def test_arguments_reach_the_client(server, stream, args, expected):
    output = f"{START}call:set_alarm{{{args}}}{END}"
    assert _tool_calls(server, output, stream) == [("set_alarm", expected)]


@pytest.mark.parametrize("stream_interval", [2, 3, 4])
def test_streamed_call_closes_when_next_call_starts_in_same_delta(
    server, stream_interval
):
    """These intervals merge the first call's end tag and the second's start tag."""
    output = (
        f"{START}call:get_weather{{city:<escape>Paris<escape>}}{END}"
        f"{START}call:get_time{{tz:<escape>UTC<escape>}}{END}"
    )
    assert _tool_calls(server, output, True, stream_interval) == [
        ("get_weather", {"city": "Paris"}),
        ("get_time", {"tz": "UTC"}),
    ]
