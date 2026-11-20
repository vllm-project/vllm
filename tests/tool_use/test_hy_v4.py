# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import pytest

from tests.utils import RemoteOpenAIServer

MODEL = "tencent/Hy4-flash-fp8"
TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record",
        "description": "Record an integer count and a text label.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "label": {"type": "string"},
            },
            "required": ["count", "label"],
            "additionalProperties": False,
        },
    },
}


@pytest.fixture(scope="module")
def hy4_server():
    with RemoteOpenAIServer(
        MODEL,
        [
            "--dtype",
            "bfloat16",
            "--data-parallel-size",
            "2",
            "--enable-expert-parallel",
            "--engram-config",
            '{"cpu_offload": true}',
            "--moe-backend",
            "triton",
            "--enforce-eager",
            "--language-model-only",
            "--max-model-len",
            "4096",
            "--max-num-batched-tokens",
            "512",
            "--max-num-seqs",
            "8",
            "--kv-cache-memory-bytes",
            "4294967296",
            "--no-enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--reasoning-parser",
            "hy_v4",
            "--tool-call-parser",
            "hy_v4",
            "--enable-auto-tool-choice",
        ],
        env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4"},
        max_wait_seconds=900,
    ) as server:
        yield server


@pytest.fixture
def hy4_client(hy4_server):
    with hy4_server.get_client() as client:
        yield client


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
def test_hy4_tool_choice_preserves_arguments_in_both_stream_modes(
    hy4_client, thinking, choice
):
    """Real generation must preserve call indices, types, and reasoning routing."""
    tool_choice = (
        {"type": "function", "function": {"name": "record"}}
        if choice == "named"
        else choice
    )
    prompt = (
        "Call record with count=7 and label=hello. Return one tool call and no text."
        if choice == "named"
        else (
            "Call record for both rows in parallel in this reply: "
            "(count=7, label=hello) and (count=9, label=world). "
            "Return two tool calls and no text."
        )
    )
    expected = [{"count": 7, "label": "hello"}]
    if choice != "named":
        expected.append({"count": 9, "label": "world"})
    expected_finish = "stop" if choice == "named" else "tool_calls"
    kwargs = dict(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Issue all independent tool calls in a single assistant turn. "
                    "Do not wait for the first result before issuing the second call."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        tools=[TOOL],
        tool_choice=tool_choice,
        parallel_tool_calls=True,
        temperature=0,
        seed=17,
        max_tokens=768,
        extra_body={}
        if thinking
        else {"chat_template_kwargs": {"reasoning_effort": "no_think"}},
    )
    full = hy4_client.chat.completions.create(**kwargs)
    message = full.choices[0].message
    assert full.choices[0].finish_reason == expected_finish
    assert [call.function.name for call in message.tool_calls] == ["record"] * len(
        expected
    )
    assert [
        json.loads(call.function.arguments) for call in message.tool_calls
    ] == expected
    assert bool(message.reasoning) == thinking
    assert not message.content
    assert (full.usage.completion_tokens_details.reasoning_tokens > 0) == thinking

    arguments = [""] * len(expected)
    names, ids = [], []
    reasoning = content = ""
    finish_reasons = []
    for chunk in hy4_client.chat.completions.create(**kwargs, stream=True):
        delta = chunk.choices[0].delta
        reasoning += getattr(delta, "reasoning", None) or ""
        content += delta.content or ""
        if chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)
        for call in delta.tool_calls or []:
            assert call.index in range(len(expected))
            if call.id:
                ids.append(call.id)
            if call.function.name:
                names.append((call.index, call.function.name))
            arguments[call.index] += call.function.arguments or ""
    assert names == [(i, "record") for i in range(len(expected))]
    assert len(set(ids)) == len(expected)
    assert [json.loads(arg) for arg in arguments] == expected
    assert bool(reasoning) == thinking
    assert not content
    assert finish_reasons == [expected_finish]


@pytest.mark.parametrize("thinking", [False, True])
def test_hy4_plain_content_does_not_expose_reasoning_markup(hy4_client, thinking):
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 1 + 1? Reply with only 2."}],
        temperature=0,
        seed=17,
        max_tokens=512,
        extra_body={}
        if thinking
        else {"chat_template_kwargs": {"reasoning_effort": "no_think"}},
    )
    response = hy4_client.chat.completions.create(**kwargs)
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content.strip() == "2"
    assert bool(response.choices[0].message.reasoning) == thinking
    content = reasoning = ""
    for chunk in hy4_client.chat.completions.create(**kwargs, stream=True):
        delta = chunk.choices[0].delta
        content += delta.content or ""
        reasoning += getattr(delta, "reasoning", None) or ""
        assert not delta.tool_calls
    assert content.strip() == "2"
    assert bool(reasoning) == thinking


def test_hy4_responses_uses_function_schema_in_both_stream_modes(hy4_client):
    kwargs = dict(
        model=MODEL,
        input="Call record with count=7 and label=hello.",
        tools=[{"type": "function", **TOOL["function"]}],
        temperature=0,
        max_output_tokens=256,
        store=False,
        extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
    )
    response = hy4_client.responses.create(**kwargs)
    calls = [item for item in response.output if item.type == "function_call"]
    assert response.status == "completed"
    assert len(calls) == 1
    assert calls[0].name == "record"
    assert json.loads(calls[0].arguments) == {"count": 7, "label": "hello"}
    arguments = ""
    completed = 0
    for event in hy4_client.responses.create(**kwargs, stream=True):
        if event.type == "response.function_call_arguments.delta":
            arguments += event.delta
        elif event.type == "response.completed":
            completed += 1
    assert json.loads(arguments) == {"count": 7, "label": "hello"}
    assert completed == 1
