# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only HY4 request parsing with the checkpoint's tokenizer and template."""

import json
from typing import Any

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.utils import StreamingToolReconstructor
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import build_response_output_items
from vllm.parser.parser_manager import ParserManager

MODEL = "tencent/Hy4-flash-fp8"
TOKENIZER_REVISION = "003107fa248f5892064ab1c43af0a7003dd68c9b"
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
def hy4_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)


def _tool_output(arguments):
    calls = "".join(
        "<tool_call:opensource>record"
        "<arg_key:opensource>count</arg_key:opensource>"
        f"<arg_value:opensource>{args['count']}</arg_value:opensource>"
        "<arg_key:opensource>label</arg_key:opensource>"
        f"<arg_value:opensource>{args['label']}</arg_value:opensource>"
        "</tool_call:opensource>"
        for args in arguments
    )
    return f"<tool_calls:opensource>{calls}</tool_calls:opensource>"


def _assert_parsing(
    tokenizer,
    request,
    body,
    *,
    reasoning="",
    content="",
    arguments=(),
    ordinary_tool_tokens=False,
):
    parser_cls = ParserManager.get_parser("hy_v4", "hy_v4", enable_auto_tools=True)
    assert parser_cls is not None
    template_kwargs = request.chat_template_kwargs or {}
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Process the request."}],
        tools=[TOOL] if request.tools else None,
        add_generation_prompt=True,
        return_dict=False,
        **template_kwargs,
    )
    prefix = reasoning + "</think:opensource>" if reasoning else ""
    output = prefix + body
    if ordinary_tool_tokens:
        # Structural-tag decoding can spell tool markers with ordinary tokens.
        token_ids = tokenizer.encode(prefix, add_special_tokens=False) + [
            token_id
            for char in body
            for token_id in tokenizer.encode(char, add_special_tokens=False)
        ]
    else:
        token_ids = tokenizer.encode(output, add_special_tokens=False)
    assert tokenizer.decode(token_ids) == output
    reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False))

    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs=template_kwargs)
    adjusted = parser.adjust_request(request.model_copy(deep=True))
    if ordinary_tool_tokens:
        assert adjusted.structured_outputs is not None
        assert adjusted.structured_outputs.structural_tag is not None
    parser.set_prompt_token_ids(prompt_ids)
    full = parser.parse(
        output,
        adjusted,
        enable_auto_tools=True,
        model_output_token_ids=token_ids,
    )
    full_reasoning, full_content, full_calls = full
    assert (full_reasoning or "", full_content or "") == (reasoning, content)
    assert [call.name for call in full_calls or []] == ["record"] * len(arguments)
    assert [json.loads(call.arguments) for call in full_calls or []] == list(arguments)
    assert parser.count_reasoning_tokens(token_ids) == reasoning_tokens

    for chunk_size in (1, 7, len(token_ids)):
        parser = parser_cls(
            tokenizer, request.tools, chat_template_kwargs=template_kwargs
        )
        adjusted = parser.adjust_request(request.model_copy(deep=True))
        reconstructed = StreamingToolReconstructor(assert_one_tool_per_delta=False)
        streamed_reasoning = previous_text = ""
        for start in range(0, len(token_ids), chunk_size):
            current_text = tokenizer.decode(token_ids[: start + chunk_size])
            delta = parser.parse_delta(
                current_text[len(previous_text) :],
                token_ids[start : start + chunk_size],
                adjusted,
                prompt_token_ids=prompt_ids,
                finished=start + chunk_size >= len(token_ids),
            )
            previous_text = current_text
            if delta is None:
                continue
            streamed_reasoning += delta.reasoning or ""
            if delta.content is not None or delta.tool_calls:
                reconstructed.append_delta(delta)
        assert streamed_reasoning == reasoning
        assert reconstructed.other_content == content
        calls = reconstructed.tool_calls
        assert [call.function.name for call in calls] == ["record"] * len(arguments)
        assert len({call.id for call in calls}) == len(arguments)
        assert [json.loads(call.function.arguments) for call in calls] == list(
            arguments
        )
        assert parser.count_reasoning_tokens(token_ids) == reasoning_tokens
    return full


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
def test_hy4_tool_choice_preserves_arguments_in_both_stream_modes(
    hy4_tokenizer, thinking, choice
):
    """Request adjustment and parsing must agree on native tool-call framing."""
    tool_choice = (
        {"type": "function", "function": {"name": "record"}}
        if choice == "named"
        else choice
    )
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[TOOL],
        tool_choice=tool_choice,
        parallel_tool_calls=True,
        chat_template_kwargs={} if thinking else {"reasoning_effort": "no_think"},
    )
    arguments = [{"count": 7, "label": "hello"}]
    if choice != "named":
        arguments.append({"count": 9, "label": "world"})
    _assert_parsing(
        hy4_tokenizer,
        request,
        _tool_output(arguments),
        reasoning="Record the requested rows." if thinking else "",
        arguments=arguments,
        ordinary_tool_tokens=choice != "auto",
    )


@pytest.mark.parametrize("thinking", [False, True])
def test_hy4_plain_content_does_not_expose_reasoning_markup(hy4_tokenizer, thinking):
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        chat_template_kwargs={} if thinking else {"reasoning_effort": "no_think"},
    )
    _assert_parsing(
        hy4_tokenizer,
        request,
        "2",
        reasoning="The sum is 2." if thinking else "",
        content="2",
    )


def test_hy4_responses_uses_function_schema_in_both_stream_modes(hy4_tokenizer):
    request = ResponsesRequest(
        model=MODEL,
        input="Call record with count=7 and label=hello.",
        tools=[{"type": "function", **TOOL["function"]}],
        chat_template_kwargs={"reasoning_effort": "no_think"},
    )
    arguments = [{"count": 7, "label": "hello"}]
    reasoning, content, calls = _assert_parsing(
        hy4_tokenizer, request, _tool_output(arguments), arguments=arguments
    )
    items = build_response_output_items(
        reasoning, content, calls, tools=request.tools, finish_reason="stop"
    )
    assert len(items) == 1
    call = items[0]
    assert call.type == "function_call"
    assert call.status == "completed"
    assert call.name == "record"
    assert json.loads(call.arguments) == arguments[0]
