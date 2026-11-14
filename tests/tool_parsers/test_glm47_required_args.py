# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for vLLM issue #67471.

A tool-call tag that appears as quoted text — not as a real invocation — used
to be promoted to a tool call whose `arguments` violated the very schema the
server had handed to the model. The parser now rejects such a call while it is
still revocable and re-surfaces the source text as content.

The tag literals are assembled from fragments below so that this file can be
read, quoted and diffed without any tooling mistaking it for a real call.
"""

from __future__ import annotations

import dataclasses
import json
import os
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = os.environ.get("GLM47_TEST_TOKENIZER", "zai-org/GLM-4.7")

TOOL_OPEN = "<tool" + "_call>"
TOOL_CLOSE = "</tool" + "_call>"
ARG_OPEN = "<arg" + "_key>"
ARG_CLOSE = "</arg" + "_key>"
VAL_OPEN = "<arg" + "_value>"
VAL_CLOSE = "</arg" + "_value>"


def render(name: str, args: dict[str, str] | None = None) -> str:
    """Build the model's raw tool-call text for *name* and *args*."""
    body = "".join(
        f"{ARG_OPEN}{key}{ARG_CLOSE}{VAL_OPEN}{value}{VAL_CLOSE}"
        for key, value in (args or {}).items()
    )
    return f"{TOOL_OPEN}{name}{body}{TOOL_CLOSE}"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def tools():
    return [
        # Requires nothing: a bare call is legitimate and must survive.
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
        # Requires `query`: a bare call is a schema violation.
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="search_docs",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


@pytest.fixture
def request_(tools) -> ChatCompletionRequest:
    req = Mock(spec=ChatCompletionRequest)
    req.tools = tools
    req.tool_choice = "auto"
    return req


@pytest.fixture
def parser(tokenizer, tools):
    return Glm47MoeModelToolParser(tokenizer, tools=tools)


def stream(parser, request_, text: str, chunk: int = 7):
    """Feed *text* to the parser in pieces, mimicking incremental decoding."""
    deltas = []
    previous = ""
    for i in range(0, len(text), chunk):
        piece = text[i : i + chunk]
        current = previous + piece
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=piece,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request_,
        )
        if delta is not None:
            deltas.append(delta)
        previous = current
    final = parser.finish_streaming()
    if final is not None:
        deltas.append(final)
    return deltas


def test_missing_required_arg_is_not_a_tool_call(parser, request_):
    """The quoted tag has no `query`; it must surface as text, not a call."""
    text = render("search_docs")

    result = parser.extract_tool_calls(text, request_)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content is not None
    assert "search_docs" in result.content


def test_missing_required_arg_keeps_source_text_verbatim(parser, request_):
    """Nothing is swallowed: the caller sees exactly what the model emitted."""
    text = render("search_docs")

    result = parser.extract_tool_calls(text, request_)

    assert result.content is not None
    assert text in result.content


def test_required_arg_present_still_produces_a_call(parser, request_):
    """The gate must not get in the way of a well-formed call."""
    text = render("search_docs", {"query": "vllm parser"})

    result = parser.extract_tool_calls(text, request_)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "search_docs"
    assert json.loads(call.function.arguments) == {"query": "vllm parser"}


def test_partial_required_args_still_a_call_when_all_required_present(parser, request_):
    """Optional parameters may be omitted freely."""
    text = render("search_docs", {"query": "x", "top_k": "8"})

    result = parser.extract_tool_calls(text, request_)

    assert result.tools_called is True
    assert json.loads(result.tool_calls[0].function.arguments)["top_k"] == 8


def test_tool_without_required_list_is_untouched(parser, request_):
    """No `required` in the schema means no constraint was violated."""
    text = render("get_current_date")

    result = parser.extract_tool_calls(text, request_)

    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "get_current_date"


def test_streaming_also_degrades_to_text(parser, request_):
    """The same rejection must hold when the text arrives in pieces."""
    text = render("search_docs")

    deltas = stream(parser, request_, text)

    tool_calls = [tc for d in deltas for tc in (d.tool_calls or [])]
    content = "".join(d.content for d in deltas if d.content)

    assert tool_calls == []
    assert "search_docs" in content


def test_streaming_valid_call_unaffected(parser, request_):
    """Streaming a valid call still yields exactly one call for index 0."""
    text = render("search_docs", {"query": "hello"})

    deltas = stream(parser, request_, text)

    # A single call legitimately spans several deltas (name, then argument
    # fragments); merge them the way a client would.
    merged: dict[int, dict[str, str]] = {}
    for delta in deltas:
        for tc in delta.tool_calls or []:
            slot = merged.setdefault(tc.index, {"name": "", "args": ""})
            if tc.function is not None:
                slot["name"] += tc.function.name or ""
                slot["args"] += tc.function.arguments or ""

    assert len(merged) == 1
    assert merged[0]["name"] == "search_docs"
    assert json.loads(merged[0]["args"]) == {"query": "hello"}


def test_gate_is_opt_in(parser, request_):
    """Disabling the switch restores the pre-fix behaviour."""
    engine = parser._parser_engine
    engine.parser_engine_config = dataclasses.replace(
        engine.parser_engine_config, validate_required_args=False
    )

    result = parser.extract_tool_calls(render("search_docs"), request_)

    assert result.tools_called is True
    assert json.loads(result.tool_calls[0].function.arguments) == {}


def test_call_with_only_optional_args_degrades(parser, request_):
    """A call that carries some arguments but omits a required one is still
    a schema violation: the gate must not depend on `arguments` being empty."""
    text = render("search_docs", {"top_k": "8"})

    result = parser.extract_tool_calls(text, request_)

    assert result.tools_called is False
    assert result.content is not None
    assert text in result.content


def test_streaming_holds_the_call_back_until_arguments_close(parser, request_):
    """Nothing about a gated call may reach the client before it is checked."""
    text = render("search_docs", {"query": "hello"})

    deltas = stream(parser, request_, text, chunk=5)

    emitted = [d for d in deltas if d.tool_calls]
    assert len(emitted) == 1
    call = emitted[0].tool_calls[0]
    assert call.function.name == "search_docs"
    assert json.loads(call.function.arguments) == {"query": "hello"}


def test_degraded_text_keeps_its_place(parser, request_):
    """The rejected call is re-surfaced where it stood, not after the text
    that followed it."""
    text = "before" + render("search_docs") + "after"

    deltas = stream(parser, request_, text, chunk=6)
    content = "".join(d.content for d in deltas if d.content)

    assert content.index("before") < content.index(TOOL_OPEN) < content.index("after")
