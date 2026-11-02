# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the Granite tool-call parser.

Granite 4.2 emits tool calls in the ``<tool_call>``/``<function=>``/
``<parameter=>`` XML format. The in-tree ``qwen3_coder`` parser already handles
that exact format, so these tests use it as a reference implementation: the
correctness bar is **byte-identical behavior vs ``qwen3_coder``** in both
streaming and non-streaming modes.

These tests run a shared corpus of model outputs — including parallel calls,
reasoning, every JSON arg type, empty args, escaped strings, multiline values,
whitespace-in-tags and the malformed/partial cases — through both parsers and
assert the results are identical.
"""

import json
import random
from collections.abc import Generator

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tool_parsers import ToolParser, ToolParserManager

# Granite 4.2 tokenizer (only the tokenizer is needed for parser tests).
GRANITE_MODEL = "ibm-granite/granite-4.2-8b"

QWEN_PARSER = "qwen3_coder"
GRANITE_PARSER = "granite_toolcall_parser"


# ---------------------------------------------------------------------------
# Tools (shared by both parsers so schema-based type coercion is exercised)
# ---------------------------------------------------------------------------
SAMPLE_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                },
                "required": ["city", "state"],
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "all_types",
            "description": "Exercise every JSON value type",
            "parameters": {
                "type": "object",
                "properties": {
                    "int_param": {"type": "integer"},
                    "float_param": {"type": "number"},
                    "bool_param": {"type": "boolean"},
                    "null_param": {"type": "null"},
                    "str_param": {"type": "string"},
                    "array_param": {"type": "array", "items": {"type": "string"}},
                    "obj_param": {"type": "object"},
                },
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "write_file",
            "description": "Write file content verbatim",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "greet",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Corpus of model outputs covering the corner cases
# ---------------------------------------------------------------------------
SINGLE_CALL = """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

CALL_WITH_LEADING_CONTENT = "Sure! Let me check the weather for you." + SINGLE_CALL

PARALLEL_CALLS = """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_current_weather>
<parameter=city>
Orlando
</parameter>
<parameter=state>
FL
</parameter>
</function>
</tool_call>"""

NO_CALL = "This is a plain response with no tool calls at all."

REASONING_THEN_CALL = (
    "<think>\nThe user wants the weather; I'll call the tool.\n</think>\n" + SINGLE_CALL
)

ALL_ARG_TYPES = """<tool_call>
<function=all_types>
<parameter=int_param>
42
</parameter>
<parameter=float_param>
3.14
</parameter>
<parameter=bool_param>
true
</parameter>
<parameter=null_param>
null
</parameter>
<parameter=str_param>
hello world
</parameter>
<parameter=array_param>
["a", "b", "c"]
</parameter>
<parameter=obj_param>
{"key": "value", "n": 10}
</parameter>
</function>
</tool_call>"""

EMPTY_ARGS = """<tool_call>
<function=get_current_weather>
</function>
</tool_call>"""

ESCAPED_STRINGS = """<tool_call>
<function=greet>
<parameter=message>
She said "hi" and left C:\\temp\\x then wrote line1
line2
</parameter>
</function>
</tool_call>"""

MULTILINE_VALUE = (
    "<tool_call>\n<function=write_file>\n"
    "<parameter=content>\n"
    "    def foo():\n        return 1\n\n"
    "</parameter>\n</function>\n</tool_call>"
)

WHITESPACE_IN_TAGS = """<tool_call>
<function=get_current_weather>
< parameter = city >
Dallas
</ parameter >
< parameter = state >
TX
</ parameter >
</function>
</tool_call>"""

# Malformed / partial cases (parser must degrade gracefully, identically).
MISSING_CLOSING_PARAMETER = """Let me check the weather for you:
<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
<parameter=state>
TX
</parameter>
</function>
</tool_call>"""

MISSING_TOOL_CALL_TAG = """I'll check the weather.

<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
</function>
</tool_call>"""

MISSING_GT_ON_NAME = (
    "<tool_call>\n"
    "<function=get_current_weather\n"
    "<parameter=city>Dallas</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

# regression issue 58227: unclosed function name followed by another.
UNCLOSED_THEN_NEXT_FUNC = (
    "<tool_call>\n"
    "<function=bad_func_no_gt\n"
    "<function=get_current_weather>\n"
    "<parameter=city>Dallas</parameter>\n"
    "<parameter=state>TX</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

# Consecutive calls with no closing </tool_call> between them.
CONSECUTIVE_NO_CLOSE = (
    "<tool_call>\n"
    "<function=get_current_weather>\n"
    "<parameter=city>Dallas</parameter>\n"
    "</function>\n"
    "<function=get_current_weather>\n"
    "<parameter=city>Orlando</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

CORPUS = {
    "single_call": SINGLE_CALL,
    "call_with_leading_content": CALL_WITH_LEADING_CONTENT,
    "parallel_calls": PARALLEL_CALLS,
    "no_call": NO_CALL,
    "reasoning_then_call": REASONING_THEN_CALL,
    "all_arg_types": ALL_ARG_TYPES,
    "empty_args": EMPTY_ARGS,
    "escaped_strings": ESCAPED_STRINGS,
    "multiline_value": MULTILINE_VALUE,
    "whitespace_in_tags": WHITESPACE_IN_TAGS,
    "missing_closing_parameter": MISSING_CLOSING_PARAMETER,
    "missing_tool_call_tag": MISSING_TOOL_CALL_TAG,
    "missing_gt_on_name": MISSING_GT_ON_NAME,
    "unclosed_then_next_func": UNCLOSED_THEN_NEXT_FUNC,
    "consecutive_no_close": CONSECUTIVE_NO_CLOSE,
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer() -> TokenizerLike:
    # Both parsers MUST share one tokenizer for a fair byte-identity comparison.
    return get_tokenizer(tokenizer_name=GRANITE_MODEL)


# Exercise both thinking modes: production serves granite-4.2 with
# enable_thinking True and False, so parity must hold in both.
@pytest.fixture(params=[True, False], ids=["think", "no_think"])
def thinking(request) -> bool:
    return request.param


def _chat_kwargs(thinking: bool) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": thinking}}


def _build(parser_name: str, tokenizer, thinking: bool) -> ToolParser:
    # Build the parser the way the serving layer would, honoring enable_thinking.
    return ToolParserManager.get_tool_parser(parser_name)(
        tokenizer, tools=SAMPLE_TOOLS, **_chat_kwargs(thinking)
    )


@pytest.fixture
def qwen_parser(tokenizer, thinking) -> ToolParser:
    return _build(QWEN_PARSER, tokenizer, thinking)


@pytest.fixture
def granite_parser(tokenizer, thinking) -> ToolParser:
    return _build(GRANITE_PARSER, tokenizer, thinking)


def _request(thinking: bool = True) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=SAMPLE_TOOLS,
        chat_template_kwargs={"enable_thinking": thinking},
    )


def _normalize(extracted) -> tuple:
    """A comparable, id-independent view of ExtractedToolCallInformation."""
    calls = []
    for tc in extracted.tool_calls:
        # arguments is a JSON string; compare the decoded object so key order
        # and whitespace in the serialization never cause spurious diffs.
        args = json.loads(tc.function.arguments) if tc.function.arguments else None
        calls.append((tc.type, tc.function.name, args))
    return (extracted.tools_called, extracted.content, calls)


def _token_deltas(
    tokenizer: TokenizerLike, model_output: str
) -> Generator[str, None, None]:
    """Yield one incremental text delta per model token, as vLLM would stream."""
    all_token_ids = tokenizer.encode(model_output, add_special_tokens=False)
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    for i in range(len(all_token_ids)):
        current_token_ids = all_token_ids[: i + 1]
        new_tokens, delta_text, new_prefix_offset, new_read_offset = (
            detokenize_incrementally(
                tokenizer=tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
        )
        yield delta_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


def _stream_reconstruct(
    parser: ToolParser,
    tokenizer: TokenizerLike,
    deltas: list[str],
    request: ChatCompletionRequest,
) -> tuple:
    """Feed text deltas to a parser and reconstruct an id-independent view."""
    content = ""
    # index -> [name, arguments]
    tools: dict[int, list] = {}
    previous_text = ""
    previous_token_ids: list[int] = []
    for delta_text in deltas:
        if not delta_text:
            continue
        delta_token_ids = tokenizer.encode(delta_text, add_special_tokens=False)
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids
        msg: DeltaMessage | None = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request=request,
        )
        previous_text = current_text
        previous_token_ids = current_token_ids
        if msg is None:
            continue
        if msg.content:
            content += msg.content
        for tc in msg.tool_calls or []:
            slot = tools.setdefault(tc.index, [None, ""])
            if tc.function and tc.function.name:
                slot[0] = tc.function.name
            if tc.function and tc.function.arguments:
                slot[1] += tc.function.arguments

    calls = []
    for idx in sorted(tools):
        name, raw_args = tools[idx]
        calls.append((name, json.loads(raw_args) if raw_args else None))
    return content, calls


def _random_chunks(s: str, min_len: int, max_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    chunks = []
    i = 0
    while i < len(s):
        n = rng.randint(min_len, max_len)
        chunks.append(s[i : i + n])
        i += n
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CORPUS))
def test_nonstreaming_parity(name, thinking, qwen_parser, granite_parser):
    """Non-streaming extraction is identical to qwen3_coder (both modes)."""
    model_output = CORPUS[name]
    req = _request(thinking)
    q = qwen_parser.extract_tool_calls(model_output, request=req)
    g = granite_parser.extract_tool_calls(model_output, request=req)
    assert _normalize(g) == _normalize(q)


@pytest.mark.parametrize("name", list(CORPUS))
def test_streaming_parity_token_deltas(
    name, thinking, tokenizer, qwen_parser, granite_parser
):
    """Realistic per-token streaming is identical to qwen3_coder (both modes)."""
    model_output = CORPUS[name]
    deltas = list(_token_deltas(tokenizer, model_output))
    req = _request(thinking)
    q = _stream_reconstruct(qwen_parser, tokenizer, deltas, req)
    g = _stream_reconstruct(granite_parser, tokenizer, deltas, req)
    assert g == q


@pytest.mark.parametrize("name", list(CORPUS))
@pytest.mark.parametrize("chunk", [(1, 1), (1, 2), (5, 7), (6, 20)])
def test_streaming_parity_random_chunks(
    name, chunk, thinking, tokenizer, qwen_parser, granite_parser
):
    """Arbitrary chunk boundaries (tags split mid-token) stay identical."""
    model_output = CORPUS[name]
    min_len, max_len = chunk
    deltas = _random_chunks(
        model_output, min_len, max_len, seed=min_len * 100 + max_len
    )
    req = _request(thinking)
    q = _stream_reconstruct(qwen_parser, tokenizer, deltas, req)
    g = _stream_reconstruct(granite_parser, tokenizer, deltas, req)
    assert g == q


@pytest.mark.parametrize("name", list(CORPUS))
def test_streaming_matches_nonstreaming(
    name, thinking, tokenizer, qwen_parser, granite_parser
):
    """Granite's streaming-vs-nonstreaming relationship matches qwen3_coder's.

    The shared engine does not guarantee that streaming and non-streaming
    produce identical results for every deeply-malformed input (e.g. an
    unclosed function name streams a trailing newline the non-streaming path
    trims). We therefore assert that Granite reproduces whatever relationship
    qwen3_coder has, rather than a stricter invariant the engine does not hold.
    """
    model_output = CORPUS[name]
    req = _request(thinking)

    def ns_calls(parser):
        ns = parser.extract_tool_calls(model_output, request=req)
        return [
            (
                tc.function.name,
                json.loads(tc.function.arguments) if tc.function.arguments else None,
            )
            for tc in ns.tool_calls
        ]

    def stream_calls(parser):
        deltas = list(_token_deltas(tokenizer, model_output))
        return _stream_reconstruct(parser, tokenizer, deltas, req)[1]

    # Non-streaming and streaming agree between the two parsers (covered
    # elsewhere too); here we assert the stream-vs-nonstream DELTA is the same.
    q_matches = stream_calls(qwen_parser) == ns_calls(qwen_parser)
    g_matches = stream_calls(granite_parser) == ns_calls(granite_parser)
    assert g_matches == q_matches


def test_works_standalone_without_reasoning_parser(tokenizer, thinking):
    """The tool parser extracts calls on its own, with no reasoning parser.

    These parser-level tests never configure a reasoning parser, so a passing
    extraction here is exactly the "tool parser used without
    granite_thinking_parser" scenario. We assert a reasoning+tool output still
    yields the tool call (and the <think> block stays in content, since no
    reasoning parser is present to split it out).
    """
    granite = _build(GRANITE_PARSER, tokenizer, thinking)
    req = _request(thinking)
    extracted = granite.extract_tool_calls(REASONING_THEN_CALL, request=req)
    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "get_current_weather"
    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args == {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
    # No reasoning parser configured -> reasoning markup remains in content.
    assert "<think>" in (extracted.content or "")
