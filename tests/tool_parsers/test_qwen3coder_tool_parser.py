# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Coder-parser-specific tests.

Tests that exercise behaviour shared with the XML parser live in
``tests/tool_parsers/test_qwen3_xml_coder_shared.py``.  Only tests that
depend on Coder-only API (e.g. ``is_tool_call_started``) or on Coder-only
streaming behaviour (e.g. character-by-character chunking) belong here.
"""

import json

import pytest

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def qwen3_tool_parser(qwen3_tokenizer):
    return Qwen3CoderToolParser(qwen3_tokenizer, tools=None)


def test_streaming_trailing_text_after_tool_with_literal_close_tag_in_value(
    qwen3_tokenizer,
):
    """A tool call's parameter value contains a literal ``</tool_call>``
    string.  After the real tool call closes, trailing free text must
    still be emitted as content.

    The naive ``current_text.count(</tool_call>)`` and
    ``current_text.find(</tool_call>)`` used by the early-advance and
    ``_advance_to_next_tool`` logic don't distinguish literal text from
    structural delimiters.  This can cause ``_sent_content_idx`` to land
    INSIDE the tool's parameter value, after which the trailing text
    fails to be emitted.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )

    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        )
    ]
    parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    # The parameter value contains a literal ``</tool_call>`` string.
    # The real ``</tool_call>`` follows after ``</function>``.
    delta_1 = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>foo.py</parameter>\n"
        "<parameter=content>\n"
        "doc = '<tool_call>example</tool_call>'\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=delta_1,
        delta_text=delta_1,
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
        request=request,
    )

    delta_2 = "\nDone, file written!"
    text2 = delta_1 + delta_2
    msg2 = parser.extract_tool_calls_streaming(
        previous_text=delta_1,
        current_text=text2,
        delta_text=delta_2,
        previous_token_ids=[1],
        current_token_ids=[1, 2],
        delta_token_ids=[2],
        request=request,
    )
    contents = []
    if msg2 and msg2.content:
        contents.append(msg2.content)
    # EOS-style empty delta to flush
    msg3 = parser.extract_tool_calls_streaming(
        previous_text=text2,
        current_text=text2,
        delta_text="",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3],
        delta_token_ids=[3],
        request=request,
    )
    if msg3 and msg3.content:
        contents.append(msg3.content)

    full = "".join(contents)
    assert "Done, file written!" in full, (
        f"Trailing text after a tool call whose parameter value contains "
        f"a literal </tool_call> was dropped. Got content: {full!r}"
    )


def test_streaming_second_tool_after_first_with_literal_close_tag_in_value(
    qwen3_tokenizer,
):
    """A first tool call's parameter value contains a literal
    ``</tool_call>``.  A SECOND structural tool call follows after the
    real ``</tool_call>``.  Both tool calls and any inter-call content
    must be emitted correctly.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )

    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "log",
                "parameters": {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                },
            },
        ),
    ]
    parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    full = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>foo.py</parameter>\n"
        "<parameter=content>\n"
        "doc = '<tool_call>example</tool_call>'\n"
        "</parameter>\n</function>\n</tool_call>"
        "\n"
        "<tool_call>\n<function=log>\n"
        "<parameter=msg>done</parameter>\n"
        "</function>\n</tool_call>"
    )

    msg = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=full,
        delta_text=full,
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
        request=request,
    )
    assert msg is not None
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 2, (
        f"Expected 2 tool calls, got {len(msg.tool_calls)}: {msg.tool_calls}"
    )
    names = [tc.function.name for tc in msg.tool_calls]
    assert names == ["write_file", "log"], f"Wrong tool names: {names}"


def test_streaming_content_before_and_between_two_tool_calls_one_delta(
    qwen3_tool_parser,
):
    """MTP / spec-decode: a single delta delivers free text BEFORE tool 1
    AND free text BETWEEN tool 1 and tool 2.  Both content fragments must
    be emitted; the recursion path used to drop the second one because of a
    ``not result.content`` guard that discarded the recursion's content
    when the outer call already had content of its own.
    """
    request = ChatCompletionRequest(model=MODEL, messages=[])
    delta = (
        "before text "
        "<tool_call>\n<function=foo>\n"
        "<parameter=a>\n1\n</parameter>\n"
        "</function>\n</tool_call>"
        "between text "
        "<tool_call>\n<function=bar>\n"
        "<parameter=b>\n2\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    msg = qwen3_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=delta,
        delta_text=delta,
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
        request=request,
    )
    assert msg is not None
    assert msg.content is not None, "outer content lost"
    assert "before text " in msg.content, (
        f"missing 'before text' content: {msg.content!r}"
    )
    assert "between text " in msg.content, (
        f"recursion content 'between text' was dropped because the outer "
        f"already had content. Got: {msg.content!r}"
    )


def test_extract_tool_calls_streaming_split_tag(qwen3_tool_parser):
    """``<tool_call>`` arrives split across two deltas (``<tool`` then
    ``_call>``).  ``is_tool_call_started`` must flip to ``True`` once the
    full tag exists in ``current_text``, and the partial tag must not leak
    into ``DeltaMessage.content``.

    This relies on the Coder parser's ``is_tool_call_started`` attribute,
    which has no equivalent on the XML parser.
    """
    request = ChatCompletionRequest(model=MODEL, messages=[])

    prev_text_1 = "I will use a tool."
    delta_text_1 = "<tool"
    curr_text_1 = prev_text_1 + delta_text_1

    msg1 = qwen3_tool_parser.extract_tool_calls_streaming(
        previous_text=prev_text_1,
        current_text=curr_text_1,
        delta_text=delta_text_1,
        previous_token_ids=[1, 2, 3],
        current_token_ids=[1, 2, 3, 4],
        delta_token_ids=[4],
        request=request,
    )

    prev_text_2 = curr_text_1
    delta_text_2 = "_call>"
    curr_text_2 = prev_text_2 + delta_text_2

    msg2 = qwen3_tool_parser.extract_tool_calls_streaming(
        previous_text=prev_text_2,
        current_text=curr_text_2,
        delta_text=delta_text_2,
        previous_token_ids=[1, 2, 3, 4],
        current_token_ids=[1, 2, 3, 4, 5],
        delta_token_ids=[5],
        request=request,
    )

    assert qwen3_tool_parser.is_tool_call_started is True

    if msg1 and msg1.content:
        assert "<tool" not in msg1.content
    if msg2 and msg2.content:
        assert "_call>" not in msg2.content


def test_streaming_char_by_char_literal_balises_in_value(qwen3_tokenizer):
    """Stress test: a WriteFile tool call whose ``content`` value embeds a
    complete literal ``<tool_call>...</tool_call>`` block — including
    ``<parameter=path>...</parameter>`` and ``<parameter=content>...
    </parameter>`` with names that match the OUTER tool's schema —
    streamed one character at a time.

    Reproduces the qwen-code scenario where the model writes a parser
    fixture file: every literal ``<tool_call>``, ``<function=...>``,
    ``<parameter=NAME>``, ``</parameter>``, ``</function>`` and
    ``</tool_call>`` inside the ``content`` value must stay inside the
    value; no spurious second tool call, no value truncation.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )

    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        )
    ]
    parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    nested_content = (
        'doc = """\n'
        "<tool_call>\n"
        "<function=write_file>\n"
        "<parameter=path>\nliteral/value.txt\n</parameter>\n"
        "<parameter=content>\nhello\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        '"""\n'
    )

    full_output = (
        "<tool_call>\n"
        "<function=write_file>\n"
        "<parameter=path>\nfixture.py\n</parameter>\n"
        f"<parameter=content>\n{nested_content}</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    tool_states: dict[int, dict] = {}
    current_text = ""
    previous_text = ""
    for ch in full_output:
        previous_text = current_text
        current_text += ch
        delta_message = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=ch,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta_message and delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index
                state = tool_states.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""}
                )
                if tool_call.id:
                    state["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        state["name"] = tool_call.function.name
                    if tool_call.function.arguments is not None:
                        state["arguments"] += tool_call.function.arguments

    assert list(tool_states.keys()) == [0], (
        f"Expected exactly one tool call; got indices "
        f"{list(tool_states.keys())} — a literal nested <tool_call> "
        f"was promoted to a real call."
    )
    state = tool_states[0]
    assert state["name"] == "write_file"
    args = json.loads(state["arguments"])
    assert list(args.keys()) == ["path", "content"], (
        f"Spurious params from embedded literals: {list(args.keys())}"
    )
    assert args["path"] == "fixture.py"
    assert args["content"] == nested_content.rstrip("\n"), (
        f"content was truncated/corrupted: {args.get('content')!r}"
    )


def test_extract_tool_calls_streaming_various_chunk_sizes(
    qwen3_tokenizer,
):
    """Coder streaming must reconstruct arguments correctly even when the
    deltas arrive a single character at a time.

    The XML parser's SAX-based streaming cannot tolerate ``chunk_size=1``
    by design (an XML tag is not parseable until ``>`` arrives), so this
    robustness test stays Coder-only.
    """
    request = ChatCompletionRequest(model="test", messages=[])

    template_text = """<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>"""

    for chunk_size in [1, 3, 15, len(template_text)]:
        parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=None)

        tool_states = {}
        current_text = ""
        previous_text = ""
        ptr = 0

        while ptr < len(template_text):
            delta = template_text[ptr:ptr + chunk_size]
            previous_text = current_text
            current_text += delta
            ptr += chunk_size

            delta_message = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )

            if delta_message and delta_message.tool_calls:
                for tool_call in delta_message.tool_calls:
                    idx = tool_call.index
                    if idx not in tool_states:
                        tool_states[idx] = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "type": None,
                        }
                    if tool_call.id:
                        tool_states[idx]["id"] = tool_call.id
                    if tool_call.type:
                        tool_states[idx]["type"] = tool_call.type
                    if tool_call.function:
                        if tool_call.function.name:
                            tool_states[idx]["name"] = tool_call.function.name
                        if tool_call.function.arguments is not None:
                            tool_states[idx]["arguments"] += (
                                tool_call.function.arguments
                            )

        assert 0 in tool_states, f"chunk_size={chunk_size}"
        assert tool_states[0]["name"] == "example_function_name"
        args = json.loads(tool_states[0]["arguments"])
        assert args["example_parameter_1"] == "value_1"
        assert args["example_parameter_2"] == (
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines"
        )


def test_extract_tool_calls_anyof_type_conversion(qwen3_tokenizer):
    """Test type conversion for anyOf/oneOf nullable schemas (Pydantic v2).

    Pydantic v2 emits anyOf for Optional[T] fields, e.g.:
        Optional[int] -> {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    The parser must extract the non-null type and apply the correct
    conversion (int(), float(), etc.) instead of returning a raw string.

    Coder-specific: this also exercises the JSON-Schema list-form
    ``{"type": ["integer", "null"]}`` which the Coder parser coerces to an
    int (the XML parser keeps it as a string), so the assertions only hold
    for the Coder parser and the test stays out of the shared suite.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "test_anyof",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anyof_int": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"},
                            ],
                            "default": 5,
                        },
                        "anyof_str": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                        },
                        "anyof_array": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ],
                        },
                        "anyof_obj": {
                            "anyOf": [
                                {"type": "object"},
                                {"type": "null"},
                            ],
                        },
                        "type_as_array": {
                            "type": ["integer", "null"],
                        },
                        "multi_non_null": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "null"},
                            ],
                        },
                    },
                },
            },
        )
    ]

    model_output = """<tool_call>
<function=test_anyof>
<parameter=anyof_int>
5
</parameter>
<parameter=anyof_str>
hello
</parameter>
<parameter=anyof_array>
["a", "b", "c"]
</parameter>
<parameter=anyof_obj>
{"key": "value"}
</parameter>
<parameter=type_as_array>
42
</parameter>
<parameter=multi_non_null>
some text
</parameter>
</function>
</tool_call>"""

    parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    extracted = parser.extract_tool_calls(model_output, request=request)

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["anyof_int"] == 5
    assert isinstance(args["anyof_int"], int)
    assert args["anyof_str"] == "hello"
    assert isinstance(args["anyof_str"], str)
    assert args["anyof_array"] == ["a", "b", "c"]
    assert isinstance(args["anyof_array"], list)
    assert args["anyof_obj"] == {"key": "value"}
    assert isinstance(args["anyof_obj"], dict)
    assert args["type_as_array"] == 42
    assert isinstance(args["type_as_array"], int)
    # Multi non-null: anyOf[string, integer, null] → first non-null is string
    assert args["multi_non_null"] == "some text"
    assert isinstance(args["multi_non_null"], str)


def test_extract_tool_calls_anyof_type_conversion_streaming(qwen3_tokenizer):
    """Test streaming e2e for anyOf/oneOf nullable schemas (Pydantic v2).

    Verifies that the full streaming pipeline correctly resolves types from
    anyOf schemas and produces valid JSON with properly typed values.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "search_web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                        },
                        "count": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"},
                            ],
                            "default": 5,
                        },
                        "verbose": {
                            "anyOf": [
                                {"type": "boolean"},
                                {"type": "null"},
                            ],
                        },
                    },
                },
            },
        )
    ]

    model_output = """<tool_call>
<function=search_web>
<parameter=query>
vllm tool parser
</parameter>
<parameter=count>
10
</parameter>
<parameter=verbose>
true
</parameter>
</function>
</tool_call>"""

    parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    reconstructor = run_tool_extraction_streaming(
        parser, model_output, request, assert_one_tool_per_delta=False
    )

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "search_web"
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args["query"] == "vllm tool parser"
    assert isinstance(args["query"], str)
    assert args["count"] == 10
    assert isinstance(args["count"], int)
    assert args["verbose"] is True
    assert isinstance(args["verbose"], bool)
