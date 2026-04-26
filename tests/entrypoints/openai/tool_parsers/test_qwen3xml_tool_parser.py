# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for StreamingXMLToolCallParser (Qwen3 XML tool-call format).

These tests exercise the character-level state machine directly and do not
require a running model server or GPU.  They cover:

* Basic correctness (char-by-char and whole-chunk modes must agree)
* Tag fragmentation at every byte position inside structural tags
* HTML / XML fragments as literal parameter-value content
* Tags that look like structural tags but aren't (partial names, etc.)
* All parameter types (integer, float, boolean, object, array, string)
* Multi-parameter and multi-call scenarios
* Streaming delta reconstruction
* Premature-abort resilience (name=None ghost call prevention)
* Resync when a second <tool_call> arrives without a closing </tool_call>
"""

import json

import pytest

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser

# ---------------------------------------------------------------------------
# Tool schema fixture used for typed-parameter tests
# ---------------------------------------------------------------------------

TOOLS_TYPED = [
    {
        "function": {
            "name": "typed_fn",
            "parameters": {
                "properties": {
                    "count": {"type": "integer"},
                    "ratio": {"type": "float"},
                    "flag": {"type": "boolean"},
                    "data": {"type": "object"},
                    "items": {"type": "array"},
                    "label": {"type": "string"},
                }
            },
        }
    }
]

BASIC = (
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=city>\n"
    "London\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def feed_char_by_char(text: str, tools=None) -> DeltaMessage:
    p = StreamingXMLToolCallParser()
    p.set_tools(tools)
    for ch in text:
        p.parse_single_streaming_chunks(ch)
    p.finalize()
    return p.collect_all()


def feed_chunks(*chunks: str, tools=None) -> DeltaMessage:
    p = StreamingXMLToolCallParser()
    p.set_tools(tools)
    for c in chunks:
        p.parse_single_streaming_chunks(c)
    p.finalize()
    return p.collect_all()


def assert_tool(
    result: DeltaMessage, fn_name: str, expected: dict, index: int = 0
) -> None:
    assert result.tool_calls and len(result.tool_calls) > index
    tc = result.tool_calls[index]
    assert tc.function is not None
    assert tc.function.name == fn_name, f"name={tc.function.name!r} != {fn_name!r}"
    got = json.loads(tc.function.arguments or "{}")
    assert got == expected, (
        f"args mismatch:\n  got:      {tc.function.arguments}\n"
        f"  expected: {json.dumps(expected)}"
    )


def all_splits(s: str):
    """Every 2-part partition of *s*."""
    return [(s[:i], s[i:]) for i in range(1, len(s))]


# ===========================================================================
# Section 1 – Basic correctness
# ===========================================================================


def test_basic_char_by_char():
    assert_tool(feed_char_by_char(BASIC), "get_weather", {"city": "London"})


def test_basic_whole_chunk():
    assert_tool(feed_chunks(BASIC), "get_weather", {"city": "London"})


def test_collect_all_idempotent():
    p = StreamingXMLToolCallParser()
    p.parse_single_streaming_chunks(BASIC)
    p.finalize()
    r1 = p.collect_all()
    r2 = p.collect_all()
    assert r1.tool_calls[0].function.arguments == r2.tool_calls[0].function.arguments
    assert json.loads(r1.tool_calls[0].function.arguments) == {"city": "London"}


def test_whole_chunk_matches_char_by_char():
    a = feed_char_by_char(BASIC).tool_calls[0].function.arguments
    b = feed_chunks(BASIC).tool_calls[0].function.arguments
    assert a == b


def test_implicit_tool_call_no_wrapper():
    # <function=…> without a preceding <tool_call> must NOT create a tool call.
    # It is treated as plain text content since the Qwen3 XML format always
    # wraps function calls in <tool_call>…</tool_call>.
    r = feed_char_by_char(
        "<function=quick>\n<parameter=z>\nhello\n</parameter>\n</function>"
    )
    assert not r.tool_calls, f"Expected no tool calls, got {r.tool_calls}"


def test_no_parameters_produces_empty_object():
    r = feed_char_by_char("<tool_call>\n<function=noop>\n</function>\n</tool_call>")
    assert r.tool_calls[0].function.arguments == "{}"


# ===========================================================================
# Section 2 – Tag fragmentation
# ===========================================================================


@pytest.mark.parametrize(
    "head,tail",
    all_splits(
        "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n</tool_call>"
    ),
)
def test_frag_tool_call_tag(head, tail):
    r = feed_chunks(head, tail)
    assert r.tool_calls and r.tool_calls[0].function.name == "f"


@pytest.mark.parametrize(
    "head,tail",
    all_splits(
        "<tool_call>\n<function=my_func>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>"
    ),
)
def test_frag_function_tag(head, tail):
    r = feed_chunks(head, tail)
    assert r.tool_calls and r.tool_calls[0].function.name == "my_func"


@pytest.mark.parametrize(
    "head,tail",
    all_splits(
        "<tool_call>\n<function=f>\n<parameter=x>\nval\n</parameter>\n</function>\n</tool_call>"
    ),
)
def test_frag_close_parameter_tag(head, tail):
    r = feed_chunks(head, tail)
    assert json.loads(r.tool_calls[0].function.arguments) == {"x": "val"}


def test_frag_one_char_per_chunk():
    assert_tool(feed_char_by_char(BASIC), "get_weather", {"city": "London"})


def test_frag_function_name_split():
    r = feed_chunks(
        "<tool_call>\n<function=get_wea",
        "ther>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    )
    assert_tool(r, "get_weather", {"city": "Paris"})


def test_frag_close_tool_call_split():
    r = feed_chunks(
        "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n</too",
        "l_call>",
    )
    assert_tool(r, "f", {"x": "v"})


def test_frag_lt_of_html_tag_is_last_char():
    r = feed_chunks(
        "<tool_call>\n<function=f>\n<parameter=h>\n<",
        "b>bold</b>\n</parameter>\n</function>\n</tool_call>",
    )
    assert_tool(r, "f", {"h": "<b>bold</b>"})


# ===========================================================================
# Section 3 – HTML / XML content inside parameter values
# ===========================================================================


def test_html_simple_tag():
    assert_tool(
        feed_char_by_char(
            "<tool_call>\n<function=f>\n<parameter=h>\n<b>bold</b>\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"h": "<b>bold</b>"},
    )


def test_html_nested():
    html = "<div><p>Hello <em>world</em></p></div>"
    assert_tool(
        feed_char_by_char(
            f"<tool_call>\n<function=f>\n<parameter=h>\n{html}\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"h": html},
    )


def test_html_tag_with_lt_in_attribute():
    html = '<img src="photo.jpg" alt="A < B">'
    assert_tool(
        feed_char_by_char(
            f"<tool_call>\n<function=f>\n<parameter=h>\n{html}\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"h": html},
    )


def test_html_self_closing():
    content = "line1<br/>line2<br/>line3"
    assert_tool(
        feed_char_by_char(
            f"<tool_call>\n<function=f>\n<parameter=t>\n{content}\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"t": content},
    )


def test_html_xml_comment():
    content = "before<!-- comment -->after"
    assert_tool(
        feed_char_by_char(
            f"<tool_call>\n<function=f>\n<parameter=c>\n{content}\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"c": content},
    )


# ===========================================================================
# Section 4 – Tags that look like structural tags but aren't
# ===========================================================================


@pytest.mark.parametrize(
    "tag",
    [
        "</param>",  # prefix of </parameter>
        "</func>",  # prefix of </function>
        "</tool>",  # prefix of </tool_call>
        "<parameter>",  # missing =name
        "<function>",  # missing =name
        "<tool_call_x>",  # similar but not equal
        "</tool_calls>",  # longer than </tool_call>
    ],
)
def test_partial_tag_names_are_literal(tag):
    r = feed_char_by_char(
        f"<tool_call>\n<function=f>\n<parameter=v>\n{tag}\n</parameter>\n</function>\n</tool_call>"
    )
    assert json.loads(r.tool_calls[0].function.arguments) == {"v": tag}


@pytest.mark.parametrize(
    "tag",
    [
        "<tool_call>",
        "<function=other>",
        "<parameter=nested>",
    ],
)
def test_opening_structural_tags_are_literal_inside_value(tag):
    r = feed_char_by_char(
        f"<tool_call>\n<function=f>\n<parameter=v>\n{tag}\n</parameter>\n</function>\n</tool_call>"
    )
    assert json.loads(r.tool_calls[0].function.arguments) == {"v": tag}


def test_bare_lt_space_is_literal():
    assert_tool(
        feed_char_by_char(
            "<tool_call>\n<function=f>\n<parameter=expr>\na < b\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"expr": "a < b"},
    )


def test_double_lt_in_value():
    r = feed_char_by_char(
        "<tool_call>\n<function=f>\n<parameter=v>\na<<b\n</parameter>\n</function>\n</tool_call>"
    )
    assert json.loads(r.tool_calls[0].function.arguments) == {"v": "a<<b"}


def test_json_fragment_with_angle_brackets():
    content = '{"key": "<value>", "other": 1}'
    r = feed_char_by_char(
        f"<tool_call>\n<function=f>\n<parameter=j>\n{content}\n</parameter>\n</function>\n</tool_call>"
    )
    assert json.loads(r.tool_calls[0].function.arguments) == {"j": content}


# ===========================================================================
# Section 5 – Format limitations (documented, no crash)
# ===========================================================================


def test_embedded_close_parameter_ends_early_no_crash():
    """A literal </parameter> inside a value closes the parameter early.
    There is no escaping in this format; test that we don't crash and
    produce valid JSON."""
    out = (
        "<tool_call>\n<function=f>\n"
        "<parameter=code>\n"
        "x = '</parameter>'\n"
        "</parameter>\n"
        "</function>\n</tool_call>"
    )
    r = feed_char_by_char(out)
    assert r.tool_calls
    json.loads(r.tool_calls[0].function.arguments)  # must be valid JSON


def test_embedded_close_function_ends_early_no_crash():
    out = (
        "<tool_call>\n<function=f>\n"
        "<parameter=v>\n"
        "text</function>more\n"
        "</parameter>\n"
        "</function>\n</tool_call>"
    )
    r = feed_char_by_char(out)
    assert r.tool_calls
    json.loads(r.tool_calls[0].function.arguments)


# ===========================================================================
# Section 6 – Parameter types
# ===========================================================================


def _typed(param: str, value: str) -> dict:
    r = feed_char_by_char(
        f"<tool_call>\n<function=typed_fn>\n"
        f"<parameter={param}>\n{value}\n</parameter>\n"
        f"</function>\n</tool_call>",
        tools=TOOLS_TYPED,
    )
    return json.loads(r.tool_calls[0].function.arguments)


def test_type_integer():
    assert _typed("count", "42") == {"count": 42}


def test_type_float():
    assert abs(_typed("ratio", "3.14")["ratio"] - 3.14) < 1e-9


def test_type_float_whole_number():
    assert _typed("ratio", "2.0") == {"ratio": 2}


def test_type_boolean_true():
    assert _typed("flag", "true") == {"flag": True}


def test_type_boolean_false():
    assert _typed("flag", "false") == {"flag": False}


def test_type_boolean_capitalised():
    assert _typed("flag", "True") == {"flag": True}


def test_type_null():
    assert _typed("label", "null") == {"label": None}


def test_type_object_json():
    assert _typed("data", '{"key": "val", "n": 1}') == {"data": {"key": "val", "n": 1}}


def test_type_object_python_dict():
    assert _typed("data", "{'key': 'val', 'n': 1}") == {"data": {"key": "val", "n": 1}}


def test_type_array_json():
    assert _typed("items", '[1, "two", true]') == {"items": [1, "two", True]}


def test_type_array_python_list():
    assert _typed("items", "['a', 'b', 'c']") == {"items": ["a", "b", "c"]}


def test_type_object_with_angle_brackets():
    payload = '{"url": "https://example.com?a=1&b=<2>"}'
    args = _typed("data", payload)
    assert args["data"]["url"] == "https://example.com?a=1&b=<2>"


# ===========================================================================
# Section 7 – String edge cases
# ===========================================================================


def test_string_double_quotes():
    assert_tool(
        feed_char_by_char(
            '<tool_call>\n<function=f>\n<parameter=msg>\nShe said "hello"\n</parameter>\n</function>\n</tool_call>'
        ),
        "f",
        {"msg": 'She said "hello"'},
    )


def test_string_backslash():
    path = "C:\\Users\\alice\\file.txt"
    assert_tool(
        feed_char_by_char(
            f"<tool_call>\n<function=f>\n<parameter=path>\n{path}\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"path": path},
    )


def test_string_unicode():
    content = "日本語テスト 🎉"
    assert_tool(
        feed_char_by_char(
            f"<tool_call>\n<function=f>\n<parameter=text>\n{content}\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"text": content},
    )


def test_string_empty():
    r = feed_char_by_char(
        "<tool_call>\n<function=f>\n<parameter=v></parameter>\n</function>\n</tool_call>"
    )
    assert json.loads(r.tool_calls[0].function.arguments) == {"v": ""}


def test_string_multiline():
    content = "first line\nsecond line\nthird line"
    assert_tool(
        feed_char_by_char(
            f"<tool_call>\n<function=f>\n<parameter=text>\n{content}\n</parameter>\n</function>\n</tool_call>"
        ),
        "f",
        {"text": content},
    )


# ===========================================================================
# Section 8 – Multi-parameter and multi-call
# ===========================================================================


def test_many_parameters():
    out = (
        "<tool_call>\n<function=send_email>\n"
        "<parameter=to>\nalice@example.com\n</parameter>\n"
        "<parameter=cc>\nbob@example.com\n</parameter>\n"
        "<parameter=subject>\nHello!\n</parameter>\n"
        "<parameter=body>\nDear Alice,\nThis is a test.\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert_tool(
        feed_char_by_char(out),
        "send_email",
        {
            "to": "alice@example.com",
            "cc": "bob@example.com",
            "subject": "Hello!",
            "body": "Dear Alice,\nThis is a test.",
        },
    )


def test_two_tool_calls():
    out = (
        "<tool_call>\n<function=f1>\n<parameter=a>\nfoo\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=f2>\n<parameter=b>\nbar\n</parameter>\n</function>\n</tool_call>"
    )
    r = feed_char_by_char(out)
    assert len(r.tool_calls) == 2
    assert_tool(r, "f1", {"a": "foo"}, index=0)
    assert_tool(r, "f2", {"b": "bar"}, index=1)


def test_three_tool_calls():
    out = "".join(
        f"<tool_call>\n<function=f{i}>\n<parameter=x>\n{i}\n</parameter>\n</function>\n</tool_call>\n"
        for i in range(3)
    )
    r = feed_char_by_char(out)
    assert len(r.tool_calls) == 3
    for i in range(3):
        assert_tool(r, f"f{i}", {"x": str(i)}, index=i)


def test_text_before_tool_call():
    out = (
        "I'll check the weather for you.\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nTokyo\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    r = feed_char_by_char(out)
    assert r.content and "weather" in r.content
    assert_tool(r, "get_weather", {"city": "Tokyo"})


# ===========================================================================
# Section 9 – Syntax variants
# ===========================================================================


def test_attr_syntax_function_and_param():
    out = (
        '<tool_call>\n<function name="send">\n'
        '<parameter name="to">\nalice\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    assert_tool(feed_char_by_char(out), "send", {"to": "alice"})


# ===========================================================================
# Section 10 – Streaming delta reconstruction
# ===========================================================================


def test_streaming_incremental_deltas_reconstruct_args():
    full = (
        "<tool_call>\n<function=calc>\n"
        "<parameter=expr>\n2 + 2\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    p = StreamingXMLToolCallParser()
    accumulated_args = ""
    fn_name = None
    for ch in full:
        delta = p.parse_single_streaming_chunks(ch)
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function:
                    if tc.function.name:
                        fn_name = tc.function.name
                    if tc.function.arguments:
                        accumulated_args += tc.function.arguments
    p.finalize()
    assert fn_name == "calc"
    assert json.loads(accumulated_args) == {"expr": "2 + 2"}


def test_streaming_text_content_before_tool_call():
    full = (
        "Let me search.\n"
        "<tool_call>\n<function=search>\n<parameter=q>\ntest\n</parameter>\n</function>\n</tool_call>"
    )
    p = StreamingXMLToolCallParser()
    content_seen = []
    for ch in full:
        delta = p.parse_single_streaming_chunks(ch)
        if delta and delta.content:
            content_seen.append(delta.content)
    p.finalize()
    assert "Let me search" in "".join(content_seen)


def test_streaming_two_calls_distinct_ids():
    out = (
        "<tool_call>\n<function=f1>\n<parameter=a>\nA\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=f2>\n<parameter=b>\nB\n</parameter>\n</function>\n</tool_call>"
    )
    p = StreamingXMLToolCallParser()
    seen_ids: set[str] = set()
    for ch in out:
        delta = p.parse_single_streaming_chunks(ch)
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.id:
                    seen_ids.add(tc.id)
    p.finalize()
    assert len(seen_ids) == 2


# ===========================================================================
# Section 11 – Premature abort / resync
# ===========================================================================


def test_abort_right_after_tool_call_open():
    r = feed_char_by_char("<tool_call>\n")
    assert not r.tool_calls, f"Expected no tool_calls, got: {r.tool_calls}"


def test_abort_mid_function_tag():
    r = feed_char_by_char("<tool_call>\n<function=get_wea")
    assert not r.tool_calls, f"Expected no tool_calls, got: {r.tool_calls}"


def test_abort_after_complete_function_tag():
    r = feed_char_by_char("<tool_call>\n<function=get_weather>\n")
    assert r.tool_calls
    assert r.tool_calls[0].function.name == "get_weather"
    assert r.tool_calls[0].function.arguments == "{}"


def test_abort_mid_parameter_tag():
    r = feed_char_by_char("<tool_call>\n<function=f>\n<parameter=ci")
    assert r.tool_calls and r.tool_calls[0].function.name == "f"
    assert r.tool_calls[0].function.arguments == "{}"


def test_abort_mid_parameter_value():
    r = feed_char_by_char("<tool_call>\n<function=f>\n<parameter=city>\nLon")
    assert r.tool_calls and r.tool_calls[0].function.name == "f"
    assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Lon"}


def test_abort_no_close_tool_call():
    assert_tool(
        feed_char_by_char(
            "<tool_call>\n<function=f>\n<parameter=x>\nval\n</parameter>\n</function>"
        ),
        "f",
        {"x": "val"},
    )


@pytest.mark.parametrize(
    "partial",
    [
        "<tool_call>",
        "<tool_call>\n",
        "<tool_call>\n<function=get_wea",
        "<tool_call>\n<function=",
    ],
)
def test_collect_all_never_returns_null_name(partial):
    r = feed_char_by_char(partial)
    if r.tool_calls:
        for tc in r.tool_calls:
            assert tc.function and tc.function.name, (
                f"name=None in result for input {partial!r}"
            )


def test_resync_missing_close_tag_both_calls_correct():
    out = (
        "<tool_call>\n<function=f1>\n<parameter=a>\nA\n</parameter>\n</function>\n"
        "<tool_call>\n<function=f2>\n<parameter=b>\nB\n</parameter>\n</function>\n</tool_call>"
    )
    r = feed_char_by_char(out)
    assert len(r.tool_calls) == 2
    assert_tool(r, "f1", {"a": "A"}, index=0)
    assert_tool(r, "f2", {"b": "B"}, index=1)


def test_resync_missing_close_tag_terminator_order():
    """call_1's terminator delta must arrive before call_2's first delta."""
    out = (
        "<tool_call>\n<function=f1>\n<parameter=a>\nA\n</parameter>\n</function>\n"
        "<tool_call>\n<function=f2>\n<parameter=b>\nB\n</parameter>\n</function>\n</tool_call>"
    )
    p = StreamingXMLToolCallParser()
    events: list[tuple[str, str | None, str | None]] = []
    for ch in out:
        delta = p.parse_single_streaming_chunks(ch)
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function:
                    events.append(
                        (tc.id or "", tc.function.name, tc.function.arguments)
                    )
    p.finalize()

    call_ids: list[str] = []
    for cid, _, _ in events:
        if cid not in call_ids:
            call_ids.append(cid)
    assert len(call_ids) == 2
    call1_id, call2_id = call_ids

    term_pos = next(
        (
            i
            for i, (cid, n, a) in enumerate(events)
            if cid == call1_id and n is None and a == ""
        ),
        None,
    )
    start2_pos = next(
        (i for i, (cid, _, _) in enumerate(events) if cid == call2_id), None
    )
    assert term_pos is not None, "call_1 never received a terminator delta"
    assert term_pos < start2_pos, (
        f"call_1 terminator (pos {term_pos}) arrived after call_2 started (pos {start2_pos})"
    )


def test_reset_streaming_state_discards_partial_call():
    p = StreamingXMLToolCallParser()
    for ch in "<tool_call>\n<function=broken":
        p.parse_single_streaming_chunks(ch)
    p.reset_streaming_state()
    good = "<tool_call>\n<function=good_fn>\n<parameter=x>\nok\n</parameter>\n</function>\n</tool_call>"
    for ch in good:
        p.parse_single_streaming_chunks(ch)
    p.finalize()
    r = p.collect_all()
    assert r.tool_calls and len(r.tool_calls) == 1
    assert_tool(r, "good_fn", {"x": "ok"})


# ===========================================================================
# Section 12 - Syntax robustness (single quotes, whitespace, etc.)
# ===========================================================================


def test_attr_syntax_single_quotes():
    out = "<tool_call>\n<function name='send'>\n<parameter name='to'>\nalice\n</parameter>\n</function>\n</tool_call>"
    assert_tool(feed_char_by_char(out), "send", {"to": "alice"})


def test_attr_syntax_whitespace_before_closing_gt():
    out = '<tool_call>\n<function name="send" >\n<parameter name="to" >\nalice\n</parameter>\n</function>\n</tool_call>'
    assert_tool(feed_char_by_char(out), "send", {"to": "alice"})


def test_attr_syntax_spaces_around_equals():
    out = '<tool_call>\n<function name = "send">\n<parameter name = "to">\nalice\n</parameter>\n</function>\n</tool_call>'
    assert_tool(feed_char_by_char(out), "send", {"to": "alice"})


def test_eq_syntax_spaces_around_equals():
    out = "<tool_call>\n<function = send>\n<parameter = to>\nalice\n</parameter>\n</function>\n</tool_call>"
    assert_tool(feed_char_by_char(out), "send", {"to": "alice"})


# ===========================================================================
# Section 13 - Integer from float representation
# ===========================================================================


def test_type_integer_from_float_string():
    p = StreamingXMLToolCallParser()
    assert p._raw_to_json("42.0", "integer") == "42"


def test_type_integer_from_float_string_with_decimals():
    p = StreamingXMLToolCallParser()
    assert p._raw_to_json("7.5", "integer") == "7"


# ===========================================================================
# Section 14 - <tool_call> lookahead (must be followed by <function=…>)
# ===========================================================================
#
# A <tool_call> seen in plain-text / reasoning context must NOT be treated as
# a tool-call opener unless the very next structural tag is <function=…>.
# All other followers cause the <tool_call> to be rolled back as literal text.


def test_fake_tool_call_followed_by_text():
    """<tool_call> with no following structural tag: rolled back as content."""
    out = "Use the <tool_call> syntax to call tools."
    result = feed_char_by_char(out)
    assert not result.tool_calls
    assert result.content == out


def test_fake_tool_call_followed_by_closing_tag():
    """<tool_call>...</tool_call> without <function=…>: rolled back as content."""
    out = "<tool_call>some text</tool_call>"
    result = feed_char_by_char(out)
    assert not result.tool_calls
    assert result.content == out


def test_fake_tool_call_followed_by_another_tool_call_then_real():
    """Two <tool_call> mentions; only the one followed by <function=…> is real."""
    text_before = "You can nest <tool_call>\n<tool_call>\n"
    real = "<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"
    out = text_before + real
    result = feed_char_by_char(out)
    assert_tool(result, "bash", {"cmd": "ls"})
    assert result.content == "You can nest <tool_call>\n"


def test_fake_tool_call_then_real_tool_call():
    """Fake <tool_call> in reasoning text, then a real tool call after."""
    fake = "Thinking: use <tool_call> here.\n"
    real = (
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\n/etc/hosts\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = feed_char_by_char(fake + real)
    assert_tool(result, "read_file", {"path": "/etc/hosts"})
    assert result.content == fake


def test_fake_tool_call_whole_chunk():
    """Same scenario fed as a single chunk instead of char-by-char."""
    fake = "Thinking: use <tool_call> here.\n"
    real = (
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\n/etc/hosts\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = feed_chunks(fake + real)
    assert_tool(result, "read_file", {"path": "/etc/hosts"})
    assert result.content == fake


def test_real_tool_call_still_works_with_lookahead():
    """Normal tool call path still works correctly after adding lookahead."""
    result = feed_char_by_char(BASIC)
    assert_tool(result, "get_weather", {"city": "London"})
    assert not result.content


def test_tool_call_pending_stream_boundary():
    """<tool_call> in chunk 1, <function=…> in chunk 2: confirmed as real."""
    result = feed_chunks(
        "<tool_call>\n",
        "<function=bash>\n<parameter=cmd>\necho hi\n</parameter>\n</function>\n</tool_call>",
    )
    assert_tool(result, "bash", {"cmd": "echo hi"})
    assert not result.content


def test_tool_call_pending_rollback_stream_boundary():
    """<tool_call> in chunk 1, non-function tag in chunk 2: rolled back."""
    result = feed_chunks(
        "<tool_call>\n",
        "</tool_call> and more text",
    )
    assert not result.tool_calls
    assert result.content == "<tool_call>\n</tool_call> and more text"


def test_multiple_fake_tool_calls_all_rolled_back():
    """Three fake <tool_call> mentions with no <function=…>: all become content."""
    out = (
        "Step 1: <tool_call>\n"
        "Step 2: <tool_call>\n"
        "Step 3: <tool_call>\n"
        "Done."
    )
    result = feed_char_by_char(out)
    assert not result.tool_calls
    assert result.content == out


def test_tool_call_pending_truncated_stream():
    """Stream ends while in TOOL_CALL_PENDING: <tool_call> is flushed as content."""
    result = feed_chunks("<tool_call>\n")
    assert not result.tool_calls
    assert result.content == "<tool_call>\n"


def test_tool_call_pending_stale_tag_rollback():
    """A second '<' arrives while reading a tag in TOOL_CALL_PENDING."""
    # "<tool_call>\n<fun" then "<function=bash>..." - the stale "<fun" is rolled
    # back together with the whole <tool_call> preamble as plain text content.
    # Without implicit start, the orphaned <function=bash> in TEXT state is also
    # treated as plain text — no tool call is produced from a corrupted stream.
    result = feed_chunks(
        "<tool_call>\n<fun",
        "<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>",
    )
    assert not result.tool_calls, (
        f"Corrupted stream should produce no tool calls, got {result.tool_calls}"
    )


def test_tool_call_with_content_before():
    """Plain-text content before a real tool call is preserved as content."""
    preamble = "Sure, let me do that.\n"
    real = (
        "<tool_call>\n"
        "<function=echo>\n"
        "<parameter=msg>\nhello\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = feed_char_by_char(preamble + real)
    assert_tool(result, "echo", {"msg": "hello"})
    assert result.content == preamble
