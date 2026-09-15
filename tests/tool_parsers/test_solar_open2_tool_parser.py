# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
from openai.types.responses.function_tool import FunctionTool

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.solar_open2_tool_parser import SolarOpen2ToolParser

# Shape produced by make_tool_call_id() for the default "random" id type.
_TOOL_CALL_ID_PREFIX = "chatcmpl-tool-"

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def mock_tokenizer():
    # The parser works purely on text; it never consults the vocabulary.
    return SimpleNamespace(get_vocab=lambda: {})


@pytest.fixture
def parser(mock_tokenizer):
    return SolarOpen2ToolParser(mock_tokenizer)


@pytest.fixture
def typed_request():
    """ChatCompletionRequest with a tools schema covering every coercion path."""
    return ChatCompletionRequest(
        model="solar-open2",
        messages=[],
        tools=[
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                    },
                },
            ),
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "do_all_types",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "s": {"type": "string"},
                            "i": {"type": "integer"},
                            "n": {"type": "number"},
                            "b": {"type": "boolean"},
                            "arr": {"type": "array"},
                            "obj": {"type": "object"},
                            "maybe_int": {"type": ["integer", "null"]},
                        },
                    },
                },
            ),
        ],
    )


@pytest.fixture
def responses_typed_request():
    """ResponsesRequest using the flat FunctionTool schema shape."""
    return ResponsesRequest(
        model="solar-open2",
        input="Use the requested tool.",
        tools=[
            FunctionTool(
                type="function",
                name="set_options",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                    },
                },
                strict=True,
            )
        ],
    )


class TestExtractToolCalls:
    """Non-streaming tool call extraction tests."""

    def test_no_tool_calls(self, parser):
        model_output = "This is a regular response."
        result = parser.extract_tool_calls(model_output, None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a regular response."

    def test_single_tool_no_content(self, parser):
        model_output = (
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].id.startswith(_TOOL_CALL_ID_PREFIX)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Seoul"}
        assert result.content is None

    def test_single_tool_with_content_prefix(self, parser):
        model_output = (
            "Let me check the weather."
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.content == "Let me check the weather."

    def test_multiple_tool_calls(self, parser):
        model_output = (
            "I'll check both."
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>\n"
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Tokyo<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Seoul"}
        assert result.tool_calls[1].function.name == "get_weather"
        assert json.loads(result.tool_calls[1].function.arguments) == {"city": "Tokyo"}
        assert result.content == "I'll check both."

    def test_multiple_arguments_without_schema(self, parser):
        """With no schema to consult, values fall back to strings."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>vLLM tutorial<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"query": "vLLM tutorial", "max_results": "5"}

    def test_multiple_arguments_with_schema(self, parser, typed_request):
        """With a schema, ``integer`` args are coerced to int."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>vLLM tutorial<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"query": "vLLM tutorial", "max_results": 5}

    def test_no_arguments(self, parser):
        model_output = "<|tool_call:start|>get_time\n<|tool_call:end|>"
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_time"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_json_like_value_in_arg(self, parser):
        model_output = (
            "<|tool_call:start|>process_data\n"
            '<|tool_arg:start|>data<|tool_arg:value|>{"key": "value"}<|tool_arg:end|>\n'
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"data": '{"key": "value"}'}

    def test_reasoning_tags_with_tool_calls(self, parser):
        """Content is everything before the first tool call, reasoning included."""
        model_output = (
            "<|think:start|>I should use a tool<|think:end|>"
            "Sure, let me help."
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.content == (
            "<|think:start|>I should use a tool<|think:end|>Sure, let me help."
        )

    def test_whitespace_only_prefix_is_no_content(self, parser):
        """A whitespace-only prefix is reported as no content."""
        model_output = (
            "  \n\n"
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert result.content is None

    def test_text_after_last_tool_call_is_not_content(self, parser):
        """Content stops at the first tool call, so trailing text is dropped."""
        model_output = (
            "pre"
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "post"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.content == "pre"

    def test_unterminated_block_returns_raw_output(self, parser, typed_request):
        """A block with no ``<|tool_call:end|>`` must not discard the output."""
        model_output = (
            "Let me check."
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Seo"
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == model_output

    def test_dropped_arg_end_does_not_swallow_later_calls(self, parser):
        """A missing ``<|tool_arg:end|>`` must not consume the next call."""
        model_output = (
            "<|tool_call:start|>broken\n"
            "<|tool_arg:start|>a<|tool_arg:value|>oops"
            "<|tool_call:end|>"
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert [tc.function.name for tc in result.tool_calls] == [
            "broken",
            "get_weather",
        ]
        assert json.loads(result.tool_calls[0].function.arguments) == {"a": "oops"}
        assert json.loads(result.tool_calls[1].function.arguments) == {"city": "Seoul"}

    @pytest.mark.parametrize(
        "fixture", ["dropped_call_end", "dropped_call_end_and_arg_end"]
    )
    def test_dropped_call_end_recovers_both_calls(self, parser, fixture):
        """A missing ``<|tool_call:end|>`` must not drop the unterminated call."""
        result = parser.extract_tool_calls(_PARITY_FIXTURES[fixture], None)
        assert [tc.function.name for tc in result.tool_calls] == [
            "search",
            "get_weather",
        ]
        assert json.loads(result.tool_calls[0].function.arguments) == {"query": "vLLM"}
        assert json.loads(result.tool_calls[1].function.arguments) == {"city": "Seoul"}

    @pytest.mark.parametrize(
        "fixture",
        [
            "restart_before_function_name",
            "restart_before_function_name_with_prefix",
            "call_end_before_function_name",
        ],
    )
    def test_boundary_before_function_name_skips_the_block(self, parser, fixture):
        """A call whose name never terminates is skipped, not read across."""
        result = parser.extract_tool_calls(_PARITY_FIXTURES[fixture], None)
        assert [tc.function.name for tc in result.tool_calls] == ["get_weather"]
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Seoul"}

    def test_function_name_stops_at_the_first_newline(self, parser):
        """The name is the rest of the start line, so it holds no newline."""
        match = parser.tool_call_pattern.search(_PARITY_FIXTURES["empty_function_name"])
        assert match is not None
        assert match.group(1) == ""

    def test_empty_function_name_is_parsed_as_empty(self, parser):
        """``<|tool_call:start|>`` then a newline yields an empty name, as streamed."""
        result = parser.extract_tool_calls(
            _PARITY_FIXTURES["empty_function_name"], None
        )
        assert [tc.function.name for tc in result.tool_calls] == [""]
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Seoul"}

    def test_truncated_multiline_value_returns_raw_output(self, parser, typed_request):
        """A value cut off mid-way stays raw output however many lines it spans."""
        model_output = (
            "Writing the file."
            "<|tool_call:start|>write_file\n"
            "<|tool_arg:start|>path<|tool_arg:value|>/tmp/a.py<|tool_arg:end|>\n"
            "<|tool_arg:start|>content<|tool_arg:value|>"
            + "".join(f"line {i}\n" for i in range(200))
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == model_output

    @pytest.mark.parametrize(
        "fixture",
        [
            "call_end_before_arg_name",
            "restart_before_arg_name",
            "arg_start_before_arg_name",
        ],
    )
    def test_boundary_before_arg_name_skips_the_argument(self, parser, fixture):
        """An argument name that runs into a boundary contributes no key."""
        result = parser.extract_tool_calls(_PARITY_FIXTURES[fixture], None)
        args = [json.loads(tc.function.arguments) for tc in result.tool_calls]
        assert all("ci" not in a for a in args)
        assert args[-1] == {"city": "Seoul"}

    def test_literal_call_sentinel_prefix_in_value(self, parser):
        """A value may contain ``<|tool_call:``; only a full sentinel is a boundary."""
        result = parser.extract_tool_calls(
            _PARITY_FIXTURES["literal_call_sentinel_in_value"], None
        )
        assert [tc.function.name for tc in result.tool_calls] == ["search"]
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "query": _LITERAL_SENTINEL_VALUE
        }
        assert result.content is None

    def test_repeated_argument_name_keeps_first_value(self, parser):
        """Repeated names collapse to the first value, as a stream must."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>first<|tool_arg:end|>\n"
            "<|tool_arg:start|>query<|tool_arg:value|>second<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tool_calls[0].function.arguments == '{"query": "first"}'


class TestTypeCoercion:
    """Schema-driven type coercion for tool call arguments."""

    def _build(self, arg_name: str, arg_value: str) -> str:
        return (
            "<|tool_call:start|>do_all_types\n"
            f"<|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )

    def test_integer(self, parser, typed_request):
        result = parser.extract_tool_calls(self._build("i", "42"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"i": 42}

    def test_number_float(self, parser, typed_request):
        result = parser.extract_tool_calls(self._build("n", "3.14"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"n": 3.14}

    def test_number_downcasts_to_int_when_fractional_is_zero(
        self, parser, typed_request
    ):
        result = parser.extract_tool_calls(self._build("n", "7.0"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"n": 7}

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("0", False),
            # ``yes``/``no`` are not JSON booleans and stay strings.
            ("yes", "yes"),
            ("no", "no"),
        ],
    )
    def test_boolean(self, parser, typed_request, raw, expected):
        result = parser.extract_tool_calls(self._build("b", raw), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"b": expected}

    def test_array(self, parser, typed_request):
        result = parser.extract_tool_calls(
            self._build("arr", "[1, 2, 3]"), typed_request
        )
        assert json.loads(result.tool_calls[0].function.arguments) == {"arr": [1, 2, 3]}

    def test_object(self, parser, typed_request):
        result = parser.extract_tool_calls(
            self._build("obj", '{"k": "v"}'), typed_request
        )
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "obj": {"k": "v"}
        }

    def test_union_with_null_prefers_non_null_type(self, parser, typed_request):
        """``["integer", "null"]`` should still coerce numeric values to int."""
        result = parser.extract_tool_calls(self._build("maybe_int", "9"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"maybe_int": 9}

    def test_literal_null_becomes_none_for_non_string_param(
        self, parser, typed_request
    ):
        """``"null"`` yields ``None`` where the declared type cannot hold it."""
        result = parser.extract_tool_calls(self._build("i", "null"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"i": None}

    def test_string_type_preserves_raw_value(self, parser, typed_request):
        result = parser.extract_tool_calls(self._build("s", "42"), typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"s": "42"}

    def test_conversion_failure_falls_back_to_string(self, parser, typed_request):
        """Malformed numeric input returns the raw string, does not raise."""
        result = parser.extract_tool_calls(
            self._build("i", "not-a-number"), typed_request
        )
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "i": "not-a-number"
        }

    def test_unknown_function_falls_back_to_string(self, parser, typed_request):
        """A hallucinated function name has no schema, so values stay strings."""
        model_output = (
            "<|tool_call:start|>unknown_fn\n"
            "<|tool_arg:start|>x<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"x": "5"}

    def test_unknown_param_falls_back_to_string(self, parser, typed_request):
        """A parameter absent from ``properties`` has no type, so it stays a string."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>undeclared<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, typed_request)
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "undeclared": "5"
        }

    def test_responses_function_tool_schema(self, parser, responses_typed_request):
        """Responses API's flat FunctionTool schema drives batch coercion."""
        model_output = (
            "<|tool_call:start|>set_options\n"
            "<|tool_arg:start|>count<|tool_arg:value|>7<|tool_arg:end|>\n"
            "<|tool_arg:start|>enabled<|tool_arg:value|>true<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, responses_typed_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"count": 7, "enabled": True}
        assert isinstance(args["count"], int)
        assert isinstance(args["enabled"], bool)

    def test_malformed_properties_fall_back_to_string(self, parser):
        """A non-object ``properties`` value must not fail the request."""
        request = ChatCompletionRequest(
            model="solar-open2",
            messages=[],
            tools=[
                ChatCompletionToolsParam(
                    type="function",
                    function={
                        "name": "malformed",
                        "parameters": {"type": "object", "properties": ["count"]},
                    },
                )
            ],
        )
        model_output = (
            "<|tool_call:start|>malformed\n"
            "<|tool_arg:start|>count<|tool_arg:value|>7<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        result = parser.extract_tool_calls(model_output, request)
        assert json.loads(result.tool_calls[0].function.arguments) == {"count": "7"}


def _run_stream(parser, model_output, request, *, chunk_size=1):
    """Stream ``model_output`` in fixed-size chunks and return
    ``(content, {index: {"name", "arguments", "id"}})`` rebuilt from the deltas.
    """
    assembled_content = ""
    tool_calls: dict[int, dict] = {}
    previous_text = ""
    previous_token_ids: list[int] = []
    for i in range(0, len(model_output), chunk_size):
        delta_text = model_output[i : i + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=previous_token_ids,
            delta_token_ids=[],
            request=request,
        )
        previous_text = current_text
        if delta is None:
            continue
        if delta.content:
            assembled_content += delta.content
        for tc in delta.tool_calls or []:
            slot = tool_calls.setdefault(
                tc.index, {"name": None, "arguments": "", "id": None}
            )
            if tc.id is not None:
                slot["id"] = tc.id
            fn = tc.function
            if fn is not None:
                if fn.name is not None:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["arguments"] += fn.arguments
    return assembled_content, tool_calls


def _assert_stream_matches_non_stream(parser, model_output, request, chunk_size):
    """Assert the streamed message equals the batched one for ``model_output``."""
    non_stream = SolarOpen2ToolParser(parser.model_tokenizer).extract_tool_calls(
        model_output, request
    )
    content, calls = _run_stream(parser, model_output, request, chunk_size=chunk_size)
    assert (content or None) == non_stream.content
    assert len(calls) == len(non_stream.tool_calls)
    for index, expected in enumerate(non_stream.tool_calls):
        assert calls[index]["name"] == expected.function.name
        assert calls[index]["arguments"] == expected.function.arguments
    return non_stream, content, calls


_TOOL_CALL = (
    "<|tool_call:start|>get_weather\n"
    "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
    "<|tool_call:end|>"
)

# Fixtures both parse paths must agree on, byte for byte.
_PARITY_FIXTURES = {
    "plain_text": "Hello! I cannot help with that.",
    "text_then_call": "I'll check the weather. " + _TOOL_CALL,
    "reasoning_then_call": (
        "<|think:start|>I should use a tool<|think:end|>Sure, let me help." + _TOOL_CALL
    ),
    "leading_newline": "\n" + _TOOL_CALL,
    "trailing_newline_before_call": "Let me check.\n" + _TOOL_CALL,
    "text_after_last_call": _TOOL_CALL + "After the call.",
    "text_between_calls": "pre" + _TOOL_CALL + "mid" + _TOOL_CALL + "post",
    "whitespace_between_calls": _TOOL_CALL + "\n\n" + _TOOL_CALL,
    "back_to_back_calls": _TOOL_CALL + _TOOL_CALL,
    "no_args": "<|tool_call:start|>get_time\n<|tool_call:end|>",
    "all_types": (
        "<|tool_call:start|>do_all_types\n"
        "<|tool_arg:start|>i<|tool_arg:value|>42<|tool_arg:end|>\n"
        "<|tool_arg:start|>n<|tool_arg:value|>7.0<|tool_arg:end|>\n"
        "<|tool_arg:start|>b<|tool_arg:value|>true<|tool_arg:end|>\n"
        "<|tool_arg:start|>arr<|tool_arg:value|>[1, 2]<|tool_arg:end|>\n"
        '<|tool_arg:start|>obj<|tool_arg:value|>{"k": "v"}<|tool_arg:end|>\n'
        "<|tool_arg:start|>s<|tool_arg:value|>hello<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    ),
    "duplicate_arg_name": (
        "<|tool_call:start|>search\n"
        "<|tool_arg:start|>query<|tool_arg:value|>first<|tool_arg:end|>\n"
        "<|tool_arg:start|>query<|tool_arg:value|>second<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    ),
    "dropped_arg_end": (
        "<|tool_call:start|>broken\n"
        "<|tool_arg:start|>a<|tool_arg:value|>oops"
        "<|tool_call:end|>" + _TOOL_CALL
    ),
    "dropped_call_end": (
        "<|tool_call:start|>search\n"
        "<|tool_arg:start|>query<|tool_arg:value|>vLLM<|tool_arg:end|>\n" + _TOOL_CALL
        # Deliberately no <|tool_call:end|> before the next call starts.
    ),
    "dropped_call_end_and_arg_end": (
        "<|tool_call:start|>search\n"
        "<|tool_arg:start|>query<|tool_arg:value|>vLLM" + _TOOL_CALL
    ),
    # A call boundary before the function name's newline: the block can never
    # become a call, so both paths must skip it and resync on the boundary.
    "restart_before_function_name": "<|tool_call:start|>get_weat" + _TOOL_CALL,
    "restart_before_function_name_with_prefix": (
        "Hmm. <|tool_call:start|>oops" + _TOOL_CALL
    ),
    "call_end_before_function_name": (
        "<|tool_call:start|>oops<|tool_call:end|>" + _TOOL_CALL
    ),
    # The function name is the rest of the start sentinel's line, so a start
    # sentinel followed straight by a newline names an empty function.
    "empty_function_name": (
        "<|tool_call:start|>\n"
        "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    ),
    # An argument name ends at <|tool_arg:value|>; a call or argument boundary
    # before it means the argument never completes and must be skipped.
    "call_end_before_arg_name": (
        "<|tool_call:start|>get_time\n<|tool_arg:start|>ci<|tool_call:end|>"
        + _TOOL_CALL
    ),
    "restart_before_arg_name": (
        "<|tool_call:start|>get_time\n<|tool_arg:start|>ci" + _TOOL_CALL
    ),
    "arg_start_before_arg_name": (
        "<|tool_call:start|>get_weather\n"
        "<|tool_arg:start|>ci"
        "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    ),
    # The boundary sits at the very start of the argument name, so the resync
    # rewinds by nothing and has to make progress from the state handoff alone.
    "boundary_at_arg_name_start": (
        "<|tool_call:start|>get_time\n<|tool_arg:start|><|tool_call:end|>" + _TOOL_CALL
    ),
    # Models quote their own markup; a bare sentinel prefix is not a boundary.
    "literal_call_sentinel_in_value": (
        "<|tool_call:start|>search\n"
        "<|tool_arg:start|>query<|tool_arg:value|>"
        "what does <|tool_call: mean<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    ),
}

# Argument value of the ``literal_call_sentinel_in_value`` fixture.
_LITERAL_SENTINEL_VALUE = "what does <|tool_call: mean"


class TestStreaming:
    """Incremental streaming extraction for solar_open2 tool calls."""

    @pytest.mark.parametrize("fixture", sorted(_PARITY_FIXTURES))
    @pytest.mark.parametrize("chunk_size", [1, 4, 1000])
    def test_parity_with_non_stream(self, parser, typed_request, fixture, chunk_size):
        """Streaming and batch parsing agree on content, call count, name and args."""
        _assert_stream_matches_non_stream(
            parser, _PARITY_FIXTURES[fixture], typed_request, chunk_size
        )

    @pytest.mark.parametrize("chunk_size", [1, 7, 1000])
    def test_responses_function_tool_schema(
        self, parser, responses_typed_request, chunk_size
    ):
        """Responses API's flat FunctionTool schema drives streaming coercion."""
        model_output = (
            "<|tool_call:start|>set_options\n"
            "<|tool_arg:start|>count<|tool_arg:value|>7<|tool_arg:end|>\n"
            "<|tool_arg:start|>enabled<|tool_arg:value|>true<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(
            parser,
            model_output,
            responses_typed_request,
            chunk_size=chunk_size,
        )
        args = json.loads(calls[0]["arguments"])
        assert args == {"count": 7, "enabled": True}
        assert isinstance(args["count"], int)
        assert isinstance(args["enabled"], bool)

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 13, 31])
    def test_various_chunk_sizes(self, parser, typed_request, chunk_size):
        """Splits in the middle of any sentinel or value yield the same args."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>i<|tool_arg:value|>7<|tool_arg:end|>\n"
            "<|tool_arg:start|>b<|tool_arg:value|>false<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(
            parser, model_output, typed_request, chunk_size=chunk_size
        )
        assert json.loads(calls[0]["arguments"]) == {"i": 7, "b": False}

    def test_content_before_tool_call_is_emitted(self, parser, typed_request):
        """Text preceding the first tool call is streamed, not swallowed."""
        model_output = (
            "I'll check the weather. "
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=4)
        assert content == "I'll check the weather. "
        assert json.loads(calls[0]["arguments"]) == {
            "query": "Seoul",
            "max_results": 3,
        }

    def test_no_tool_calls_passthrough(self, parser, typed_request):
        """Pure text output (no sentinels) streams as content only."""
        model_output = "Hello! I cannot help with that."
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert content == model_output
        assert calls == {}

    def test_trailing_partial_tool_sentinel_is_held_back(self, parser, typed_request):
        """Document the stream-only holdback when generation ends mid-sentinel."""
        model_output = "see <|to"
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        non_stream = SolarOpen2ToolParser(parser.model_tokenizer).extract_tool_calls(
            model_output, typed_request
        )

        assert content == "see "
        assert calls == {}
        assert non_stream.content == model_output
        assert non_stream.tool_calls == []

    def test_trailing_whitespace_is_streamed(self, parser, typed_request):
        """Trailing whitespace of a text-only answer must not be withheld."""
        model_output = "Hello world.\n\n"
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert content == model_output
        assert calls == {}

    def test_multiple_tool_calls(self, parser, typed_request):
        """Consecutive calls get distinct indices and independent args JSON."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Tokyo<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=5)
        assert set(calls.keys()) == {0, 1}
        assert json.loads(calls[0]["arguments"]) == {
            "query": "Seoul",
            "max_results": 2,
        }
        assert json.loads(calls[1]["arguments"]) == {
            "query": "Tokyo",
            "max_results": 3,
        }

    def test_empty_args_call(self, parser, typed_request):
        """A call with zero args streams a literal ``{}``."""
        model_output = "<|tool_call:start|>get_time\n<|tool_call:end|>"
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert calls[0]["name"] == "get_time"
        assert calls[0]["arguments"] == "{}"
        assert json.loads(calls[0]["arguments"]) == {}

    def test_malformed_value_falls_back_to_string(self, parser, typed_request):
        """A value that fails to coerce degrades to the raw string, as in batch."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>i<|tool_arg:value|>not-a-number<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=4)
        assert json.loads(calls[0]["arguments"]) == {"i": "not-a-number"}

    def test_literal_null_becomes_none(self, parser, typed_request):
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            "<|tool_arg:start|>maybe_int<|tool_arg:value|>null<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        assert json.loads(calls[0]["arguments"]) == {"maybe_int": None}

    def test_streamed_state_reset_between_streams(self, parser, typed_request):
        """A second stream must not inherit the first stream's index or buffer."""
        first = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        second = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls1 = _run_stream(parser, first, typed_request, chunk_size=3)
        _, calls2 = _run_stream(parser, second, typed_request, chunk_size=3)
        assert list(calls1.keys()) == [0]
        assert list(calls2.keys()) == [0]
        assert json.loads(calls2[0]["arguments"]) == {
            "query": "B",
            "max_results": 2,
        }

    def test_value_containing_left_angle_bracket(self, parser, typed_request):
        """A ``<`` byte inside a value is not a sentinel and must survive."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>a<b<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert json.loads(calls[0]["arguments"]) == {
            "query": "a<b",
            "max_results": 1,
        }

    def test_reasoning_tags_before_tool_call(self, parser, typed_request):
        """A ``<|`` prefix that is not a tool sentinel must not stay held back."""
        model_output = (
            "<|think:start|>I should use a tool<|think:end|>"
            "Sure, let me help."
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        assert content == (
            "<|think:start|>I should use a tool<|think:end|>Sure, let me help."
        )
        assert json.loads(calls[0]["arguments"]) == {
            "query": "Seoul",
            "max_results": 3,
        }

    def test_content_between_tool_calls_is_dropped(self, parser, typed_request):
        """Text after the first call is not content, and both calls survive."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "Also checking..."
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert content == ""
        assert set(calls.keys()) == {0, 1}
        assert json.loads(calls[0]["arguments"]) == {"query": "A", "max_results": 1}
        assert json.loads(calls[1]["arguments"]) == {"query": "B", "max_results": 2}

    def test_empty_string_value(self, parser, typed_request):
        """A zero-length value round-trips as ``""``."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|><|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert json.loads(calls[0]["arguments"]) == {"query": "", "max_results": 1}

    def test_false_positive_sentinel_prefix_in_content(self, parser, typed_request):
        """Text sharing the ``<|`` prefix must not be hoarded forever."""
        model_output = (
            "<|thinking about it|> here is the plan. "
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=1)
        assert content == "<|thinking about it|> here is the plan. "
        assert json.loads(calls[0]["arguments"]) == {"query": "x", "max_results": 1}

    def test_unicode_value(self, parser, typed_request):
        """Multibyte characters survive the streaming path unchanged."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>서울 날씨<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>3<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert json.loads(calls[0]["arguments"]) == {
            "query": "서울 날씨",
            "max_results": 3,
        }

    def test_truncated_stream_before_tool_call_end(self, parser, typed_request):
        """A stream ending mid call must not raise; the closing ``}`` is absent."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            # Deliberately no <|tool_call:end|>.
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=2)
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"] == '{"query": "x", "max_results": 5'

    def test_dropped_arg_end_recovers_the_next_call(self, parser, typed_request):
        """A missing ``<|tool_arg:end|>`` must not wedge the state machine."""
        model_output = _PARITY_FIXTURES["dropped_arg_end"]
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert [calls[i]["name"] for i in sorted(calls)] == ["broken", "get_weather"]
        assert json.loads(calls[0]["arguments"]) == {"a": "oops"}
        assert json.loads(calls[1]["arguments"]) == {"city": "Seoul"}

    @pytest.mark.parametrize(
        "fixture", ["dropped_call_end", "dropped_call_end_and_arg_end"]
    )
    @pytest.mark.parametrize("chunk_size", [1, 3, 1000])
    def test_dropped_call_end_resyncs_on_bare_start(
        self, parser, typed_request, fixture, chunk_size
    ):
        """A bare ``<|tool_call:start|>`` closes the open call and starts a new one."""
        _, calls = _run_stream(
            parser, _PARITY_FIXTURES[fixture], typed_request, chunk_size=chunk_size
        )
        assert [calls[i]["name"] for i in sorted(calls)] == ["search", "get_weather"]
        assert json.loads(calls[0]["arguments"]) == {"query": "vLLM"}
        assert json.loads(calls[1]["arguments"]) == {"city": "Seoul"}

    @pytest.mark.parametrize(
        "fixture",
        [
            "restart_before_function_name",
            "restart_before_function_name_with_prefix",
            "call_end_before_function_name",
        ],
    )
    @pytest.mark.parametrize("chunk_size", [1, 3, 1000])
    def test_boundary_before_function_name_resyncs(
        self, parser, typed_request, fixture, chunk_size
    ):
        """A name that runs into a call boundary streams no call for that block."""
        content, calls = _run_stream(
            parser, _PARITY_FIXTURES[fixture], typed_request, chunk_size=chunk_size
        )
        assert [calls[i]["name"] for i in sorted(calls)] == ["get_weather"]
        assert json.loads(calls[0]["arguments"]) == {"city": "Seoul"}
        # Content stops at the first start sentinel even though it never parsed.
        assert content == ("Hmm. " if "with_prefix" in fixture else "")

    @pytest.mark.parametrize(
        "fixture",
        [
            "call_end_before_arg_name",
            "restart_before_arg_name",
            "arg_start_before_arg_name",
        ],
    )
    @pytest.mark.parametrize("chunk_size", [1, 3, 1000])
    def test_boundary_before_arg_name_resyncs(
        self, parser, typed_request, fixture, chunk_size
    ):
        """An argument name that runs into a boundary streams no key for it."""
        _, calls = _run_stream(
            parser, _PARITY_FIXTURES[fixture], typed_request, chunk_size=chunk_size
        )
        last = calls[max(calls)]
        assert json.loads(last["arguments"]) == {"city": "Seoul"}
        for slot in calls.values():
            assert "ci<" not in slot["arguments"]

    @pytest.mark.parametrize("chunk_size", [1, 3, 1000])
    def test_literal_call_sentinel_prefix_streams_verbatim(
        self, parser, typed_request, chunk_size
    ):
        """A ``<|tool_call:`` inside a value must round-trip byte for byte."""
        _, calls = _run_stream(
            parser,
            _PARITY_FIXTURES["literal_call_sentinel_in_value"],
            typed_request,
            chunk_size=chunk_size,
        )
        assert list(calls) == [0]
        assert json.loads(calls[0]["arguments"]) == {"query": _LITERAL_SENTINEL_VALUE}

    def test_repeated_argument_name_is_emitted_once(self, parser, typed_request):
        """A repeated name must not produce a duplicate JSON key."""
        model_output = _PARITY_FIXTURES["duplicate_arg_name"]
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert calls[0]["arguments"] == '{"query": "first"}'

    def test_delta_protocol_shape(self, parser, typed_request):
        """Only the first delta of a call carries id/type/name; index is stable."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        raw_deltas: list = []
        previous_text = ""
        for i in range(len(model_output)):
            delta_text = model_output[i]
            current_text = previous_text + delta_text
            d = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=typed_request,
            )
            previous_text = current_text
            if d is not None:
                raw_deltas.extend(d.tool_calls)

        first = raw_deltas[0]
        assert first.index == 0
        assert first.type == "function"
        assert first.id.startswith(_TOOL_CALL_ID_PREFIX)
        assert first.function is not None
        assert first.function.name == "search"
        assert first.function.arguments == ""

        for d in raw_deltas[1:]:
            assert d.index == 0
            assert d.type is None
            assert d.id is None
            assert d.function is not None
            assert d.function.name is None
            assert d.function.arguments is not None and d.function.arguments != ""

    def test_base_class_state_after_stream(self, parser, typed_request):
        """``prev_tool_call_arr`` and ``streamed_args_for_tool`` mirror the deltas."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>1<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>2<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert len(parser.prev_tool_call_arr) == 2
        assert parser.prev_tool_call_arr[0]["name"] == "search"
        assert json.loads(parser.prev_tool_call_arr[0]["arguments"]) == {
            "query": "A",
            "max_results": 1,
        }
        assert parser.prev_tool_call_arr[1]["name"] == "search"
        assert json.loads(parser.prev_tool_call_arr[1]["arguments"]) == {
            "query": "B",
            "max_results": 2,
        }
        for i in (0, 1):
            assert parser.streamed_args_for_tool[i] == calls[i]["arguments"], (
                f"tool {i}: stream tracking diverged from emitted deltas"
            )
        assert parser.current_tool_id == 1  # zero-indexed, two calls emitted

    def test_streaming_without_tools_schema(self, parser):
        """Without ``request.tools`` there is no schema, so values stay strings."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>vLLM<|tool_arg:end|>\n"
            "<|tool_arg:start|>max_results<|tool_arg:value|>5<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, None, chunk_size=4)
        assert json.loads(calls[0]["arguments"]) == {
            "query": "vLLM",
            "max_results": "5",
        }

    def test_back_to_back_tool_calls_no_separator(self, parser, typed_request):
        """``<|tool_call:end|><|tool_call:start|>`` yields two distinct calls."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        content, calls = _run_stream(parser, model_output, typed_request, chunk_size=4)
        assert content == ""
        assert set(calls.keys()) == {0, 1}
        assert json.loads(calls[0]["arguments"]) == {"query": "A"}
        assert json.loads(calls[1]["arguments"]) == {"query": "B"}

    def test_zero_args_call_followed_by_args_call(self, parser, typed_request):
        """The first-arg flag resets per call, so the second opens with ``{``."""
        model_output = (
            "<|tool_call:start|>get_time\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        assert calls[0]["arguments"] == "{}"
        assert calls[1]["arguments"].startswith("{")
        assert json.loads(calls[1]["arguments"]) == {"query": "x"}

    def test_unique_ids_across_tool_calls(self, parser, typed_request):
        """Every tool call in one stream gets a distinct id."""
        model_output = (
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>A<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>B<|tool_arg:end|>\n"
            "<|tool_call:end|>"
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>C<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        ids = [calls[i]["id"] for i in sorted(calls.keys())]
        assert len(ids) == 3
        assert len(set(ids)) == 3, f"duplicate ids: {ids}"
        assert all(cid.startswith(_TOOL_CALL_ID_PREFIX) for cid in ids)

    def test_empty_delta_text_no_raise(self, parser, typed_request):
        """A delta with no visible text (e.g. a stop token) fabricates nothing."""
        previous_text = "<|tool_call:start|>search\n"
        d = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=previous_text,
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[0],
            delta_token_ids=[0],
            request=typed_request,
        )
        assert d is None or (not d.content and not d.tool_calls)

    def test_content_and_tool_name_in_same_delta(self, parser, typed_request):
        """One ``DeltaMessage`` may carry both content and a tool call."""
        model_output = (
            "prefix "
            "<|tool_call:start|>search\n"
            "<|tool_arg:start|>query<|tool_arg:value|>x<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        d = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=model_output,
            delta_text=model_output,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=typed_request,
        )
        assert d is not None
        assert d.content == "prefix "
        assert len(d.tool_calls) >= 1
        first_tc = d.tool_calls[0]
        assert first_tc.function is not None
        assert first_tc.function.name == "search"

    @pytest.mark.parametrize(
        "arg_name,raw_value,expected_py",
        [
            ("i", "42", 42),
            ("i", "null", None),
            ("i", "not-a-number", "not-a-number"),  # conversion failure fallback
            ("n", "3.14", 3.14),
            ("n", "7.0", 7),  # number → int downcast
            ("b", "true", True),
            ("b", "False", False),
            ("arr", "[1, 2, 3]", [1, 2, 3]),
            ("obj", '{"k": "v"}', {"k": "v"}),
            ("maybe_int", "9", 9),  # [integer, null] union prefers non-null
            ("s", "42", "42"),  # explicit string preserves raw
        ],
    )
    def test_parity_with_non_stream_across_all_coercions(
        self, parser, typed_request, arg_name, raw_value, expected_py
    ):
        """Every coercion path streams to the same value the batch path produces."""
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            f"<|tool_arg:start|>{arg_name}<|tool_arg:value|>{raw_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        non_stream, _, calls = _assert_stream_matches_non_stream(
            parser, model_output, typed_request, chunk_size=3
        )
        assert json.loads(non_stream.tool_calls[0].function.arguments) == {
            arg_name: expected_py
        }
        assert json.loads(calls[0]["arguments"]) == {arg_name: expected_py}

    def test_string_arg_streams_before_arg_end(self, parser, typed_request):
        """A long string must produce argument deltas before TOOL_ARG_END."""
        raw_value = "abcdefghijklmnopqrstuvwxyz" * 4
        model_output = (
            "<|tool_call:start|>search\n"
            f"<|tool_arg:start|>query<|tool_arg:value|>{raw_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        arg_end_pos = model_output.index("<|tool_arg:end|>")
        first_arg_delta_pos = None
        arg_deltas = []
        previous_text = ""

        for pos, delta_text in enumerate(model_output):
            current_text = previous_text + delta_text
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=typed_request,
            )
            previous_text = current_text
            if delta is None:
                continue
            for tool_call in delta.tool_calls or []:
                fragment = tool_call.function.arguments if tool_call.function else None
                if not fragment:
                    continue
                arg_deltas.append(fragment)
                if first_arg_delta_pos is None:
                    first_arg_delta_pos = pos

        assert first_arg_delta_pos is not None
        assert first_arg_delta_pos < arg_end_pos
        assert len(arg_deltas) > 10
        assert json.loads("".join(arg_deltas)) == {"query": raw_value}

    @pytest.mark.parametrize("raw_value", ["null", " NULL ", "n", "nu", "nul", ""])
    def test_null_and_null_prefix_strings_stay_atomic(
        self, parser, typed_request, raw_value
    ):
        """Do not open a JSON string while the value can still become null."""
        model_output = (
            "<|tool_call:start|>search\n"
            f"<|tool_arg:start|>query<|tool_arg:value|>{raw_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        arg_end_complete_pos = (
            model_output.index("<|tool_arg:end|>") + len("<|tool_arg:end|>") - 1
        )
        previous_text = ""
        emitted_before_end = []
        assembled = ""

        for pos, delta_text in enumerate(model_output):
            current_text = previous_text + delta_text
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=typed_request,
            )
            previous_text = current_text
            if delta is None:
                continue
            for tool_call in delta.tool_calls or []:
                fragment = tool_call.function.arguments if tool_call.function else None
                if not fragment:
                    continue
                assembled += fragment
                if pos < arg_end_complete_pos:
                    emitted_before_end.append(fragment)

        assert emitted_before_end == []
        # ``query`` is declared ``string``, so the null literal stays text.
        assert json.loads(assembled) == {"query": raw_value}

    @pytest.mark.parametrize(
        "raw_value",
        [
            'line1\nline2 "quoted" and \\ trailing\\',
            "서울 날씨 🙂\t\r\n다음 줄",
            "prefix <|tool_arX suffix",
            "nullish",
            "  nullable",
            "line separator \u2028 paragraph \u2029",
        ],
    )
    @pytest.mark.parametrize("chunk_size", [1, 2, 7])
    def test_partial_string_json_escaping_matches_non_stream(
        self, parser, typed_request, raw_value, chunk_size
    ):
        model_output = (
            "<|tool_call:start|>search\n"
            f"<|tool_arg:start|>query<|tool_arg:value|>{raw_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, _, calls = _assert_stream_matches_non_stream(
            parser, model_output, typed_request, chunk_size
        )
        assert json.loads(calls[0]["arguments"])["query"] == raw_value

    @pytest.mark.parametrize(
        "arg_name,raw_value",
        [
            ("i", "12345"),
            ("b", "false"),
            ("arr", "[1, 2, 3]"),
            ("obj", '{"nested": {"ok": true}}'),
        ],
    )
    def test_non_string_args_remain_atomic(
        self, parser, typed_request, arg_name, raw_value
    ):
        model_output = (
            "<|tool_call:start|>do_all_types\n"
            f"<|tool_arg:start|>{arg_name}<|tool_arg:value|>{raw_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        arg_end_complete_pos = (
            model_output.index("<|tool_arg:end|>") + len("<|tool_arg:end|>") - 1
        )
        previous_text = ""
        emitted_before_end = []

        for pos, delta_text in enumerate(model_output):
            current_text = previous_text + delta_text
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=typed_request,
            )
            previous_text = current_text
            if delta is None:
                continue
            for tool_call in delta.tool_calls or []:
                fragment = tool_call.function.arguments if tool_call.function else None
                if fragment and pos < arg_end_complete_pos:
                    emitted_before_end.append(fragment)

        assert emitted_before_end == []

    def test_multiple_string_args_stream_incrementally_without_schema(self, parser):
        first = "A" * 80
        second = "B" * 80
        model_output = (
            "<|tool_call:start|>custom\n"
            f"<|tool_arg:start|>first<|tool_arg:value|>{first}<|tool_arg:end|>\n"
            f"<|tool_arg:start|>second<|tool_arg:value|>{second}<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        _, calls = _run_stream(parser, model_output, None, chunk_size=1)
        arguments = calls[0]["arguments"]

        assert json.loads(arguments) == {"first": first, "second": second}
        assert parser.streamed_args_for_tool[0] == arguments
        assert parser.prev_tool_call_arr[0]["arguments"] == arguments

    def test_truncated_mid_string_keeps_emitted_state_consistent(
        self, parser, typed_request
    ):
        raw_value = "print('hello')\n" * 20
        model_output = (
            "<|tool_call:start|>search\n"
            f"<|tool_arg:start|>query<|tool_arg:value|>{raw_value}"
            # Deliberately no TOOL_ARG_END or TOOL_CALL_END.
        )
        _, calls = _run_stream(parser, model_output, typed_request, chunk_size=3)
        arguments = calls[0]["arguments"]

        assert arguments.startswith('{"query": "')
        assert not arguments.endswith('"')
        assert parser._stream_buffer == ""
        assert parser.streamed_args_for_tool[0] == arguments
        assert parser.prev_tool_call_arr[0]["arguments"] == arguments

    def test_streamed_string_does_not_retain_full_raw_buffer(
        self, parser, typed_request
    ):
        raw_value = "0123456789abcdef" * 512
        model_output = (
            "<|tool_call:start|>search\n"
            f"<|tool_arg:start|>query<|tool_arg:value|>{raw_value}"
            "<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        previous_text = ""
        max_value_buffer = 0

        for start in range(0, len(model_output), 7):
            delta_text = model_output[start : start + 7]
            current_text = previous_text + delta_text
            parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=typed_request,
            )
            previous_text = current_text
            if parser._stream_state == parser._STATE_READING_ARG_VALUE:
                max_value_buffer = max(max_value_buffer, len(parser._stream_buffer))

        assert max_value_buffer < len(parser.TOOL_ARG_END)
        assert json.loads(parser.streamed_args_for_tool[0]) == {"query": raw_value}


class TestStreamingWhitespaceNormalization:
    """Content is the text before the first tool call and a whitespace-only
    prefix carries none, identically on both paths."""

    TOOL = (
        "<|tool_call:start|>get_weather\n"
        "<|tool_arg:start|>location<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    )

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_ws_only_before_tool_call_is_dropped(
        self, parser, typed_request, chunk_size
    ):
        non_stream, content, _ = _assert_stream_matches_non_stream(
            parser, "\n" + self.TOOL, typed_request, chunk_size
        )
        assert content == ""
        assert non_stream.content is None
        assert len(non_stream.tool_calls) == 1

    @pytest.mark.parametrize("chunk_size", [1, 4, 1000])
    def test_ws_between_tool_calls_is_dropped(self, parser, typed_request, chunk_size):
        non_stream, content, _ = _assert_stream_matches_non_stream(
            parser, self.TOOL + "\n\n" + self.TOOL, typed_request, chunk_size
        )
        assert content == ""
        assert len(non_stream.tool_calls) == 2

    @pytest.mark.parametrize("chunk_size", [1, 5, 1000])
    def test_trailing_ws_after_text_before_first_call_is_kept(
        self, parser, typed_request, chunk_size
    ):
        non_stream, content, _ = _assert_stream_matches_non_stream(
            parser, "Let me check.\n" + self.TOOL, typed_request, chunk_size
        )
        assert content == "Let me check.\n"
        assert non_stream.content == "Let me check.\n"
