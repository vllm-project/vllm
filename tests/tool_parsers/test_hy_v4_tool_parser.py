# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HYV4 tool call parser.

The extractor takes a plain ``{token: id}`` vocab, so these run offline against
a synthetic vocab instead of downloading a checkpoint.
"""

import json

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers.hy_v4_tool_parser import (
    HYV4ToolExtractor,
    HYV4ToolParser,
    detect_token_suffix,
)
from vllm.tool_parsers.structural_tag_registry import (
    VLLM_BUILTIN_STRUCTURAL_TAG_MODELS,
    get_model_structural_tag,
)

SUFFIX = ":opensource"

TOOLS = [
    ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="get_weather",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        ),
    ),
    ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="get_current_date",
            parameters={"type": "object", "properties": {}},
        ),
    ),
]
PLAIN_TOOLS = [
    {"name": t.function.name, "parameters": t.function.parameters} for t in TOOLS
]

# Exercises the schema-driven value coercion, including non-standard type names
# the model's schema may carry.
TYPED_TOOL = [
    {
        "name": "typed",
        "parameters": {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "b": {"type": "boolean"},
                "n": {"type": "number"},
                "arr": {"type": "array"},
                "obj": {"type": "object"},
                "alias_int": {"type": "int"},
                "alias_list": {"type": "list"},
                "prefix_uint": {"type": "uint32"},
                "union": {"type": ["string", "object"]},
                "any_of": {"anyOf": [{"type": "string"}, {"type": "array"}]},
                "one_of": {"oneOf": [{"type": "number"}]},
                "untyped": {},
            },
        },
    }
]


def vocab(suffix: str = SUFFIX) -> dict[str, int]:
    tokens = ["think", "tool_calls", "tool_call", "arg_key", "arg_value"]
    return {
        tag: i
        for i, tag in enumerate(
            [f"<{t}{suffix}>" for t in tokens] + [f"</{t}{suffix}>" for t in tokens]
        )
    }


class FakeTokenizer:
    def __init__(self, suffix: str = SUFFIX):
        self._vocab = vocab(suffix)
        self.init_kwargs: dict = {}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


@pytest.fixture
def extractor() -> HYV4ToolExtractor:
    return HYV4ToolExtractor(vocab(), SUFFIX, strict=True)


@pytest.fixture
def lenient() -> HYV4ToolExtractor:
    return HYV4ToolExtractor(vocab(), SUFFIX, strict=False)


def arg(key: str, value: str) -> str:
    return (
        f"<arg_key{SUFFIX}>{key}</arg_key{SUFFIX}>"
        f"<arg_value{SUFFIX}>{value}</arg_value{SUFFIX}>"
    )


def call(name: str, args: str = "") -> str:
    return f"<tool_call{SUFFIX}>{name}{args}</tool_call{SUFFIX}>"


def block(body: str) -> str:
    return f"<tool_calls{SUFFIX}>{body}</tool_calls{SUFFIX}>"


def run_stream(
    extractor: HYV4ToolExtractor,
    chunks: list[str],
    tools: list[dict] | None = None,
) -> tuple[str, dict[int, str], dict[int, str]]:
    """Feed ``chunks`` through the streaming path and reassemble the deltas."""
    previous, content = "", ""
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for chunk in chunks:
        current = previous + chunk
        delta = extractor.extract_tool_calls_streaming(
            previous,
            current,
            chunk,
            [],
            [],
            [],
            PLAIN_TOOLS if tools is None else tools,
        )
        if delta is not None:
            content += delta["content"] or ""
            for tool_call in delta["tool_calls"]:
                index = tool_call["index"]
                if tool_call["name"]:
                    names[index] = tool_call["name"]
                if tool_call["arguments"]:
                    arguments[index] = arguments.get(index, "") + tool_call["arguments"]
        previous = current
    return content, names, arguments


class TestDetectTokenSuffix:
    """Mirrored in the reasoning parser; both must agree on the suffix."""

    def test_suffixed_tokens(self):
        assert detect_token_suffix(FakeTokenizer()) == SUFFIX

    def test_unsuffixed_tokens(self):
        assert detect_token_suffix(FakeTokenizer("")) == ""


class TestExtractorInit:
    def test_requires_tool_calls_tokens(self):
        with pytest.raises(RuntimeError, match="start/end tokens"):
            HYV4ToolExtractor({}, SUFFIX, strict=True)

    def test_token_suffix_is_exposed_for_the_structural_tag(self, extractor):
        assert extractor.token_suffix == SUFFIX


class TestArgumentValueTypes:
    """Values arrive as raw text; the tool's JSON Schema drives the coercion."""

    @staticmethod
    def parse(key: str, raw: str):
        extractor = HYV4ToolExtractor(vocab(), SUFFIX, strict=True)
        result = extractor.extract_tool_calls(
            block(call("typed", arg(key, raw))), TYPED_TOOL
        )
        assert result["tools_called"], result
        return json.loads(result["tool_calls"][0]["arguments"])[key]

    @pytest.mark.parametrize(
        "key,raw,expected",
        [
            ("s", "Beijing", "Beijing"),
            ("s", "", ""),
            ("i", "3", 3),
            ("i", "-7", -7),
            ("b", "true", True),
            ("b", "True", True),
            ("b", "false", False),
            ("n", "5", 5),
            ("n", "5.5", 5.5),
            ("n", "1e3", 1000.0),
            ("arr", '["a", "b"]', ["a", "b"]),
            ("obj", '{"k": 1}', {"k": 1}),
            # Non-standard type names: exact alias, then prefix matching.
            ("alias_int", "7", 7),
            ("alias_list", "[1]", [1]),
            ("prefix_uint", "42", 42),
            # Union / anyOf / oneOf.
            ("union", '{"a": 1}', {"a": 1}),
            ("union", "plain", "plain"),
            ("any_of", '["x"]', ["x"]),
            ("any_of", "plain", "plain"),
            ("one_of", "2.5", 2.5),
            # No declared type falls back to string.
            ("untyped", "v", "v"),
        ],
    )
    def test_value_coercion(self, key, raw, expected):
        assert self.parse(key, raw) == expected

    @pytest.mark.parametrize(
        "key,raw",
        [("i", "notanint"), ("b", "yes"), ("arr", "notjson")],
    )
    def test_unparsable_value_is_kept_as_text(self, key, raw):
        """Never drop the value: a wrong type is better than a lost argument."""
        assert self.parse(key, raw) == raw

    def test_unknown_argument_is_kept_as_text(self):
        assert self.parse("nope", "v") == "v"

    def test_unknown_tool_keeps_values_as_text(self, extractor):
        result = extractor.extract_tool_calls(
            block(call("mystery", arg("x", "1"))), PLAIN_TOOLS
        )
        assert result["tool_calls"] == [{"name": "mystery", "arguments": '{"x": "1"}'}]

    def test_no_tools_declared(self, extractor):
        result = extractor.extract_tool_calls(block(call("x", arg("a", "1"))), None)
        assert result["tool_calls"] == [{"name": "x", "arguments": '{"a": "1"}'}]


class TestExtractToolCalls:
    def test_no_tool_call(self, extractor):
        result = extractor.extract_tool_calls("A plain response.", PLAIN_TOOLS)
        assert not result["tools_called"]
        assert result["content"] == "A plain response."

    def test_values_typed_from_schema(self, extractor):
        output = "sure!" + block(
            call("get_weather", arg("city", "Beijing") + arg("days", "3"))
        )
        result = extractor.extract_tool_calls(output, PLAIN_TOOLS)
        assert result["tools_called"]
        assert result["content"] == "sure!"
        assert result["tool_calls"] == [
            {"name": "get_weather", "arguments": '{"city": "Beijing", "days": 3}'}
        ]

    def test_content_is_none_when_the_block_starts_the_output(self, extractor):
        result = extractor.extract_tool_calls(
            block(call("get_current_date")), PLAIN_TOOLS
        )
        assert result["content"] is None

    def test_zero_arg_call(self, extractor):
        result = extractor.extract_tool_calls(
            block(call("get_current_date")), PLAIN_TOOLS
        )
        assert result["tool_calls"] == [{"name": "get_current_date", "arguments": "{}"}]

    def test_multiple_calls_keep_their_order(self, extractor):
        output = block(call("get_weather", arg("city", "A")) + call("get_current_date"))
        result = extractor.extract_tool_calls(output, PLAIN_TOOLS)
        assert result["tool_calls"] == [
            {"name": "get_weather", "arguments": '{"city": "A"}'},
            {"name": "get_current_date", "arguments": "{}"},
        ]

    def test_non_ascii_values_are_not_escaped(self, extractor):
        result = extractor.extract_tool_calls(
            block(call("get_weather", arg("city", "北京"))), PLAIN_TOOLS
        )
        assert result["tool_calls"][0]["arguments"] == '{"city": "北京"}'

    def test_regex_metacharacters_in_suffix(self):
        """The suffix comes from the vocab, so it must be escaped, not compiled."""
        suffix = ":a+b(c"
        extractor = HYV4ToolExtractor(vocab(suffix), suffix, strict=True)
        output = (
            f"<tool_calls{suffix}><tool_call{suffix}>get_current_date"
            f"</tool_call{suffix}></tool_calls{suffix}>"
        )
        result = extractor.extract_tool_calls(output, PLAIN_TOOLS)
        assert result["tool_calls"] == [{"name": "get_current_date", "arguments": "{}"}]


# Malformed shapes the strict validator must reject. ``lenient`` recovers from
# all but the unbalanced <tool_call>, which no regex can pair up.
MALFORMED = {
    "missing_tool_calls_end": f"<tool_calls{SUFFIX}>" + call("get_current_date"),
    "unbalanced_tool_call": block(f"<tool_call{SUFFIX}>get_weather" + arg("city", "X")),
    "empty_function_name": block(call("")),
    "unparsed_argument_payload": block(
        f"<tool_call{SUFFIX}>get_weather"
        + arg("city", "X")
        + "JUNK"
        + f"</tool_call{SUFFIX}>"
    ),
    "unbalanced_arg_tags": block(
        f"<tool_call{SUFFIX}>get_weather<arg_key{SUFFIX}>city</arg_key{SUFFIX}>"
        f"</tool_call{SUFFIX}>"
    ),
    "end_before_start": (
        f"</tool_calls{SUFFIX}>" + call("get_current_date") + f"<tool_calls{SUFFIX}>"
    ),
}


class TestStrictMode:
    @pytest.mark.parametrize("name", sorted(MALFORMED))
    def test_strict_returns_raw_content(self, extractor, name):
        """A half-parsed tool call is worse than no tool call."""
        output = MALFORMED[name]
        result = extractor.extract_tool_calls(output, PLAIN_TOOLS)
        assert not result["tools_called"]
        assert result["content"] == output
        assert result["tool_calls"] == []

    @pytest.mark.parametrize("name", sorted(set(MALFORMED) - {"unbalanced_tool_call"}))
    def test_lenient_recovers_best_effort(self, lenient, name):
        result = lenient.extract_tool_calls(MALFORMED[name], PLAIN_TOOLS)
        assert result["tools_called"]
        assert result["tool_calls"]

    def test_lenient_still_reports_unpairable_tags(self, lenient):
        output = MALFORMED["unbalanced_tool_call"]
        result = lenient.extract_tool_calls(output, PLAIN_TOOLS)
        assert not result["tools_called"]
        assert result["content"] == output

    def test_both_modes_agree_on_well_formed_output(self, extractor, lenient):
        output = block(call("get_weather", arg("city", "A")))
        assert extractor.extract_tool_calls(
            output, PLAIN_TOOLS
        ) == lenient.extract_tool_calls(output, PLAIN_TOOLS)


class TestExtractToolCallsStreaming:
    def test_matches_non_streaming(self, extractor):
        output = "sure!" + block(
            call("get_weather", arg("city", "Beijing") + arg("days", "3"))
        )
        content, names, arguments = run_stream(extractor, list(output))
        assert content == "sure!"
        assert names == {0: "get_weather"}
        assert arguments == {0: '{"city": "Beijing", "days": 3}'}

    def test_multiple_tool_calls(self, extractor):
        output = block(
            call("get_weather", arg("city", "Beijing")) + call("get_current_date")
        )
        _, names, arguments = run_stream(extractor, list(output))
        assert names == {0: "get_weather", 1: "get_current_date"}
        assert arguments == {0: '{"city": "Beijing"}', 1: "{}"}

    def test_zero_arg_call(self, extractor):
        _, names, arguments = run_stream(
            extractor, list(block(call("get_current_date")))
        )
        assert names == {0: "get_current_date"}
        assert arguments == {0: "{}"}

    def test_whole_block_in_one_delta(self, extractor):
        """The name and the full arguments must still both be emitted."""
        output = block(call("get_weather", arg("city", "Beijing")))
        content, names, arguments = run_stream(extractor, [output])
        assert content == ""
        assert names == {0: "get_weather"}
        assert arguments == {0: '{"city": "Beijing"}'}

    def test_non_string_argument_is_not_streamed_partially(self, extractor):
        """Only pure strings stream char-by-char; an int must arrive typed."""
        output = block(call("get_weather", arg("city", "A") + arg("days", "7")))
        _, _, arguments = run_stream(extractor, list(output))
        assert json.loads(arguments[0]) == {"city": "A", "days": 7}

    def test_structural_tokens_split_across_deltas(self, extractor):
        """Guided decoding emits the tags as text, so they can be split.

        Regression: the partial tag must be held back instead of dropped, or the
        whole tool call is silently reported as content.
        """
        output = "hi" + block("\n  " + call("get_weather", arg("city", "Beijing")))
        chunks = ["hi", "<tool_call", f"s{SUFFIX}>", "\n ", " <tool_", f"call{SUFFIX}>"]
        chunks += list(output[len("".join(chunks)) :])
        content, names, arguments = run_stream(extractor, chunks)
        assert content == "hi"
        assert names == {0: "get_weather"}
        assert arguments == {0: '{"city": "Beijing"}'}

    def test_arg_value_end_tag_split_across_deltas(self, extractor):
        """Regression: a split ``</arg_value>`` must not leak into the value.

        String arguments stream incrementally, so a partial closing tag looks
        like more value text unless it is held back.
        """
        output = block(call("get_weather", arg("city", "Beijing")))
        head = (
            f"<tool_calls{SUFFIX}><tool_call{SUFFIX}>get_weather"
            f"<arg_key{SUFFIX}>city</arg_key{SUFFIX}>"
            f"<arg_value{SUFFIX}>Beijing"
        )
        chunks = [
            head,
            "</arg_",
            f"value{SUFFIX}>",
            f"</tool_call{SUFFIX}>",
            f"</tool_calls{SUFFIX}>",
        ]
        assert "".join(chunks) == output
        _, names, arguments = run_stream(extractor, chunks)
        assert names == {0: "get_weather"}
        assert arguments == {0: '{"city": "Beijing"}'}

    def test_content_only_stream_never_buffers(self, extractor):
        content, names, arguments = run_stream(extractor, ["Plain ", "answer."])
        assert content == "Plain answer."
        assert not names and not arguments

    def test_empty_delta_yields_no_text(self, extractor):
        delta = extractor.extract_tool_calls_streaming(
            "a", "a", "", [], [], [], PLAIN_TOOLS
        )
        assert delta is None or not delta["content"]

    def test_serving_state_is_updated(self, extractor):
        """The serving layer reads these off the parser after each delta."""
        output = block(call("get_weather", arg("city", "A")) + call("get_current_date"))
        run_stream(extractor, list(output))
        assert extractor.current_tool_id == 1
        assert extractor.prev_tool_call_arr == [
            {"name": "get_weather", "arguments": {"city": "A"}},
            {"name": "get_current_date", "arguments": {}},
        ]
        assert extractor.streamed_args_for_tool == ['{"city": "A"}', "{}"]


class TestStructuralTag:
    @staticmethod
    def tag(tool_choice, tools=TOOLS, reasoning: bool = False, suffix=SUFFIX):
        return get_model_structural_tag(
            model="hy_v4",
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            token_suffix=suffix,
        )

    @classmethod
    def grammar(cls, tool_choice, **kwargs) -> Grammar:
        tag = cls.tag(tool_choice, **kwargs)
        assert tag is not None
        return Grammar.from_structural_tag(tag)

    def test_registered_as_vllm_builtin(self):
        assert "hy_v4" in VLLM_BUILTIN_STRUCTURAL_TAG_MODELS

    def test_not_applied_without_tools_or_tool_choice_none(self):
        assert self.tag("auto", tools=None) is None
        assert self.tag("none") is None

    def test_auto_only_applies_to_strict_tools(self):
        """Follows the shared gate in ``get_model_structural_tag``."""
        non_strict = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_current_date", parameters={"type": "object"}
                )
            )
        ]
        assert self.tag("auto", tools=non_strict) is None

    def test_auto_keeps_plain_text_legal(self):
        assert _is_grammar_accept_string(self.grammar("auto"), "no tool needed")

    @pytest.mark.parametrize("tool_choice", ["auto", "required"])
    def test_optional_arguments_are_allowed(self, tool_choice):
        grammar = self.grammar(tool_choice)
        assert _is_grammar_accept_string(
            grammar, block(call("get_weather", arg("city", "Beijing")))
        )
        assert _is_grammar_accept_string(
            grammar,
            block(call("get_weather", arg("city", "Beijing") + arg("days", "3"))),
        )

    def test_optional_only_tool_still_constrains_key_names(self):
        """A tool with no required keys must not degrade to "anything goes".

        This is the case the optional-key branch actually decides: without it
        the tag body would be free text and any invented argument key would be
        legal.
        """
        optional_only = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="opt",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            "b": {"type": "string"},
                        },
                    },
                )
            )
        ]
        grammar = self.grammar("required", tools=optional_only)
        assert _is_grammar_accept_string(grammar, block(call("opt")))
        assert _is_grammar_accept_string(grammar, block(call("opt", arg("a", "1"))))
        # Any order, and repeats, are fine.
        assert _is_grammar_accept_string(
            grammar, block(call("opt", arg("b", "2") + arg("a", "1")))
        )
        assert not _is_grammar_accept_string(grammar, block(call("opt", arg("z", "9"))))
        assert not _is_grammar_accept_string(grammar, block(call("opt", "junk")))

    def test_known_limitation_values_can_absorb_a_following_pair(self):
        """Argument values are unquoted free text, so the grammar cannot fence
        them off: an ``<arg_value>`` body may swallow a following ``arg_key``
        pair and close on the later ``</arg_value>``.

        Pinned so nobody assumes the grammar rejects undeclared keys whenever a
        required key is present -- the parser, not the grammar, is what ignores
        them.
        """
        grammar = self.grammar("required")
        assert _is_grammar_accept_string(
            grammar, block(call("get_weather", arg("city", "A") + arg("ghost", "9")))
        )
        # Text that is not a well-formed pair still cannot slip through.
        assert not _is_grammar_accept_string(
            grammar, block(call("get_weather", arg("city", "A") + "junk"))
        )

    def test_required_rejects_missing_required_argument(self):
        assert not _is_grammar_accept_string(
            self.grammar("required"), block(call("get_weather", arg("days", "3")))
        )
        assert not _is_grammar_accept_string(self.grammar("required"), "no tool needed")

    def test_required_accepts_any_declared_tool(self):
        grammar = self.grammar("required")
        assert _is_grammar_accept_string(grammar, block(call("get_current_date")))
        assert _is_grammar_accept_string(
            grammar, block(call("get_weather", arg("city", "A")))
        )

    def test_required_rejects_undeclared_names_and_keys(self):
        grammar = self.grammar("required")
        assert not _is_grammar_accept_string(grammar, block(call("mystery")))
        assert not _is_grammar_accept_string(
            grammar, block(call("get_weather", arg("town", "A")))
        )

    def test_forced_allows_exactly_one_named_call(self):
        grammar = self.grammar(
            ChatCompletionNamedToolChoiceParam(
                type="function",
                function=ChatCompletionNamedFunction(name="get_weather"),
            )
        )
        one = call("get_weather", arg("city", "Beijing"))
        assert _is_grammar_accept_string(grammar, block(one))
        assert not _is_grammar_accept_string(grammar, block(one + one))
        assert not _is_grammar_accept_string(grammar, block(call("get_current_date")))

    def test_zero_arg_tool_needs_no_arguments(self):
        only_zero_arg = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_current_date",
                    strict=True,
                    parameters={"type": "object", "properties": {}},
                )
            )
        ]
        grammar = self.grammar("required", tools=only_zero_arg)
        assert _is_grammar_accept_string(grammar, block(call("get_current_date")))

    def test_required_key_absent_from_properties_is_still_enforced(self):
        odd = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="odd",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": ["ghost"],
                    },
                )
            )
        ]
        grammar = self.grammar("required", tools=odd)
        assert _is_grammar_accept_string(grammar, block(call("odd", arg("ghost", "1"))))
        assert not _is_grammar_accept_string(grammar, block(call("odd")))

    def test_unsuffixed_checkpoint(self):
        grammar = self.grammar("required", suffix="")
        assert _is_grammar_accept_string(
            grammar,
            "<tool_calls><tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>A</arg_value></tool_call></tool_calls>",
        )

    def test_reasoning_prefix_is_skipped(self):
        grammar = self.grammar("required", reasoning=True)
        assert _is_grammar_accept_string(
            grammar,
            f"thinking</think{SUFFIX}>" + block(call("get_weather", arg("city", "A"))),
        )

    def test_reasoning_prefix_is_not_allowed_when_disabled(self):
        """The engine drops the grammar during reasoning, so the tag covers
        only the post-think structure."""
        grammar = self.grammar("required", reasoning=False)
        assert not _is_grammar_accept_string(
            grammar,
            f"thinking</think{SUFFIX}>" + block(call("get_weather", arg("city", "A"))),
        )

    def test_xgrammar_builtin_rejects_token_suffix(self):
        with pytest.raises(ValueError, match="token_suffix"):
            get_model_structural_tag(
                model="qwen_3",
                tools=TOOLS,
                tool_choice="required",
                reasoning=False,
                token_suffix=SUFFIX,
            )


class TestToolParserWrapper:
    """The wrapper adapts plain dicts to vLLM protocol objects."""

    @pytest.fixture
    def parser(self) -> HYV4ToolParser:
        return HYV4ToolParser(FakeTokenizer(), TOOLS)

    @pytest.fixture
    def request_(self) -> ChatCompletionRequest:
        return ChatCompletionRequest(model="m", messages=[], tools=TOOLS)

    def test_hyv4_owns_required_and_named_parsing(self, parser):
        """The generic JSON streaming path cannot read HYV4's tokens."""
        assert parser.supports_required_and_named is False
        assert parser.structural_tag_model == "hy_v4"

    def test_malformed_output_is_returned_as_content(self, parser, request_):
        """The wrapper opts into strict validation, so a broken block must not
        surface as a half-parsed tool call."""
        output = MALFORMED["unparsed_argument_payload"]
        result = parser.extract_tool_calls(output, request_)
        assert not result.tools_called
        assert result.content == output
        assert result.tool_calls == []

    def test_non_streaming_mints_ids(self, parser, request_):
        output = "hi" + block(
            call("get_weather", arg("city", "Beijing")) + call("get_current_date")
        )
        result = parser.extract_tool_calls(output, request_)
        assert result.tools_called
        assert result.content == "hi"
        assert [tc.function.name for tc in result.tool_calls] == [
            "get_weather",
            "get_current_date",
        ]
        assert all(tc.type == "function" and tc.id for tc in result.tool_calls)
        assert len({tc.id for tc in result.tool_calls}) == 2

    def test_streaming_sets_id_and_type_once(self, parser, request_):
        """Later deltas must carry only ``index`` + ``arguments``."""
        output = block(call("get_weather", arg("city", "Beijing")))
        previous = ""
        dumps = []
        for chunk in list(output):
            current = previous + chunk
            message = parser.extract_tool_calls_streaming(
                previous, current, chunk, [], [], [], request_
            )
            if message is not None and message.tool_calls:
                dumps += [
                    tc.model_dump(exclude_unset=True) for tc in message.tool_calls
                ]
            previous = current

        with_id = [d for d in dumps if "id" in d]
        assert len(with_id) == 1
        assert with_id[0]["type"] == "function"
        assert with_id[0]["function"]["name"] == "get_weather"
        # ``arguments`` must never be serialized as null: clients concatenate it.
        assert all(
            d.get("function", {}).get("arguments") is not None
            for d in dumps
            if "arguments" in d.get("function", {})
        )
        streamed = "".join(
            d["function"]["arguments"]
            for d in dumps
            if "arguments" in d.get("function", {})
        )
        assert json.loads(streamed) == {"city": "Beijing"}

    def test_streaming_drops_empty_deltas(self, parser, request_):
        """Returning a message here would emit a ``content: null`` chunk."""
        assert (
            parser.extract_tool_calls_streaming("a", "a", "", [], [], [], request_)
            is None
        )

    def test_streaming_mirrors_serving_state(self, parser, request_):
        output = block(call("get_weather", arg("city", "A")))
        previous = ""
        for chunk in list(output):
            current = previous + chunk
            parser.extract_tool_calls_streaming(
                previous, current, chunk, [], [], [], request_
            )
            previous = current
        assert parser.current_tool_id == 0
        assert parser.prev_tool_call_arr == [
            {"name": "get_weather", "arguments": {"city": "A"}}
        ]
        assert parser.streamed_args_for_tool == ['{"city": "A"}']

    def test_request_tools_are_used_when_none_at_construction(self, request_):
        parser = HYV4ToolParser(FakeTokenizer())
        result = parser.extract_tool_calls(
            block(call("get_weather", arg("days", "3"))), request_
        )
        assert result.tool_calls[0].function.arguments == '{"days": 3}'

    def test_structural_tag_passes_the_detected_suffix(self, parser):
        request = ChatCompletionRequest(
            model="m", messages=[], tools=TOOLS, tool_choice="required"
        )
        tag = parser.get_structural_tag(request, reasoning=False)
        assert tag is not None
        grammar = Grammar.from_structural_tag(tag)
        assert _is_grammar_accept_string(
            grammar, block(call("get_weather", arg("city", "A")))
        )

    def test_structural_tag_is_none_without_tools(self, parser):
        request = ChatCompletionRequest(model="m", messages=[], tools=None)
        assert parser.get_structural_tag(request, reasoning=False) is None
