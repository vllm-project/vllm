# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.minimax_m3 import (
    INVOKE_END,
    MAX_ELEMENT_DEPTH,
    NAMESPACE,
    TOOL_CALL_END,
    TOOL_CALL_START,
    MinimaxM3Parser,
)

NS = NAMESPACE

TOOLS = [
    ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="create_order",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "note": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "shipping": {
                        "type": "object",
                        "properties": {"zip": {"type": "integer"}},
                    },
                    "either": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ]
                    },
                    "free": {"type": "object"},
                },
            },
        ),
    )
]


@pytest.fixture
def parser():
    return MinimaxM3Parser(make_mock_tokenizer({}), tools=TOOLS)


def element(name: str, body: str) -> str:
    return f"{NS}<{name}>{body}{NS}</{name}>"


def tool_block(*invokes: tuple[str, str]) -> str:
    body = "\n".join(
        f'{NS}<invoke name="{name}">{params}{INVOKE_END}' for name, params in invokes
    )
    return f"{TOOL_CALL_START}\n{body}\n{TOOL_CALL_END}"


def parse_arguments(parser, request, params: str, name: str = "create_order"):
    result = parser.extract_tool_calls(tool_block((name, params)), request)
    assert [tc.function.name for tc in result.tool_calls] == [name]
    return json.loads(result.tool_calls[0].function.arguments)


class TestArguments:
    def test_nested_elements_follow_schema(self, parser, mock_request):
        params = (
            element("tags", element("item", "a") + element("item", "b"))
            + element("shipping", element("zip", "018956"))
            + element("either", element("item", "1") + element("item", "2"))
            + element("free", element("t", "x") + element("t", "y"))
        )
        assert parse_arguments(parser, mock_request, params) == {
            "tags": ["a", "b"],
            "shipping": {"zip": 18956},
            "either": [1, 2],
            "free": {"t": ["x", "y"]},
        }

    def test_empty_elements_follow_schema(self, parser, mock_request):
        # The chat template renders [] and {} as empty elements.
        params = (
            element("tags", "")
            + element("shipping", "")
            + element("note", "")
            + element("either", "")
        )
        assert parse_arguments(parser, mock_request, params) == {
            "tags": [],
            "shipping": {},
            "note": "",
            "either": "",
        }

    def test_unknown_tool_keeps_text_and_nests_objects(self, parser, mock_request):
        params = element("n", "1") + element(
            "list", element("item", "a") + element("item", "b")
        )
        assert parse_arguments(parser, mock_request, params, name="other") == {
            "n": "1",
            "list": {"item": ["a", "b"]},
        }

    def test_mixed_text_uses_reserved_field(self, parser, mock_request):
        params = element("free", "hi " + element("k", "v") + " there") + element(
            "shipping", "x" + element("$text", "child")
        )
        assert parse_arguments(parser, mock_request, params) == {
            "free": {"k": "v", "$text": "hi  there"},
            "shipping": {"$text": "child", "$$text": "x"},
        }

    def test_leaf_text_is_verbatim_for_strings(self, parser, mock_request):
        params = element("note", "\n  a <b> & c\n") + element("user_id", " 7 ")
        assert parse_arguments(parser, mock_request, params) == {
            "note": "\n  a <b> & c\n",
            "user_id": 7,
        }

    def test_block_marker_named_element_is_argument(self, parser, mock_request):
        params = element("free", element("tool_call", "v"))
        assert parse_arguments(parser, mock_request, params) == {
            "free": {"tool_call": "v"},
        }

    @pytest.mark.parametrize(
        "tail",
        [
            "junk" + element("note", "dropped"),
            f"{NS}<note>a{NS}</other>",
            f"{NS}<note>unterminated",
        ],
    )
    def test_malformed_element_keeps_preceding_params(self, parser, mock_request, tail):
        params = element("user_id", "1") + tail
        assert parse_arguments(parser, mock_request, params) == {"user_id": 1}

    def test_depth_limit(self, parser, mock_request):
        def nested(depth: int) -> str:
            body = "v"
            for _ in range(depth - 1):
                body = element("n", body)
            return element("free", body)

        accepted = parse_arguments(parser, mock_request, nested(MAX_ELEMENT_DEPTH))
        assert "free" in accepted
        rejected = parse_arguments(parser, mock_request, nested(MAX_ELEMENT_DEPTH + 1))
        assert rejected == {}


class TestToolBlock:
    def test_multiple_invokes_and_prefix_content(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "Checking.\n"
            + tool_block(
                ("create_order", element("user_id", "1")),
                ("create_order", element("user_id", "2")),
            ),
            mock_request,
        )
        assert result.content == "Checking.\n"
        assert [json.loads(tc.function.arguments) for tc in result.tool_calls] == [
            {"user_id": 1},
            {"user_id": 2},
        ]

    def test_text_after_tool_block_is_dropped(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "pre" + tool_block(("create_order", "")) + "post", mock_request
        )
        assert result.content == "pre"
        assert len(result.tool_calls) == 1

    def test_invoke_outside_tool_block_is_content(self, parser, mock_request):
        text = f'{NS}<invoke name="create_order">{INVOKE_END}'
        result = parser.extract_tool_calls(text, mock_request)
        assert not result.tools_called
        assert result.content == text


class TestStreaming:
    def test_arguments_arrive_with_invoke_end(self, parser, mock_request):
        block = tool_block(
            ("create_order", element("user_id", "42") + element("tags", ""))
        )
        invoke_end = block.index(INVOKE_END)
        results = simulate_tool_streaming(
            parser,
            mock_request,
            ["Sure. ", block[:invoke_end], block[invoke_end:]],
        )

        before_end, at_end = results[1][0], results[2][0]
        assert before_end is not None and before_end.tool_calls
        function = before_end.tool_calls[0].function
        assert function is not None
        assert function.name == "create_order"
        assert not function.arguments
        assert collect_content(results) == "Sure. "
        assert json.loads(collect_tool_arguments([(at_end, "")])) == {
            "user_id": 42,
            "tags": [],
        }

    def test_char_chunks_match_complete_parse(self, parser, mock_request):
        text = "a" + tool_block(
            ("create_order", element("shipping", element("zip", "1"))),
            ("create_order", element("note", "x")),
        )
        results = simulate_tool_streaming(parser, mock_request, list(text))

        assert collect_content(results) == "a"
        assert collect_function_name(results) == "create_order"
        streamed = [
            tc.function.arguments
            for delta, _ in results
            if delta and delta.tool_calls
            for tc in delta.tool_calls
            if tc.function and tc.function.arguments
        ]
        assert [json.loads(args) for args in streamed] == [
            {"shipping": {"zip": 1}},
            {"note": "x"},
        ]

    def test_truncated_invoke_keeps_completed_params(self, parser, mock_request):
        block = tool_block(
            ("create_order", element("user_id", "1") + element("note", "x"))
        )
        truncated = block[: block.index(f"{NS}</note>")]
        results = simulate_tool_streaming(parser, mock_request, [truncated])
        results.append((parser.finish_streaming(), truncated))

        assert collect_function_name(results) == "create_order"
        assert json.loads(collect_tool_arguments(results)) == {"user_id": 1}


def test_adjust_request_keeps_special_tokens_skipped(parser, mock_request):
    mock_request.skip_special_tokens = True
    assert parser.adjust_request(mock_request).skip_special_tokens is True
