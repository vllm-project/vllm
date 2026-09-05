# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.reasoning.test_plamo3_reasoning_parser import _DummyTokenizer
from tests.tool_parsers.utils import (
    run_tool_extraction_nonstreaming,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.plamo3_tool_parser import (
    BEGIN_TOOL_ARGS_TAG,
    BEGIN_TOOL_NAME_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    BEGIN_TOOL_REQUESTS_TAG,
    END_TOOL_ARGS_TAG,
    END_TOOL_NAME_TAG,
    END_TOOL_REQUEST_TAG,
    END_TOOL_REQUESTS_TAG,
    EOT_TAG,
    Plamo3ToolParser,
)


@pytest.fixture
def parser() -> Plamo3ToolParser:
    return Plamo3ToolParser(_DummyTokenizer())


def _tool_call(name: str, arguments: str) -> str:
    return (
        f"{BEGIN_TOOL_REQUEST_TAG}"
        f"{BEGIN_TOOL_NAME_TAG}{name}{END_TOOL_NAME_TAG}"
        f"{BEGIN_TOOL_ARGS_TAG}{arguments}{END_TOOL_ARGS_TAG}"
        f"{END_TOOL_REQUEST_TAG}"
    )


def _tool_requests(*calls: str) -> str:
    return f"{BEGIN_TOOL_REQUESTS_TAG}{''.join(calls)}{END_TOOL_REQUESTS_TAG}{EOT_TAG}"


@pytest.mark.parametrize(
    ("model_output", "expected_content", "expected_calls"),
    [
        ("Regular response", "Regular response", []),
        (f"Regular response{EOT_TAG}ignored", "Regular response", []),
        ("Answer<|plamo:begin_", "Answer", []),
        (
            "Checking the weather."
            + _tool_requests(_tool_call("get_weather", '{"city": "Tokyo"}')),
            "Checking the weather.",
            [("get_weather", '{"city": "Tokyo"}')],
        ),
        (
            _tool_requests(
                _tool_call("echo", '{"text": "hi"}'),
                _tool_call("sum", '{"a": 1, "b": 2}'),
            ),
            "",
            [
                ("echo", '{"text": "hi"}'),
                ("sum", '{"a": 1, "b": 2}'),
            ],
        ),
    ],
)
def test_extract_tool_calls(parser, model_output, expected_content, expected_calls):
    result = run_tool_extraction_nonstreaming(parser, model_output)

    assert result.content == expected_content
    assert result.tools_called is bool(expected_calls)
    actual_calls = [
        (call.function.name, call.function.arguments) for call in result.tool_calls
    ]
    assert actual_calls == expected_calls


def test_extract_tool_calls_rejects_incomplete_request(parser):
    name_prefix = BEGIN_TOOL_REQUEST_TAG + BEGIN_TOOL_NAME_TAG + "broken"
    named_request = name_prefix + END_TOOL_NAME_TAG
    arguments_prefix = named_request + BEGIN_TOOL_ARGS_TAG + '{"a": 1}'
    for incomplete_request in [
        "",
        BEGIN_TOOL_REQUEST_TAG,
        name_prefix,
        named_request,
        arguments_prefix,
        arguments_prefix + END_TOOL_ARGS_TAG,
        _tool_call("broken", '{"a": 1}'),
    ]:
        model_output = "Before tools." + BEGIN_TOOL_REQUESTS_TAG + incomplete_request

        result = run_tool_extraction_nonstreaming(parser, model_output)

        assert result.content == "Before tools."
        assert result.tools_called is False
        assert result.tool_calls == []


@pytest.mark.parametrize(
    ("deltas", "expected_content"),
    [
        (["Regular response"], "Regular response"),
        ([f"Regular response{EOT_TAG}ignored", "later"], "Regular response"),
    ],
)
def test_streaming_plain_content(parser, deltas, expected_content):
    reconstructor = run_tool_extraction_streaming(parser, deltas)

    assert reconstructor.other_content == expected_content
    assert reconstructor.tool_calls == []


@pytest.mark.parametrize(
    "steps",
    [
        [("first ", "first ", ()), ("second", "second", ())],
        [
            (
                "Checking the weather." + BEGIN_TOOL_REQUESTS_TAG,
                "Checking the weather.",
                (),
            ),
        ],
        [(EOT_TAG, None, None), ("ignored", None, None)],
        [
            (
                "Before tools."
                + _tool_requests(
                    _tool_call("echo", '{"text": "hi"}'),
                    _tool_call("sum", '{"a": 1, "b": 2}'),
                ),
                "Before tools.",
                (
                    ("echo", '{"text": "hi"}'),
                    ("sum", '{"a": 1, "b": 2}'),
                ),
            ),
        ],
        [
            (
                BEGIN_TOOL_REQUESTS_TAG + _tool_call("echo", '{"text": "hi"}'),
                None,
                (("echo", '{"text": "hi"}'),),
            ),
            (
                _tool_call("sum", '{"a": 1}'),
                None,
                (("sum", '{"a": 1}'),),
            ),
        ],
    ],
)
def test_streaming_delta_messages(parser, steps):
    request = ChatCompletionRequest(messages=[], model="test-model")
    current_text = ""
    tool_indices: list[int] = []
    tool_ids: list[str | None] = []
    for delta_text, expected_content, expected_tool_calls in steps:
        previous_text = current_text
        current_text += delta_text
        message = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        if expected_content is None and expected_tool_calls is None:
            assert message is None
        else:
            assert message is not None
            assert message.content == expected_content
            actual_tool_calls = message.tool_calls or []
            assert [
                (call.function.name, call.function.arguments)
                for call in actual_tool_calls
            ] == list(expected_tool_calls)
            tool_indices.extend(call.index for call in actual_tool_calls)
            tool_ids.extend(call.id for call in actual_tool_calls)

    assert tool_indices == list(range(len(tool_indices)))
    assert all(tool_ids)
    assert len(set(tool_ids)) == len(tool_ids)


def test_streaming_content_then_tool_call(parser):
    reconstructor = run_tool_extraction_streaming(
        parser,
        [
            "Checking the weather." + BEGIN_TOOL_REQUESTS_TAG,
            (
                BEGIN_TOOL_REQUEST_TAG
                + BEGIN_TOOL_NAME_TAG
                + "get_weather"
                + END_TOOL_NAME_TAG
                + BEGIN_TOOL_ARGS_TAG
                + '{"city": "Tokyo"'
            ),
            "}"
            + END_TOOL_ARGS_TAG
            + END_TOOL_REQUEST_TAG
            + END_TOOL_REQUESTS_TAG
            + EOT_TAG,
        ],
    )

    assert reconstructor.other_content == "Checking the weather."
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert reconstructor.tool_calls[0].function.arguments == '{"city": "Tokyo"}'


def test_streaming_holds_fragmented_delimiters(parser):
    opening = BEGIN_TOOL_REQUESTS_TAG
    closing = END_TOOL_ARGS_TAG
    partial_opening = opening[: len(opening) - 1]
    partial_closing = closing[: len(closing) - 1]

    reconstructor = run_tool_extraction_streaming(
        parser,
        [
            "Before tools." + partial_opening,
            opening[len(partial_opening) :]
            + BEGIN_TOOL_REQUEST_TAG
            + BEGIN_TOOL_NAME_TAG
            + "sum"
            + END_TOOL_NAME_TAG
            + BEGIN_TOOL_ARGS_TAG
            + '{"a": 1}'
            + partial_closing,
            closing[len(partial_closing) :]
            + END_TOOL_REQUEST_TAG
            + END_TOOL_REQUESTS_TAG
            + EOT_TAG,
        ],
    )

    assert reconstructor.other_content == "Before tools."
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "sum"
    assert reconstructor.tool_calls[0].function.arguments == '{"a": 1}'


def test_streaming_handles_incomplete_request_boundaries():
    name_prefix = BEGIN_TOOL_REQUEST_TAG + BEGIN_TOOL_NAME_TAG + "sum"
    named_request = name_prefix + END_TOOL_NAME_TAG
    arguments_prefix = named_request + BEGIN_TOOL_ARGS_TAG + '{"a": 1}'
    for incomplete_request, expected_calls in [
        ("", []),
        (BEGIN_TOOL_REQUEST_TAG, []),
        (name_prefix, []),
        (named_request, [("sum", "")]),
        (arguments_prefix, [("sum", '{"a": 1}')]),
        (arguments_prefix + END_TOOL_ARGS_TAG, [("sum", '{"a": 1}')]),
        (_tool_call("sum", '{"a": 1}'), [("sum", '{"a": 1}')]),
        (
            _tool_call("sum", '{"a": 1}') + END_TOOL_REQUESTS_TAG,
            [("sum", '{"a": 1}')],
        ),
    ]:
        reconstructor = run_tool_extraction_streaming(
            Plamo3ToolParser(_DummyTokenizer()),
            ["Before tools." + BEGIN_TOOL_REQUESTS_TAG + incomplete_request],
        )

        assert reconstructor.other_content == "Before tools."
        assert [
            (call.function.name, call.function.arguments)
            for call in reconstructor.tool_calls
        ] == expected_calls
