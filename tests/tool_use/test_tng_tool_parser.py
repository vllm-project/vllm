# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import AsyncIterator
from typing import Union
from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.tool_parsers.utils import run_tool_extraction
from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionToolsParam,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall, FunctionCall,
                                              FunctionDefinition)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.tool_parsers import (TngToolParser,
                                                  ToolParserManager)
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

# Use a common model that is likely to be available
MODEL = "tngtech/DeepSeek-TNG-R1T2-Chimera"


@pytest.fixture(scope="module")
def tng_tokenizer() -> AnyTokenizer:
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def tng_tool_parser(tng_tokenizer: AnyTokenizer) -> TngToolParser:
    return TngToolParser(tng_tokenizer)


SIMPLE_WEATHER_TOOL = ChatCompletionToolsParam(
    function=FunctionDefinition(name="get_current_weather",
                                parameters={
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string"
                                        }
                                    }
                                }))

NESTED_WEATHER_TOOL = ChatCompletionToolsParam(function=FunctionDefinition(
    name="get_weather_with_details",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string"
            },
            "coordinates": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude coordinate"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude coordinate"
                    },
                },
            }
        }
    }))

MODEL_OUTPUT_SINGLE_TOOL_CALL_NO_CONTENT = """<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n</tool_call>"""  # noqa: E501
MODEL_OUTPUT_SINGLE_TOOL_CALL = """<think>Okay, ...</think>\nSome prefix\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n</tool_call>\nSome suffix"""  # noqa: E501
MODEL_OUTPUT_SINGLE_TOOL_CALL_NO_THINK = """Some prefix\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n</tool_call>\nSome suffix"""  # noqa: E501
MODEL_OUTPUT_MULTIPLE_TOOL_CALLS = """<think>Okay, ...</think>\nSome prefix\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n</tool_call>\nSome middle\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Munich"}}\n</tool_call>\nSome suffix"""  # noqa: E501
MODEL_OUTPUT_SINGLE_TOOL_CALL_NESTED = """<think>Okay, ...</think>\nSome prefix\n<tool_call>\n{"name": "get_weather_with_details", "arguments": {"location": "Seattle", "coordinates": {"latitude": 47.6062, "longitude": 122.3321}}}\n</tool_call>\nSome suffix"""  # noqa: E501
MODEL_OUTPUT_SINGLE_TOOL_CALL_LIST = """<think>Okay, ...</think>\nSome prefix\n<tool_call>\n[ {"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n,\n{"name": "get_current_weather", "arguments": {"location": "Munich"}} ]\n</tool_call>\nSome suffix"""  # noqa: E501
MODEL_OUTPUT_MULTIPLE_TOOL_CALL_LISTS = """<think>Okay, ...</think>\nSome prefix\n<tool_call>\n[{"name": "get_weather_with_details", "arguments": {"location": "Seattle", "coordinates": {"latitude": 47.6062, "longitude": 122.3321}}}]\n</tool_call>\nSome middle<tool_call>[{"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n,\n{"name": "get_current_weather", "arguments": {"location": "Munich"}}]</tool_call>Some suffix"""  # noqa: E501
MODEL_OUTPUT_WITH_TOOL_CALL_IN_THINK = """<think>Okay, ...<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n</tool_call>\n</think>After think\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Munich"}}\n</tool_call>\nSome suffix"""  # noqa: E501
MODEL_OUTPUT_WITH_TOOL_CALL_WITHOUT_ARGS = """<tool_call>{"name": "refresh", "arguments": {}}</tool_call>"""  # noqa: E501
MODEL_OUTPUT_WITH_TOOL_CALL_WIKIPEDIA = """<think>\nOkay, this is a reasoning trace.\n</think>\n\n<tool_call>\n[{"name": "Wikipedia", "arguments": {"query": "Joe Biden"}}]\n</tool_call>"""  # noqa: E501

TOOL_CALL_WEATHER_SEATTLE = FunctionCall(
    name="get_current_weather",
    arguments='{"location": "Seattle"}',
)
TOOL_CALL_WEATHER_MUNICH = FunctionCall(
    name="get_current_weather",
    arguments='{"location": "Munich"}',
)
TOOL_CALL_WEATHER_DETAILS_SEATTLE = FunctionCall(
    name="get_weather_with_details",
    arguments=
    '{"location": "Seattle", "coordinates": {"latitude": 47.6062, "longitude": 122.3321}}',  # noqa: E501
)
TOOL_CALL_REFRESH = FunctionCall(
    name="refresh",
    arguments='{}',
)


def _chunked(text: str, chunk_length: int) -> list[str]:
    return [
        text[i:i + chunk_length] for i in range(0, len(text), chunk_length)
    ]


@pytest.mark.parametrize(
    ("model_output", "expected_tool_calls", "expected_content"),
    [
        pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL_NO_CONTENT,
                     [TOOL_CALL_WEATHER_SEATTLE],
                     None,
                     id="single_tool_call_no_content"),
        pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL,
                     [TOOL_CALL_WEATHER_SEATTLE],
                     "<think>Okay, ...</think>\nSome prefix\n\nSome suffix",
                     id="single_tool_call"),
        pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL_NO_THINK,
                     [TOOL_CALL_WEATHER_SEATTLE],
                     "Some prefix\n\nSome suffix",
                     id="single_tool_call_no_think"),
        pytest.param(
            MODEL_OUTPUT_MULTIPLE_TOOL_CALLS,
            [TOOL_CALL_WEATHER_SEATTLE, TOOL_CALL_WEATHER_MUNICH],
            "<think>Okay, ...</think>\nSome prefix\n\nSome middle\n\nSome suffix",  # noqa: E501
            id="multiple_tool_calls"),
        pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL_NESTED,
                     [TOOL_CALL_WEATHER_DETAILS_SEATTLE],
                     "<think>Okay, ...</think>\nSome prefix\n\nSome suffix",
                     id="single_tool_call_with_nested_arguments"),
        pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL_LIST,
                     [TOOL_CALL_WEATHER_SEATTLE, TOOL_CALL_WEATHER_MUNICH],
                     "<think>Okay, ...</think>\nSome prefix\n\nSome suffix",
                     id="single_json_list_of_tool_calls"),
        pytest.param(
            MODEL_OUTPUT_MULTIPLE_TOOL_CALL_LISTS, [
                TOOL_CALL_WEATHER_DETAILS_SEATTLE, TOOL_CALL_WEATHER_SEATTLE,
                TOOL_CALL_WEATHER_MUNICH
            ],
            "<think>Okay, ...</think>\nSome prefix\n\nSome middleSome suffix",
            id="multiple_json_lists_of_tool_calls"),
        pytest.param(
            MODEL_OUTPUT_WITH_TOOL_CALL_IN_THINK,
            [TOOL_CALL_WEATHER_MUNICH],
            """<think>Okay, ...<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Seattle"}}\n</tool_call>\n</think>After think\n\nSome suffix""",  # noqa: E501
            id="ignore_tool_call_in_think_trace"),
        pytest.param(MODEL_OUTPUT_WITH_TOOL_CALL_WITHOUT_ARGS,
                     [TOOL_CALL_REFRESH],
                     """""",
                     id="tool_call_without_args"),
        pytest.param("model output with no tool call", [],
                     "model output with no tool call",
                     id="no_tool_call"),
    ])
@pytest.mark.parametrize("chunk_length", [1, 3, 15, 0, None])
def test_streaming_and_non_streaming(model_output: str,
                                     expected_tool_calls: list[FunctionCall],
                                     expected_content: str,
                                     chunk_length: Union[int, None]):
    """Test model outputs in streaming mode with different chunk lengths
    and in non-streaming mode. Only check re-assembled tool calls,
    not their chunked representations."""
    tool_parser = TngToolParser(tokenizer=MagicMock())
    if chunk_length:
        model_output = _chunked(model_output,
                                chunk_length=chunk_length)  # type: ignore
    streaming = chunk_length is not None

    actual_content, actual_tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming)

    assert len(actual_tool_calls) == len(expected_tool_calls)
    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):
        assert actual_tool_call.function == expected_tool_call
    if expected_content:
        assert actual_content == expected_content
    else:
        assert not actual_content  # may be "" or None


@pytest.mark.parametrize(
    ("model_output", "expected_stream"), [
        pytest.param(
            [
                "<think>", "Okay", ", ", "...", "</think>", "\n", "<", "tool_",
                "call", ">", "\n", "{\"", "name", "\": ", "\"get",
                "_current_weather\"", ", \"arguments\"", ": ", "{\"", "locat",
                "ion", "\": ", "\"Seat", "tle", "\"", "}", "}", "\n", "</",
                "tool_", "call", ">"
            ],
            [
                DeltaMessage(content=chunk) for chunk in
                ["<think>", "Okay", ", ", "...", "</think>", "\n"]
            ] + [
                DeltaMessage(tool_calls=[
                    DeltaToolCall(index=0,
                                  function=DeltaFunctionCall(
                                      name="get_current_weather",
                                      arguments="{\""))
                ])
            ] + [
                DeltaMessage(tool_calls=[
                    DeltaToolCall(index=0,
                                  function=DeltaFunctionCall(arguments=chunk))
                ]) for chunk in
                ["locat", "ion", "\": ", "\"Seat", "tle", "\"", "}"]
            ],
            id="think_trace_and_single_tool_call"),
    ])
def test_streamed_tool_call_chunks(tng_tool_parser, model_output: list[str],
                                   expected_stream: list):
    """Test that model output is streamed in correct chunks"""
    mock_request = MagicMock(ChatCompletionRequest)
    mock_request.tools = [SIMPLE_WEATHER_TOOL, NESTED_WEATHER_TOOL]
    previous_text = ""
    actual_stream: list[DeltaMessage] = []
    for chunk in model_output:
        result = tng_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=previous_text + chunk,
            delta_text=chunk,
            previous_token_ids=[],  # token ids do not matter for TngToolParser
            current_token_ids=[],
            delta_token_ids=[],
            request=mock_request,
        )
        previous_text += chunk
        if result and (result.content is not None or result.reasoning_content
                       is not None or result.tool_calls != []):
            actual_stream.append(result)

    assert len(actual_stream) == len(expected_stream)
    for expected_chunk, actual_chunk in zip(expected_stream, actual_stream):
        assert actual_chunk.content == expected_chunk.content
        assert (
            actual_chunk.reasoning_content == expected_chunk.reasoning_content)
        if expected_chunk.tool_calls:
            assert (len(actual_chunk.tool_calls) == len(
                expected_chunk.tool_calls))
            for expected_call, actual_call in zip(expected_chunk.tool_calls,
                                                  actual_chunk.tool_calls):
                assert actual_call.index == expected_call.index
                assert actual_call.function == expected_call.function


@pytest.mark.parametrize(
    ("model_output", ),
    [
        pytest.param("""<tool_call>{"name": "mytool""}</tool_call>""",
                     id="tool_call_json_syntax_error"),
        pytest.param("""<tool_call>{"name": "mytool"}</tool_call>""",
                     id="tool_call_no_arguments"),
        pytest.param(
            """<tool_call>{"name": "mytool", "arguments": {"param": "test"}}""",
            id="tool_call_no_closing_tag"),
        pytest.param(
            """<tool_call>{"name": "mytool", "arguments": {}} Some additional text</tool_call>""",  # noqa: E501
            id="tool_call_text_after_json"),
        pytest.param(
            """<tool_call>prefix{"name": "mytool", "arguments": {}}</tool_call>""",  # noqa: E501
            id="tool_call_text_before_json"),
        pytest.param("""<tool_call>text-only</tool_call>""",
                     id="tool_call_no_json"),
        pytest.param("""<tool_call> </tool_call>""", id="tool_call_empty"),
        pytest.param("""<tool_call>{"name": "mytool", "arguments": "test"}""",
                     id="tool_call_text_arguments"),
    ])
@pytest.mark.parametrize("chunk_length", [1, 3, 15, 0, None])
def test_streaming_and_non_streaming_error_scenarios(
        model_output: str, chunk_length: Union[int, None]):
    tool_parser = TngToolParser(tokenizer=MagicMock())
    if chunk_length:
        model_output = _chunked(model_output,
                                chunk_length=chunk_length)  # type: ignore
    streaming = chunk_length is not None

    # just make sure no exceptions are raised
    _, _ = run_tool_extraction(tool_parser, model_output, streaming=streaming)


@pytest.mark.parametrize(("full_model_output", ), [
    pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL_NO_CONTENT,
                 id="end_with_tool_call_end_tag"),
    pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL_NO_CONTENT + "\n\n\n\n",
                 id="end_with_tool_call_end_tag_and_trailing_whitespace"),
    pytest.param(MODEL_OUTPUT_SINGLE_TOOL_CALL,
                 id="end_with_content_after_tool_call"),
    pytest.param(MODEL_OUTPUT_WITH_TOOL_CALL_WIKIPEDIA,
                 id="realistic_tool_call"),
    pytest.param((
        MODEL_OUTPUT_SINGLE_TOOL_CALL_NO_CONTENT.removesuffix("</tool_call>")),
                 id="end_with_tool_call_without_end_tag"),
])
@pytest.mark.parametrize("chunk_length", [1, 3, 15])
@pytest.mark.parametrize("trailing_empty_chunk", [False, True])
@pytest.mark.asyncio
async def test_chat_completion_serving(full_model_output: str,
                                       chunk_length: int,
                                       trailing_empty_chunk: bool):
    # Some streaming logic is hidden in entrypoints/openai/serving_chat.py,
    # here we test the full streaming behavior
    ToolParserManager.register_module("tng", module=TngToolParser)
    serving = OpenAIServingChat(engine_client=MagicMock(),
                                model_config=MagicMock(),
                                models=MagicMock(),
                                response_role="assistant",
                                request_logger=None,
                                chat_template=None,
                                chat_template_content_format="auto",
                                enable_auto_tools=True,
                                tool_parser="tng")
    request = ChatCompletionRequest(
        messages=[],  # irrelevant
        seed=0,
        tools=[SIMPLE_WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
    )
    request_id = "my-request-id"
    model_output = _chunked(full_model_output, chunk_length=chunk_length)
    if trailing_empty_chunk:
        # this simulates an empty chunk for the eot token
        model_output += [""]
    chunks = [
        RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=None,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=output_chunk,
                    token_ids=[idx],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=("length" if
                                   (idx == len(model_output) - 1) else None),
                )
            ],
            finished=(idx == len(model_output) - 1),
        ) for idx, output_chunk in enumerate(model_output)
    ]

    async def result_generator() -> AsyncIterator[RequestOutput]:
        for chunk in chunks:
            yield chunk

    async for response in serving.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id=request_id,
            model_name=MODEL,
            conversation=[],  # irrelevant
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
            enable_force_include_usage=False,
    ):
        if response.strip() == "data: [DONE]":
            continue
        response_json = json.loads(response.removeprefix("data:").strip())
        choices = response_json["choices"]
        if len(choices) and choices[0].get("finish_reason"):
            return  # success
    pytest.fail("Stream must end with finish_reason")
