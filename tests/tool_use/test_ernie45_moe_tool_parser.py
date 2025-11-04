# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
from collections.abc import Generator

import pytest

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.ernie45_tool_parser import Ernie45ToolParser
from vllm.transformers_utils.detokenizer_utils import detokenize_incrementally
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

# Use a common model that is likely to be available
MODEL = "baidu/ERNIE-4.5-21B-A3B-Thinking"


@pytest.fixture(scope="module")
def ernie45_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def ernie45_tool_parser(ernie45_tokenizer):
    return Ernie45ToolParser(ernie45_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 0

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        # Compare arguments as JSON objects to handle formatting differences
        actual_args = json.loads(actual_tool_call.function.arguments)
        expected_args = json.loads(expected_tool_call.function.arguments)
        assert actual_args == expected_args


def test_extract_tool_calls_no_tools(ernie45_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = ernie45_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_call",
        "multiple_tool_calls",
        "tool_call_with_content_before",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Beijing"}}
</tool_call>
""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_temperature",
                        arguments=json.dumps(
                            {
                                "location": "Beijing",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Beijing"}}
</tool_call>
<tool_call>
{"name": "get_temperature_unit", "arguments": {"location": "Guangzhou", "unit": "c"}}
</tool_call>
""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_temperature",
                        arguments=json.dumps(
                            {
                                "location": "Beijing",
                            }
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_temperature_unit",
                        arguments=json.dumps(
                            {
                                "location": "Guangzhou",
                                "unit": "c",
                            }
                        ),
                    )
                ),
            ],
            None,
        ),
        (
            """I need to call two tools to handle these two issues separately.
</think>

<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Beijing"}}
</tool_call>
<tool_call>
{"name": "get_temperature_unit", "arguments": {"location": "Guangzhou", "unit": "c"}}
</tool_call>
""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_temperature",
                        arguments=json.dumps(
                            {
                                "location": "Beijing",
                            }
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_temperature_unit",
                        arguments=json.dumps(
                            {
                                "location": "Guangzhou",
                                "unit": "c",
                            }
                        ),
                    )
                ),
            ],
            "I need to call two tools to handle these two issues separately.\n</think>",
        ),
    ],
)
def test_extract_tool_calls(
    ernie45_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = ernie45_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def stream_delta_message_generator(
    ernie45_tool_parser: Ernie45ToolParser,
    ernie45_tokenizer: AnyTokenizer,
    model_output: str,
    request: ChatCompletionRequest | None = None,
) -> Generator[DeltaMessage, None, None]:
    all_token_ids = ernie45_tokenizer.encode(model_output, add_special_tokens=False)

    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    for i, delta_token in enumerate(all_token_ids):
        delta_token_ids = [delta_token]
        previous_token_ids = all_token_ids[:i]
        current_token_ids = all_token_ids[: i + 1]

        (new_tokens, delta_text, new_prefix_offset, new_read_offset) = (
            detokenize_incrementally(
                tokenizer=ernie45_tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = ernie45_tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request=request,
        )
        if delta_message:
            yield delta_message

        previous_text = current_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


@pytest.mark.parametrize(
    ids=[
        "single_tool_call",
        "multiple_tool_calls",
        "tool_call_with_content_before",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Beijing"}}
</tool_call>
""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_temperature",
                        arguments=json.dumps(
                            {
                                "location": "Beijing",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Beijing"}}
</tool_call>
<tool_call>
{"name": "get_temperature_unit", "arguments": {"location": "Guangzhou", "unit": "c"}}
</tool_call>
""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_temperature",
                        arguments=json.dumps(
                            {
                                "location": "Beijing",
                            }
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_temperature_unit",
                        arguments=json.dumps(
                            {
                                "location": "Guangzhou",
                                "unit": "c",
                            }
                        ),
                    )
                ),
            ],
            None,
        ),
        (
            """I need to call two tools to handle these two issues separately.
</think>

<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Beijing"}}
</tool_call>
<tool_call>
{"name": "get_temperature_unit", "arguments": {"location": "Guangzhou", "unit": "c"}}
</tool_call>
""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_temperature",
                        arguments=json.dumps(
                            {
                                "location": "Beijing",
                            }
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_temperature_unit",
                        arguments=json.dumps(
                            {
                                "location": "Guangzhou",
                                "unit": "c",
                            }
                        ),
                    )
                ),
            ],
            "I need to call two tools to handle these two issues separately.\n</think>",
        ),
    ],
)
def test_extract_tool_calls_streaming_incremental(
    ernie45_tool_parser,
    ernie45_tokenizer,
    model_output,
    expected_tool_calls,
    expected_content,
):
    """Verify the Ernie45 Parser streaming behavior by verifying each chunk is as expected."""  # noqa: E501
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=[])

    tool_calls_dict = {}
    for delta_message in stream_delta_message_generator(
        ernie45_tool_parser, ernie45_tokenizer, model_output, request
    ):
        if (
            delta_message.role is None
            and delta_message.content is None
            and delta_message.reasoning_content is None
            and len(delta_message.tool_calls) == 0
        ):
            continue
        tool_calls = delta_message.tool_calls
        for tool_call_chunk in tool_calls:
            index = tool_call_chunk.index
            if index not in tool_calls_dict:
                if tool_call_chunk.function.arguments is None:
                    tool_call_chunk.function.arguments = ""
                tool_calls_dict[index] = tool_call_chunk
            else:
                tool_calls_dict[
                    index
                ].function.arguments += tool_call_chunk.function.arguments
    actual_tool_calls = list(tool_calls_dict.values())

    assert len(actual_tool_calls) > 0
    # check tool call format
    assert_tool_calls(actual_tool_calls, expected_tool_calls)
