# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
from collections.abc import Generator

import pytest

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaMessage,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.seed_oss_tool_parser import SeedOssToolParser
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

pytestmark = pytest.mark.cpu_test

# Use a common model that is likely to be available
MODEL = "ByteDance-Seed/Seed-OSS-36B-Instruct"


@pytest.fixture(scope="module")
def seed_oss_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def seed_oss_tool_parser(seed_oss_tokenizer):
    return SeedOssToolParser(seed_oss_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia",
                        },
                        "unit": {
                            "type": "string",
                            "description": "this is the unit of temperature",
                        },
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
                "returns": {
                    "type": "object",
                    "properties": {
                        "temperature": {
                            "type": "number",
                            "description": "temperature in celsius",
                        }
                    },
                    "required": ["temperature"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ),
    ]


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        # Seed-OSS tool call will not generate id
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function

        assert actual_tool_call.function.name == expected_tool_call.function.name
        assert (
            actual_tool_call.function.arguments == expected_tool_call.function.arguments
        )


def test_extract_tool_calls_no_tools(seed_oss_tool_parser):
    model_output = "This is a test response without any tool calls"
    extracted_tool_calls = seed_oss_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "tool_call_0_thinking_budget",
        "tool_call_512_thinkg_budget",
        "tool_call_unlimited_thinking_budget",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<seed:tool_call>\n<function=get_weather>\n"""
            """<parameter=location>Barcelona, Spain</parameter>\n</function>\n</seed:tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "location": "Barcelona, Spain",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            None,
        ),
        (
            """<seed:think>The user\'s current thinking budget is 512.</seed:cot_budget_reflect>\nLet me analyze the """
            """question. The user wants to know the weather in Barcelona, Spain. Looking at the functions available, """
            """there\'s a get_weather function that can retrieve the current temperature for a given location. \n\nFirst, """
            """check the parameters required by get_weather: location is mandatory (needs city and country), and unit is """
            """optional. The user provided "Barcelona Spain" as the location, which fits the required format (city, """
            """country). \n<seed:cot_budget_reflect>I have used 131 tokens, and there are 381 tokens remaining for use."""
            """</seed:cot_budget_reflect>\n Since the unit isn\'t specified, the function will default to Celsius, which """
            """is fine. \n\nThere\'s no need to ask for more information because the location is clear. So I should call """
            """the get_weather function with location set to "Barcelona, Spain" (adding a comma for clarity, though the """
            """user\'s input has a space, but the function might accept either; to be safe, using the standard format """
            """with a comma).\n<seed:cot_budget_reflect>I have used 257 tokens, and there are 255 tokens remaining for """
            """use.</seed:cot_budget_reflect>\n The unit parameter can be omitted since it\'s optional.</seed:think>\n"""
            """<seed:tool_call>\n<function=get_weather>\n<parameter=location>Barcelona, Spain</parameter>\n</function>"""
            """\n</seed:tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "location": "Barcelona, Spain",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            """<seed:think>The user\'s current thinking budget is 512.</seed:cot_budget_reflect>\nLet me analyze the """
            """question. The user wants to know the weather in Barcelona, Spain. Looking at the functions available, """
            """there\'s a get_weather function that can retrieve the current temperature for a given location. \n\nFirst, """
            """check the parameters required by get_weather: location is mandatory (needs city and country), and unit is """
            """optional. The user provided "Barcelona Spain" as the location, which fits the required format (city, """
            """country). \n<seed:cot_budget_reflect>I have used 131 tokens, and there are 381 tokens remaining for use."""
            """</seed:cot_budget_reflect>\n Since the unit isn\'t specified, the function will default to Celsius, which """
            """is fine. \n\nThere\'s no need to ask for more information because the location is clear. So I should call """
            """the get_weather function with location set to "Barcelona, Spain" (adding a comma for clarity, though the """
            """user\'s input has a space, but the function might accept either; to be safe, using the standard format """
            """with a comma).\n<seed:cot_budget_reflect>I have used 257 tokens, and there are 255 tokens remaining for """
            """use.</seed:cot_budget_reflect>\n The unit parameter can be omitted since it\'s optional.</seed:think>\n""",
        ),
        (
            """<seed:think>\nGot it, let\'s see. The user asked for the weather in Barcelona, Spain. """
            """First, I need to remember the function I can use: get_weather. The function requires a """
            """location (city and country) which is "Barcelona, Spain" here, and unit is optional. Since """
            """the user didn\'t specify the unit, the default in the function is Celsius, right? Wait, """
            """let me check the function docstring again. Oh, the function says unit is optional, and """
            """returns temperature in Celsius. So I should call get_weather with location "Barcelona, """
            """Spain" and maybe omit unit or set to Celsius. Let me format the function call correctly. """
            """The format is <seed:tool_call>\n<function=get_weather>\n<parameter=location>Barcelona, """
            """Spain</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</seed:tool_call>. """
            """Wait, but does the unit parameter accept "celsius"? The docstring says unit is the unit """
            """of temperature, but the return is in Celsius anyway. Maybe even if I don\'t pass unit, """
            """it\'s okay, but to be explicit, maybe pass "celsius". Let me go with that. So the function """
            """call should be as above. Then wait for the result to come back and tell the user the """
            """temperature in Celsius.</seed:think><seed:tool_call>\n<function=get_weather>\n<parameter=location>"""
            """Barcelona, Spain</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</seed:tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "location": "Barcelona, Spain",
                                "unit": "celsius",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            """<seed:think>\nGot it, let\'s see. The user asked for the weather in Barcelona, Spain. """
            """First, I need to remember the function I can use: get_weather. The function requires a """
            """location (city and country) which is "Barcelona, Spain" here, and unit is optional. Since """
            """the user didn\'t specify the unit, the default in the function is Celsius, right? Wait, """
            """let me check the function docstring again. Oh, the function says unit is optional, and """
            """returns temperature in Celsius. So I should call get_weather with location "Barcelona, """
            """Spain" and maybe omit unit or set to Celsius. Let me format the function call correctly. """
            """The format is <seed:tool_call>\n<function=get_weather>\n<parameter=location>Barcelona, """
            """Spain</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</seed:tool_call>. """
            """Wait, but does the unit parameter accept "celsius"? The docstring says unit is the unit """
            """of temperature, but the return is in Celsius anyway. Maybe even if I don\'t pass unit, """
            """it\'s okay, but to be explicit, maybe pass "celsius". Let me go with that. So the function """
            """call should be as above. Then wait for the result to come back and tell the user the """
            """temperature in Celsius.</seed:think>""",
        ),
    ],
)
def test_extract_tool_calls(
    seed_oss_tool_parser,
    sample_tools,
    model_output,
    expected_tool_calls,
    expected_content,
):
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    extracted_tool_calls = seed_oss_tool_parser.extract_tool_calls(
        model_output, request=request
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_streaming_tool_calls_no_tools(seed_oss_tool_parser):
    model_output = "This is a test response without any tool calls"

    result = seed_oss_tool_parser.extract_tool_calls_streaming(
        previous_text="his is a test response",
        current_text=model_output,
        delta_text=" without any tool calls.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return the delta text as content
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == " without any tool calls."


def stream_delta_message_generator(
    seed_oss_tool_parser: SeedOssToolParser,
    seed_oss_tokenizer: TokenizerLike,
    model_output: str,
    request: ChatCompletionRequest | None = None,
) -> Generator[DeltaMessage, None, None]:
    all_token_ids = seed_oss_tokenizer.encode(model_output, add_special_tokens=False)

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
                tokenizer=seed_oss_tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = seed_oss_tool_parser.extract_tool_calls_streaming(
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
        "tool_call_0_thinking_budget",
        "tool_call_512_thinkg_budget",
        "tool_call_unlimited_thinking_budget",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """<seed:think>\n</seed:cot_budget_reflect>\n</seed:cot_budget_reflect>\n"""
            """The current thinking budget is 0, so I will directly start answering the question.\n</seed:think>\n"""
            """<seed:tool_call>\n<function=get_weather>\n"""
            """<parameter=location>Barcelona, Spain</parameter>\n</function>\n</seed:tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "location": "Barcelona, Spain",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            """<seed:think>\n</seed:cot_budget_reflect>\n</seed:cot_budget_reflect>\n"""
            """The current thinking budget is 0, so I will directly start answering the question.\n</seed:think>\n""",
        ),
        (
            """<seed:think>The user\'s current thinking budget is 512.</seed:cot_budget_reflect>\nLet me analyze the """
            """question. The user wants to know the weather in Barcelona, Spain. Looking at the functions available, """
            """there\'s a get_weather function that can retrieve the current temperature for a given location. \n\nFirst, """
            """check the parameters required by get_weather: location is mandatory (needs city and country), and unit is """
            """optional. The user provided "Barcelona Spain" as the location, which fits the required format (city, """
            """country). \n<seed:cot_budget_reflect>I have used 131 tokens, and there are 381 tokens remaining for use."""
            """</seed:cot_budget_reflect>\n Since the unit isn\'t specified, the function will default to Celsius, which """
            """is fine. \n\nThere\'s no need to ask for more information because the location is clear. So I should call """
            """the get_weather function with location set to "Barcelona, Spain" (adding a comma for clarity, though the """
            """user\'s input has a space, but the function might accept either; to be safe, using the standard format """
            """with a comma).\n<seed:cot_budget_reflect>I have used 257 tokens, and there are 255 tokens remaining for """
            """use.</seed:cot_budget_reflect>\n The unit parameter can be omitted since it\'s optional.</seed:think>\n"""
            """<seed:tool_call>\n<function=get_weather>\n<parameter=location>Barcelona, Spain</parameter>\n</function>"""
            """\n</seed:tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "location": "Barcelona, Spain",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            """<seed:think>The user\'s current thinking budget is 512.</seed:cot_budget_reflect>\nLet me analyze the """
            """question. The user wants to know the weather in Barcelona, Spain. Looking at the functions available, """
            """there\'s a get_weather function that can retrieve the current temperature for a given location. \n\nFirst, """
            """check the parameters required by get_weather: location is mandatory (needs city and country), and unit is """
            """optional. The user provided "Barcelona Spain" as the location, which fits the required format (city, """
            """country). \n<seed:cot_budget_reflect>I have used 131 tokens, and there are 381 tokens remaining for use."""
            """</seed:cot_budget_reflect>\n Since the unit isn\'t specified, the function will default to Celsius, which """
            """is fine. \n\nThere\'s no need to ask for more information because the location is clear. So I should call """
            """the get_weather function with location set to "Barcelona, Spain" (adding a comma for clarity, though the """
            """user\'s input has a space, but the function might accept either; to be safe, using the standard format """
            """with a comma).\n<seed:cot_budget_reflect>I have used 257 tokens, and there are 255 tokens remaining for """
            """use.</seed:cot_budget_reflect>\n The unit parameter can be omitted since it\'s optional.</seed:think>\n""",
        ),
        (
            """<seed:think>\nGot it, let\'s see. The user asked for the weather in Barcelona, Spain. """
            """First, I need to remember the function I can use: get_weather. The function requires a """
            """location (city and country) which is "Barcelona, Spain" here, and unit is optional. Since """
            """the user didn\'t specify the unit, the default in the function is Celsius, right? Wait, """
            """let me check the function docstring again. Oh, the function says unit is optional, and """
            """returns temperature in Celsius. So I should call get_weather with location "Barcelona, """
            """Spain" and maybe omit unit or set to Celsius. Let me format the function call correctly. """
            """The format is <seed:tool_call>\n<function=get_weather>\n<parameter=location>Barcelona, """
            """Spain</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</seed:tool_call>. """
            """Wait, but does the unit parameter accept "celsius"? The docstring says unit is the unit """
            """of temperature, but the return is in Celsius anyway. Maybe even if I don\'t pass unit, """
            """it\'s okay, but to be explicit, maybe pass "celsius". Let me go with that. So the function """
            """call should be as above. Then wait for the result to come back and tell the user the """
            """temperature in Celsius.</seed:think><seed:tool_call>\n<function=get_weather>\n<parameter=location>"""
            """Barcelona, Spain</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</seed:tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "location": "Barcelona, Spain",
                                "unit": "celsius",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            """<seed:think>\nGot it, let\'s see. The user asked for the weather in Barcelona, Spain. """
            """First, I need to remember the function I can use: get_weather. The function requires a """
            """location (city and country) which is "Barcelona, Spain" here, and unit is optional. Since """
            """the user didn\'t specify the unit, the default in the function is Celsius, right? Wait, """
            """let me check the function docstring again. Oh, the function says unit is optional, and """
            """returns temperature in Celsius. So I should call get_weather with location "Barcelona, """
            """Spain" and maybe omit unit or set to Celsius. Let me format the function call correctly. """
            """The format is <seed:tool_call>\n<function=get_weather>\n<parameter=location>Barcelona, """
            """Spain</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</seed:tool_call>. """
            """Wait, but does the unit parameter accept "celsius"? The docstring says unit is the unit """
            """of temperature, but the return is in Celsius anyway. Maybe even if I don\'t pass unit, """
            """it\'s okay, but to be explicit, maybe pass "celsius". Let me go with that. So the function """
            """call should be as above. Then wait for the result to come back and tell the user the """
            """temperature in Celsius.</seed:think>""",
        ),
    ],
)
def test_streaming_tool_calls(
    seed_oss_tool_parser,
    seed_oss_tokenizer,
    sample_tools,
    model_output,
    expected_tool_calls,
    expected_content,
):
    """Test incremental streaming behavior"""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    other_content = ""
    tool_states = {}  # Track state per tool index

    for delta_message in stream_delta_message_generator(
        seed_oss_tool_parser, seed_oss_tokenizer, model_output, request
    ):
        # role should never be streamed from tool parser
        assert not delta_message.role

        if delta_message.content:
            other_content += delta_message.content

        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index

                # Initialize state for new tool
                if idx not in tool_states:
                    tool_states[idx] = {
                        "id": None,
                        "name": None,
                        "arguments": "",
                        "type": None,
                    }

                # First chunk should have id, name, and type
                if tool_call.id:
                    tool_states[idx]["id"] = tool_call.id

                if tool_call.type:
                    assert tool_call.type == "function"
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        # Should only be set once
                        assert tool_states[idx]["name"] is None
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        # Accumulate arguments incrementally
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    # Verify final content
    assert other_content == expected_content

    # Verify we got all expected tool calls
    assert len(tool_states) == len(expected_tool_calls)

    # Verify each tool call
    for idx, expected_tool in enumerate(expected_tool_calls):
        state = tool_states[idx]
        assert state["id"] is not None
        assert state["type"] == "function"
        assert state["name"] == expected_tool.function.name

        # Parse accumulated arguments
        arguments_str = state["arguments"]
        assert arguments_str is not None
        actual_args = json.loads(arguments_str)
        expected_args = json.loads(expected_tool.function.arguments)
        assert actual_args == expected_args
