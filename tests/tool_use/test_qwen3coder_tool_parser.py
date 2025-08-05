# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Generator
from typing import Optional

import pytest

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionToolsParam,
                                              DeltaMessage, FunctionCall,
                                              ToolCall)
from vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser import (
    Qwen3CoderToolParser)
from vllm.transformers_utils.detokenizer import detokenize_incrementally
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def qwen3_tool_parser(qwen3_tokenizer):
    return Qwen3CoderToolParser(qwen3_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(type="function",
                                 function={
                                     "name": "get_current_weather",
                                     "description": "Get the current weather",
                                     "parameters": {
                                         "type": "object",
                                         "properties": {
                                             "city": {
                                                 "type": "string",
                                                 "description": "The city name"
                                             },
                                             "state": {
                                                 "type": "string",
                                                 "description":
                                                 "The state code"
                                             },
                                             "unit": {
                                                 "type": "string",
                                                 "enum":
                                                 ["fahrenheit", "celsius"]
                                             }
                                         },
                                         "required": ["city", "state"]
                                     }
                                 }),
        ChatCompletionToolsParam(type="function",
                                 function={
                                     "name": "calculate_area",
                                     "description":
                                     "Calculate area of a shape",
                                     "parameters": {
                                         "type": "object",
                                         "properties": {
                                             "shape": {
                                                 "type": "string"
                                             },
                                             "dimensions": {
                                                 "type": "object"
                                             },
                                             "precision": {
                                                 "type": "integer"
                                             }
                                         }
                                     }
                                 })
    ]


def assert_tool_calls(actual_tool_calls: list[ToolCall],
                      expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):
        # Qwen3 parser doesn't generate IDs during extraction
        assert actual_tool_call.type == "function"
        assert (
            actual_tool_call.function.name == expected_tool_call.function.name)
        assert (json.loads(actual_tool_call.function.arguments) == json.loads(
            expected_tool_call.function.arguments))


def stream_delta_message_generator(
    qwen3_tool_parser: Qwen3CoderToolParser,
    qwen3_tokenizer: AnyTokenizer,
    model_output: str,
    request: Optional[ChatCompletionRequest] = None
) -> Generator[DeltaMessage, None, None]:
    all_token_ids = qwen3_tokenizer.encode(model_output,
                                           add_special_tokens=False)

    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    for i, delta_token in enumerate(all_token_ids):
        delta_token_ids = [delta_token]
        previous_token_ids = all_token_ids[:i]
        current_token_ids = all_token_ids[:i + 1]

        (new_tokens, delta_text, new_prefix_offset,
         new_read_offset) = detokenize_incrementally(
             tokenizer=qwen3_tokenizer,
             all_input_ids=current_token_ids,
             prev_tokens=previous_tokens,
             prefix_offset=prefix_offset,
             read_offset=read_offset,
             skip_special_tokens=False,
             spaces_between_special_tokens=True,
         )

        current_text = previous_text + delta_text

        delta_message = qwen3_tool_parser.extract_tool_calls_streaming(
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
        previous_tokens = (previous_tokens +
                           new_tokens if previous_tokens else new_tokens)
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


def test_extract_tool_calls_no_tools(qwen3_tool_parser):
    model_output = "This is a test response without any tool calls"
    extracted_tool_calls = qwen3_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool",
        "single_tool_with_content",
        "single_tool_multiline_param",
        "parallel_tools",
        "tool_with_typed_params",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        ('''<tool_call>
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
</tool_call>''', [
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Dallas",
                                          "state": "TX",
                                          "unit": "fahrenheit"
                                      })))
        ], None),
        ('''Sure! Let me check the weather for you.<tool_call>
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
</tool_call>''', [
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Dallas",
                                          "state": "TX",
                                          "unit": "fahrenheit"
                                      })))
        ], "Sure! Let me check the weather for you."),
        ('''<tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10, 
 "height": 20}
</parameter>
<parameter=precision>
2
</parameter>
</function>
</tool_call>''', [
            ToolCall(function=FunctionCall(name="calculate_area",
                                           arguments=json.dumps({
                                               "shape": "rectangle",
                                               "dimensions": {
                                                   "width": 10,
                                                   "height": 20
                                               },
                                               "precision": 2
                                           })))
        ], None),
        ('''<tool_call>
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
</tool_call>
<tool_call>
<function=get_current_weather>
<parameter=city>
Orlando
</parameter>
<parameter=state>
FL
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>''', [
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Dallas",
                                          "state": "TX",
                                          "unit": "fahrenheit"
                                      }))),
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Orlando",
                                          "state": "FL",
                                          "unit": "fahrenheit"
                                      })))
        ], None),
        ('''Let me calculate that area for you.<tool_call>
<function=calculate_area>
<parameter=shape>
circle
</parameter>
<parameter=dimensions>
{"radius": 15.5}
</parameter>
<parameter=precision>
3
</parameter>
</function>
</tool_call>''', [
            ToolCall(function=FunctionCall(name="calculate_area",
                                           arguments=json.dumps({
                                               "shape": "circle",
                                               "dimensions": {
                                                   "radius": 15.5
                                               },
                                               "precision": 3
                                           })))
        ], "Let me calculate that area for you."),
    ],
)
def test_extract_tool_calls(qwen3_tool_parser, sample_tools, model_output,
                            expected_tool_calls, expected_content):
    request = ChatCompletionRequest(model=MODEL,
                                    messages=[],
                                    tools=sample_tools)
    extracted_tool_calls = qwen3_tool_parser.extract_tool_calls(
        model_output, request=request)
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_fallback_no_tags(qwen3_tool_parser, sample_tools):
    """Test fallback parsing when XML tags are missing"""
    model_output = '''<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>'''

    request = ChatCompletionRequest(model=MODEL,
                                    messages=[],
                                    tools=sample_tools)
    extracted_tool_calls = qwen3_tool_parser.extract_tool_calls(
        model_output, request=request)

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert (extracted_tool_calls.tool_calls[0].function.name ==
            "get_current_weather")


def test_extract_tool_calls_type_conversion(qwen3_tool_parser):
    """Test parameter type conversion based on tool schema"""
    tools = [
        ChatCompletionToolsParam(type="function",
                                 function={
                                     "name": "test_types",
                                     "parameters": {
                                         "type": "object",
                                         "properties": {
                                             "int_param": {
                                                 "type": "integer"
                                             },
                                             "float_param": {
                                                 "type": "float"
                                             },
                                             "bool_param": {
                                                 "type": "boolean"
                                             },
                                             "str_param": {
                                                 "type": "string"
                                             },
                                             "obj_param": {
                                                 "type": "object"
                                             }
                                         }
                                     }
                                 })
    ]

    model_output = '''<tool_call>
<function=test_types>
<parameter=int_param>
42
</parameter>
<parameter=float_param>
3.14
</parameter>
<parameter=bool_param>
true
</parameter>
<parameter=str_param>
hello world
</parameter>
<parameter=obj_param>
{"key": "value"}
</parameter>
</function>
</tool_call>'''

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    extracted_tool_calls = qwen3_tool_parser.extract_tool_calls(
        model_output, request=request)

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args["int_param"] == 42
    assert args["float_param"] == 3.14
    assert args["bool_param"] is True
    assert args["str_param"] == "hello world"
    assert args["obj_param"] == {"key": "value"}


@pytest.mark.parametrize(
    ids=[
        "no_tools",
        "single_tool",
        "single_tool_with_content",
        "parallel_tools",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        ("This is a test without tools", [], "This is a test without tools"),
        ('''<tool_call>
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
</tool_call>''', [
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Dallas",
                                          "state": "TX",
                                          "unit": "fahrenheit"
                                      })))
        ], ""),
        ('''Sure! Let me check the weather for you.<tool_call>
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
</tool_call>''', [
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Dallas",
                                          "state": "TX",
                                          "unit": "fahrenheit"
                                      })))
        ], "Sure! Let me check the weather for you."),
        ('''<tool_call>
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
</tool_call>
<tool_call>
<function=get_current_weather>
<parameter=city>
Orlando
</parameter>
<parameter=state>
FL
</parameter>
<parameter=unit>
celsius
</parameter>
</function>
</tool_call>''', [
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Dallas",
                                          "state": "TX",
                                          "unit": "fahrenheit"
                                      }))),
            ToolCall(
                function=FunctionCall(name="get_current_weather",
                                      arguments=json.dumps({
                                          "city": "Orlando",
                                          "state": "FL",
                                          "unit": "celsius"
                                      })))
        ], ""),
    ],
)
def test_extract_tool_calls_streaming(qwen3_tool_parser, qwen3_tokenizer,
                                      sample_tools, model_output,
                                      expected_tool_calls, expected_content):
    """Test incremental streaming behavior"""
    request = ChatCompletionRequest(model=MODEL,
                                    messages=[],
                                    tools=sample_tools)

    other_content = ''
    tool_states = {}  # Track state per tool index

    for delta_message in stream_delta_message_generator(
            qwen3_tool_parser, qwen3_tokenizer, model_output, request):
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
                        "type": None
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
                        tool_states[idx][
                            "arguments"] += tool_call.function.arguments

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


def test_extract_tool_calls_streaming_incremental(qwen3_tool_parser,
                                                  qwen3_tokenizer,
                                                  sample_tools):
    """Test that streaming is truly incremental"""
    model_output = '''I'll check the weather.<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call>'''

    request = ChatCompletionRequest(model=MODEL,
                                    messages=[],
                                    tools=sample_tools)

    chunks = []
    for delta_message in stream_delta_message_generator(
            qwen3_tool_parser, qwen3_tokenizer, model_output, request):
        chunks.append(delta_message)

    # Should have multiple chunks
    assert len(chunks) > 3

    # First chunk(s) should be content
    assert chunks[0].content is not None
    assert chunks[0].tool_calls is None or chunks[0].tool_calls == []

    # Should have a chunk with tool header (id, name, type)
    header_found = False
    for chunk in chunks:
        if chunk.tool_calls and chunk.tool_calls[0].id:
            header_found = True
            assert (chunk.tool_calls[0].function.name == "get_current_weather")
            assert chunk.tool_calls[0].type == "function"
            # Empty initially
            assert chunk.tool_calls[0].function.arguments == ""
            break
    assert header_found

    # Should have chunks with incremental arguments
    arg_chunks = []
    for chunk in chunks:
        if chunk.tool_calls and chunk.tool_calls[0].function.arguments:
            arg_chunks.append(chunk.tool_calls[0].function.arguments)

    # Arguments should be streamed incrementally
    assert len(arg_chunks) > 1

    # Concatenated arguments should form valid JSON
    full_args = "".join(arg_chunks)
    parsed_args = json.loads(full_args)
    assert parsed_args["city"] == "Dallas"
    assert parsed_args["state"] == "TX"
