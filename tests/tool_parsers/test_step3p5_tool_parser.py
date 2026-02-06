# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Generator

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tool_parsers.step3p5_tool_parser import Step3p5ToolParser

MODEL = "stepfun-ai/Step-3.5-Flash"


@pytest.fixture(scope="module")
def step3p5_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def step3p5_tool_parser(step3p5_tokenizer):
    return Step3p5ToolParser(step3p5_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "calculate_area",
                "description": "Calculate area of a shape",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string"},
                        "dimensions": {"type": "object"},
                        "precision": {"type": "integer"},
                    },
                },
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
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        assert json.loads(actual_tool_call.function.arguments) == json.loads(
            expected_tool_call.function.arguments
        )


def stream_delta_message_generator(
    step3p5_tool_parser,
    step3p5_tokenizer: TokenizerLike,
    model_output: str,
    request: ChatCompletionRequest | None = None,
) -> Generator[DeltaMessage, None, None]:
    all_token_ids = step3p5_tokenizer.encode(model_output, add_special_tokens=False)

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
                tokenizer=step3p5_tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = step3p5_tool_parser.extract_tool_calls_streaming(
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


def stream_delta_message_generator_from_chunks(
    step3p5_tool_parser,
    step3p5_tokenizer: TokenizerLike,
    delta_text_chunks: list[str],
    request: ChatCompletionRequest | None = None,
) -> Generator[DeltaMessage, None, None]:
    previous_text = ""
    previous_token_ids: list[int] = []

    for delta_text in delta_text_chunks:
        delta_token_ids = step3p5_tokenizer.encode(delta_text, add_special_tokens=False)
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids

        delta_message = step3p5_tool_parser.extract_tool_calls_streaming(
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
        previous_token_ids = current_token_ids


def test_extract_tool_calls_no_tools(step3p5_tool_parser):
    model_output = "This is a test response without any tool calls"
    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
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
        (
            """<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """Sure! Let me check the weather for you.<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            "Sure! Let me check the weather for you.",
        ),
        (
            """<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="calculate_area",
                        arguments=json.dumps(
                            {
                                "shape": "rectangle",
                                "dimensions": {"width": 10, "height": 20},
                                "precision": 2,
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}
                        ),
                    )
                ),
            ],
            None,
        ),
        (
            """Let me calculate that area for you.<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="calculate_area",
                        arguments=json.dumps(
                            {
                                "shape": "circle",
                                "dimensions": {"radius": 15.5},
                                "precision": 3,
                            }
                        ),
                    )
                )
            ],
            "Let me calculate that area for you.",
        ),
    ],
)
def test_extract_tool_calls(
    step3p5_tool_parser,
    sample_tools,
    model_output,
    expected_tool_calls,
    expected_content,
):
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=request
    )
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_fallback_no_tags(step3p5_tool_parser, sample_tools):
    """Test fallback parsing when XML tags are missing"""
    model_output = """<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=request
    )

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_weather"


def test_extract_tool_calls_type_conversion(step3p5_tool_parser):
    """Test parameter type conversion based on tool schema"""
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "test_types",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "int_param": {"type": "integer"},
                        "float_param": {"type": "float"},
                        "bool_param": {"type": "boolean"},
                        "str_param": {"type": "string"},
                        "obj_param": {"type": "object"},
                    },
                },
            },
        )
    ]

    model_output = """<tool_call>
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
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=request
    )

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
        "single_tool_multiline_param",
        "parallel_tools",
        "tool_with_typed_params",  # Added this test case
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        ("This is a test without tools", [], "This is a test without tools"),
        (
            """<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """Sure! Let me check the weather for you.<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            "Sure! Let me check the weather for you.",
        ),
        (
            """<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="calculate_area",
                        arguments=json.dumps(
                            {
                                "shape": "rectangle",
                                "dimensions": {"width": 10, "height": 20},
                                "precision": 2,
                            }
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Orlando", "state": "FL", "unit": "celsius"}
                        ),
                    )
                ),
            ],
            None,
        ),
        # Added tool_with_typed_params test case
        (
            """Let me calculate that area for you.<tool_call>
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
</tool_call>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="calculate_area",
                        arguments=json.dumps(
                            {
                                "shape": "circle",
                                "dimensions": {"radius": 15.5},
                                "precision": 3,
                            }
                        ),
                    )
                )
            ],
            "Let me calculate that area for you.",
        ),
    ],
)
def test_extract_tool_calls_streaming(
    step3p5_tool_parser,
    step3p5_tokenizer,
    sample_tools,
    model_output,
    expected_tool_calls,
    expected_content,
):
    """Test incremental streaming behavior including typed parameters"""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    other_content = ""
    tool_states = {}  # Track state per tool index

    for delta_message in stream_delta_message_generator(
        step3p5_tool_parser, step3p5_tokenizer, model_output, request
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
    assert other_content == (expected_content or "")  # Handle None case

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


def test_extract_tool_calls_missing_closing_parameter_tag(
    step3p5_tool_parser, sample_tools
):
    """Test handling of missing closing </parameter> tag"""
    # Using get_current_weather from sample_tools but with malformed XML
    model_output = """Let me check the weather for you:
<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=request
    )

    # The parser should handle the malformed XML gracefully
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1

    # Verify the function name is correct
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_weather"

    # Verify the arguments are parsed despite the missing closing tag
    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert "city" in args
    assert args["city"] == "Dallas"
    assert args["state"] == "TX"
    assert args["unit"] == "fahrenheit"

    # Check that content before the tool call is preserved
    assert "Let me check the weather for you:" in extracted_tool_calls.content


def test_extract_tool_calls_streaming_missing_closing_tag(
    step3p5_tool_parser, step3p5_tokenizer, sample_tools
):
    """Test streaming with missing closing </parameter> tag"""
    # Using get_current_weather from sample_tools but with malformed XML
    model_output = """Let me check the weather for you:
<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    other_content = ""
    tool_states = {}

    for delta_message in stream_delta_message_generator(
        step3p5_tool_parser, step3p5_tokenizer, model_output, request
    ):
        if delta_message.content:
            other_content += delta_message.content

        if delta_message.tool_calls:
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
                    assert tool_call.type == "function"
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    # Verify content was streamed
    assert "Let me check the weather for you:" in other_content

    # Verify we got the tool call
    assert len(tool_states) == 1
    state = tool_states[0]
    assert state["id"] is not None
    assert state["type"] == "function"
    assert state["name"] == "get_current_weather"

    # Verify arguments were parsed correctly despite missing closing tag
    assert state["arguments"] is not None
    args = json.loads(state["arguments"])
    assert args["city"] == "Dallas"
    assert args["state"] == "TX"
    assert args["unit"] == "fahrenheit"


def test_extract_tool_calls_streaming_incremental(
    step3p5_tool_parser, step3p5_tokenizer, sample_tools
):
    """Test that streaming is truly incremental"""
    model_output = """I'll check the weather.<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    chunks = []
    for delta_message in stream_delta_message_generator(
        step3p5_tool_parser, step3p5_tokenizer, model_output, request
    ):
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
            assert chunk.tool_calls[0].function.name == "get_current_weather"
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


def test_extract_tool_calls_complex_type_with_single_quote(step3p5_tool_parser):
    """Test parameter type conversion based on tool schema"""
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "test_types",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "int_param": {"type": "integer"},
                        "float_param": {"type": "float"},
                        "bool_param": {"type": "boolean"},
                        "str_param": {"type": "string"},
                        "obj_param": {"type": "object"},
                    },
                },
            },
        )
    ]

    model_output = """<tool_call>
<function=test_types>
<parameter=obj_param>
{'key': 'value'}
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=request
    )

    args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args["obj_param"] == {"key": "value"}


def test_extract_tool_calls_streaming_mixed_content_and_multiple_tool_calls(
    step3p5_tool_parser, step3p5_tokenizer, sample_tools
):
    """Test mixed content with multiple complete tool calls.

    Scenario: Model outputs "hello" + complete tool call + "hi" + complete tool call.
    Expected: "hello" as content, first tool call parsed (index=0), "hi" as content,
    second tool call parsed (index=1).
    """
    # Model output: hello + complete tool call + hi + complete tool call
    model_output = """hello<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call>hi<tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10, "height": 5}
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    other_content = ""
    tool_states = {}

    for delta_message in stream_delta_message_generator(
        step3p5_tool_parser, step3p5_tokenizer, model_output, request
    ):
        if delta_message.content:
            other_content += delta_message.content

        if delta_message.tool_calls:
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
                    assert tool_call.type == "function"
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    # Should have exactly two complete tool calls
    assert len(tool_states) == 2, "Should have exactly two complete tool calls"

    # Verify the first tool call (index=0)
    assert tool_states[0]["name"] == "get_current_weather"
    assert tool_states[0]["arguments"]
    args_dict_0 = json.loads(tool_states[0]["arguments"])
    assert args_dict_0["city"] == "Dallas"
    assert args_dict_0["state"] == "TX"

    # Verify the second tool call (index=1)
    assert tool_states[1]["name"] == "calculate_area"
    assert tool_states[1]["arguments"]
    args_dict_1 = json.loads(tool_states[1]["arguments"])
    assert args_dict_1["shape"] == "rectangle"
    assert isinstance(args_dict_1["dimensions"], dict), "dimensions should be a dict"
    assert args_dict_1["dimensions"]["width"] == 10
    assert args_dict_1["dimensions"]["height"] == 5
    # Verify content: should contain "hello", "hi"
    assert "hello" in other_content, "Should contain 'hello' as content"
    assert "hi" in other_content, "Should contain 'hi' as content"

    # Verify the order: hello should come first, then hi
    hello_index = other_content.find("hello")
    hi_index = other_content.find("hi")

    assert hello_index >= 0, "'hello' should be in content"
    assert hi_index > hello_index, "'hi' should come after 'hello'"

    # Verify that tool call tags are NOT in the content
    # We should not see complete tool call structures in content
    assert "<function=get_current_weather>" not in other_content, (
        "First tool call should not be in content"
    )
    assert "<function=calculate_area>" not in other_content, (
        "Second tool call should not be in content"
    )


def test_extract_tool_calls_non_streaming_mixed_content_and_multiple_tool_calls(
    step3p5_tool_parser, sample_tools
):
    """Test non-streaming extraction with mixed content and multiple tool calls.

    Scenario: Model outputs "hello" + complete tool call + "hi" + complete tool call.
    Expected: "hello" as content, first tool call parsed (index=0), "hi" as content,
    second tool call parsed (index=1)
    """
    # Model output: hello + complete tool call + hi + complete tool call
    model_output = """hello<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call>hi<tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10, "height": 5}
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=request
    )

    # Should have exactly two complete tool calls
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2, (
        "Should have exactly two complete tool calls"
    )

    # Verify the first tool call (index=0)
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_weather"
    args_dict_0 = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args_dict_0["city"] == "Dallas"
    assert args_dict_0["state"] == "TX"

    # Verify the second tool call (index=1)
    assert extracted_tool_calls.tool_calls[1].function.name == "calculate_area"
    args_dict_1 = json.loads(extracted_tool_calls.tool_calls[1].function.arguments)
    assert args_dict_1["shape"] == "rectangle"
    assert isinstance(args_dict_1["dimensions"], dict), "dimensions should be a dict"
    assert args_dict_1["dimensions"]["width"] == 10
    assert args_dict_1["dimensions"]["height"] == 5

    # Verify content: should contain "hello", "hi"
    assert extracted_tool_calls.content is not None
    assert "hello" in extracted_tool_calls.content, "Should contain 'hello' as content"
    assert "hi" in extracted_tool_calls.content, "Should contain 'hi' as content"

    # Verify the order: hello should come first, then hi
    hello_index = extracted_tool_calls.content.find("hello")
    hi_index = extracted_tool_calls.content.find("hi")

    assert hello_index >= 0, "'hello' should be in content"
    assert hi_index > hello_index, "'hi' should come after 'hello'"

    # Verify that tool call tags are NOT in the content
    assert "<function=get_current_weather>" not in extracted_tool_calls.content, (
        "First tool call should not be in content"
    )
    assert "<function=calculate_area>" not in extracted_tool_calls.content, (
        "Second tool call should not be in content"
    )


def test_extract_tool_calls_streaming_full_input_mixed_content_and_multiple_tool_calls(
    step3p5_tool_parser, step3p5_tokenizer, sample_tools
):
    """Test streaming with entire input as single delta_text.

    Scenario: Model outputs "hello" + complete tool call + "hi" + complete tool call.
    This test simulates the case where the entire input is sent as a single delta_text.
    Expected: "hello" as content, first tool call parsed (index=0), "hi" as content,
    second tool call parsed (index=1).
    """
    # Model output: hello + complete tool call + hi + complete tool call
    model_output = """hello<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call>hi<tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10, "height": 5}
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    other_content = ""
    tool_states = {}

    # Encode all content tokens at once
    all_token_ids = step3p5_tokenizer.encode(model_output, add_special_tokens=False)
    eos_token_id = getattr(step3p5_tokenizer, "eos_token_id", None)

    # Include EOS token in delta_token_ids if available
    if eos_token_id is not None:
        delta_token_ids = all_token_ids + [eos_token_id]
    else:
        delta_token_ids = all_token_ids

    # current_token_ids includes all content tokens (EOS is not part of the text)
    current_token_ids = all_token_ids
    previous_token_ids: list[int] = []

    # Decode all tokens to get the full text
    current_text = step3p5_tokenizer.decode(
        current_token_ids, skip_special_tokens=False
    )
    previous_text = ""
    delta_text = current_text

    # Call parser once with all tokens including EOS
    delta_result = step3p5_tool_parser.extract_tool_calls_streaming(
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request=request,
    )

    # Process delta result
    if delta_result:
        if delta_result.content:
            other_content += delta_result.content
        if delta_result.tool_calls:
            for tool_call in delta_result.tool_calls:
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
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    # Should have exactly two complete tool calls
    assert len(tool_states) == 2, "Should have exactly two complete tool calls"

    # Verify the first tool call (index=0)
    assert tool_states[0]["name"] == "get_current_weather"
    assert tool_states[0]["arguments"]
    args_dict_0 = json.loads(tool_states[0]["arguments"])
    assert args_dict_0["city"] == "Dallas"
    assert args_dict_0["state"] == "TX"

    # Verify the second tool call (index=1)
    assert tool_states[1]["name"] == "calculate_area"
    assert tool_states[1]["arguments"]
    args_dict_1 = json.loads(tool_states[1]["arguments"])
    assert args_dict_1["shape"] == "rectangle"
    assert isinstance(args_dict_1["dimensions"], dict), "dimensions should be a dict"
    assert args_dict_1["dimensions"]["width"] == 10
    assert args_dict_1["dimensions"]["height"] == 5

    # Verify content: should contain "hello", "hi"
    assert "hello" in other_content, "Should contain 'hello' as content"
    assert "hi" in other_content, "Should contain 'hi' as content"

    # Verify the order: hello should come first, then hi
    hello_index = other_content.find("hello")
    hi_index = other_content.find("hi")

    assert hello_index >= 0, "'hello' should be in content"
    assert hi_index > hello_index, "'hi' should come after 'hello'"

    # Verify that tool call tags are NOT in the content
    assert "<function=get_current_weather>" not in other_content, (
        "First tool call should not be in content"
    )
    assert "<function=calculate_area>" not in other_content, (
        "Second tool call should not be in content"
    )


def test_extract_tool_calls_streaming_multiple_tool_calls_no_content_between(
    step3p5_tool_parser, step3p5_tokenizer, sample_tools
):
    """Test multiple tool calls with no content between them.

    Scenario: Model outputs "hello" + tool call + tool call
    Expected: "hello" as content, first tool call parsed (index=0),
    second tool call parsed (index=1).
    No content should appear between the two tool calls.
    """
    # Model output: hello + tool call + tool call (no content between tool calls)
    model_output = """hello<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call><tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10, "height": 5}
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    other_content = ""
    tool_states = {}

    for delta_message in stream_delta_message_generator(
        step3p5_tool_parser, step3p5_tokenizer, model_output, request
    ):
        if delta_message.content:
            other_content += delta_message.content

        if delta_message.tool_calls:
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
                    assert tool_call.type == "function"
                    tool_states[idx]["type"] = tool_call.type

                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name

                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    # Should have exactly two complete tool calls
    assert len(tool_states) == 2, "Should have exactly two complete tool calls"

    # Verify the first tool call (index=0)
    assert tool_states[0]["name"] == "get_current_weather"
    assert tool_states[0]["arguments"]
    args_dict_0 = json.loads(tool_states[0]["arguments"])
    assert args_dict_0["city"] == "Dallas"
    assert args_dict_0["state"] == "TX"

    # Verify the second tool call (index=1)
    assert tool_states[1]["name"] == "calculate_area"
    assert tool_states[1]["arguments"]
    args_dict_1 = json.loads(tool_states[1]["arguments"])
    assert args_dict_1["shape"] == "rectangle"
    assert isinstance(args_dict_1["dimensions"], dict), "dimensions should be a dict"
    assert args_dict_1["dimensions"]["width"] == 10
    assert args_dict_1["dimensions"]["height"] == 5

    assert "hello" in other_content, "Should contain 'hello' as content"

    # Verify that tool call tags are NOT in the content
    assert "<function=get_current_weather>" not in other_content, (
        "First tool call should not be in content"
    )
    assert "<function=calculate_area>" not in other_content, (
        "Second tool call should not be in content"
    )


def test_extract_tool_calls_streaming_multi_token_chunk_boundary(
    step3p5_tool_parser, step3p5_tokenizer, sample_tools
):
    """Ensure fallback doesn't close a new tool_call when boundary is in one chunk."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    delta_text_chunks = [
        """<tool_call>
<function=get_current_weather>
<parameter=city>
Sys""",
        """
</parameter>
</function>
""",
        """</tool_call><tool_call>
<""",
        """function=calculate_area>
<parameter=shape>
rectangle""",
        """</parameter>
</function>
</tool_call>""",
    ]
    boundary_chunk = delta_text_chunks[1]
    assert len(step3p5_tokenizer.encode(boundary_chunk, add_special_tokens=False)) > 1

    tool_states = {}
    for delta_message in stream_delta_message_generator_from_chunks(
        step3p5_tool_parser, step3p5_tokenizer, delta_text_chunks, request
    ):
        print(delta_message)
        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                idx = tool_call.index
                if idx not in tool_states:
                    tool_states[idx] = {
                        "name": None,
                        "arguments": "",
                    }
                if tool_call.function:
                    if tool_call.function.name:
                        tool_states[idx]["name"] = tool_call.function.name
                    if tool_call.function.arguments is not None:
                        tool_states[idx]["arguments"] += tool_call.function.arguments

    assert len(tool_states) == 2
    assert all(state["name"] for state in tool_states.values())
    assert tool_states[0]["name"] == "get_current_weather"
    assert tool_states[1]["name"] == "calculate_area"


def test_extract_tool_calls_non_streaming_multiple_tool_calls_no_content_between(
    step3p5_tool_parser, sample_tools
):
    """Test non-streaming extraction with tool calls and no content between them.

    Scenario: Model outputs "hello" + tool call + tool call.
    Expected: "hello" as content, first tool call parsed (index=0),
    second tool call parsed (index=1).
    No content should appear between the two tool calls.
    """
    # Model output: hello + tool call + tool call (no content between tool calls)
    model_output = """hello<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>
</tool_call><tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10, "height": 5}
</parameter>
</function>
</tool_call>"""

    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    extracted_tool_calls = step3p5_tool_parser.extract_tool_calls(
        model_output, request=request
    )

    # Should have exactly two complete tool calls
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2, (
        "Should have exactly two complete tool calls"
    )

    # Verify the first tool call (index=0)
    assert extracted_tool_calls.tool_calls[0].function.name == "get_current_weather"
    args_dict_0 = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args_dict_0["city"] == "Dallas"
    assert args_dict_0["state"] == "TX"

    # Verify the second tool call (index=1)
    assert extracted_tool_calls.tool_calls[1].function.name == "calculate_area"
    args_dict_1 = json.loads(extracted_tool_calls.tool_calls[1].function.arguments)
    assert args_dict_1["shape"] == "rectangle"
    assert isinstance(args_dict_1["dimensions"], dict), "dimensions should be a dict"
    assert args_dict_1["dimensions"]["width"] == 10
    assert args_dict_1["dimensions"]["height"] == 5

    # Verify content: should contain "hello"
    assert extracted_tool_calls.content is not None
    assert "hello" in extracted_tool_calls.content, "Should contain 'hello' as content"

    # Verify that tool call tags are NOT in the content
    assert "<function=get_current_weather>" not in extracted_tool_calls.content, (
        "First tool call should not be in content"
    )
    assert "<function=calculate_area>" not in extracted_tool_calls.content, (
        "Second tool call should not be in content"
    )
