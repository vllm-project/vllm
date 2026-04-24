# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser
from tests.tool_parsers.utils import run_tool_extraction_streaming

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)

class TestQwen3xmlToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="qwen3_xml",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>",
            parallel_tool_calls_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call><tool_call>\n<function=get_time>\n<parameter=timezone>Asia/Tokyo</parameter>\n</function>\n</tool_call>",
            various_data_types_output=(
                "<tool_call>\n<function=test_function>\n"
                "<parameter=string_field>hello</parameter>\n"
                "<parameter=int_field>42</parameter>\n"
                "<parameter=float_field>3.14</parameter>\n"
                "<parameter=bool_field>true</parameter>\n"
                "<parameter=null_field>null</parameter>\n"
                '<parameter=array_field>["a", "b", "c"]</parameter>\n'
                '<parameter=object_field>{"nested": "value"}</parameter>\n'
                "</function>\n</tool_call>"
            ),
            empty_arguments_output="<tool_call>\n<function=refresh>\n</function>\n</tool_call>",
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>Tokyo</parameter>\n"
                "</function>\n</tool_call>\n\n"
                "I will get that information."
            ),
            escaped_strings_output=(
                "<tool_call>\n<function=test_function>\n"
                '<parameter=quoted>He said "hello"</parameter>\n'
                "<parameter=path>C:\\Users\\file.txt</parameter>\n"
                "<parameter=newline>line1\nline2</parameter>\n"
                "</function>\n</tool_call>"
            ),
            malformed_input_outputs=[
                "<tool_call><function=func>",
                "<tool_call><function=></function></tool_call>",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            supports_typed_arguments=False,
        )

    def test_qwen3xml_async_streaming_free_text(self, qwen3_tokenizer):
        parser = Qwen3XMLToolParser(qwen3_tokenizer)
        
        # 1. First tool call
        # 2. Free text
        # 3. Second tool call
        text_to_stream = (
            "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
            "\nNext, I will check the weather for London:\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>London</parameter>\n</function>\n</tool_call>"
        )
        
        request = ChatCompletionRequest(messages=[], model="test")
        emitted_messages = []
        previous_text = ""
        previous_tokens = []
        token_ids = qwen3_tokenizer.encode(text_to_stream, add_special_tokens=False)
        
        for i in range(1, len(token_ids) + 1):
            current_token_ids = token_ids[:i]
            current_text = qwen3_tokenizer.decode(current_token_ids)
            delta_text = current_text[len(previous_text):]
            token_delta = current_token_ids[len(previous_tokens):]
            
            delta = parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_tokens,
                current_token_ids,
                token_delta,
                request
            )
            if delta is not None:
                emitted_messages.append(delta)
                
            previous_text = current_text
            previous_tokens = current_token_ids

        # Check that the free text is emitted BEFORE London's arguments are emitted.
        found_early = False
        accumulated_content = ""
        for i, msg in enumerate(emitted_messages):
            if msg.content:
                accumulated_content += msg.content
            
            if "Next, I will check the weather for London" in accumulated_content:
                # Check if we already saw "London" in any previous or current tool call arguments
                is_london_emitted = any(
                    tc.function.arguments and "London" in tc.function.arguments 
                    for m in emitted_messages[:i+1] if m.tool_calls 
                    for tc in m.tool_calls
                )
                if not is_london_emitted:
                    found_early = True
                break
        
        assert found_early, "Free text between tool calls should be emitted as soon as the second tool call starts, not delayed."

    def test_qwen3xml_streaming_text_after_tool_call(self, qwen3_tokenizer):
        parser = Qwen3XMLToolParser(qwen3_tokenizer)
        
        # Tool call followed by free text
        text_to_stream = (
            "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
            "\nI hope this helps!"
        )
        
        request = ChatCompletionRequest(messages=[], model="test")
        emitted_messages = []
        previous_text = ""
        previous_tokens = []
        token_ids = qwen3_tokenizer.encode(text_to_stream, add_special_tokens=False)
        
        for i in range(1, len(token_ids) + 1):
            current_token_ids = token_ids[:i]
            current_text = qwen3_tokenizer.decode(current_token_ids)
            delta_text = current_text[len(previous_text):]
            token_delta = current_token_ids[len(previous_tokens):]
            
            delta = parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_tokens,
                current_token_ids,
                token_delta,
                request
            )
            if delta is not None:
                emitted_messages.append(delta)
                
            previous_text = current_text
            previous_tokens = current_token_ids

        # Aggregate all emitted content
        all_content = "".join([m.content for m in emitted_messages if m.content])
        
        assert "I hope this helps!" in all_content, "Free text after the last tool call should be emitted."


def test_qwen36_anyof_parameter_xml_not_double_encoded(qwen3_tokenizer):
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "update_record",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # anyOf schema — no top-level "type" key
                        "data": {
                            "anyOf": [{"type": "object"}, {"type": "null"}],
                        },
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    model_output = (
        "<tool_call>\n"
        "<function=update_record>\n"
        '<parameter=data>{"key": "value", "count": 42}</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    result = parser.extract_tool_calls(model_output, request=request)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].function.arguments)

    assert isinstance(args["data"], dict), (
        f"anyOf parameter was double-encoded: data={args['data']!r}. "
        "StreamingXMLToolCallParser._get_param_type ignores anyOf schemas."
    )
    assert args["data"] == {"key": "value", "count": 42}


def test_qwen36_anyof_parameter_xml_streaming_not_double_encoded(qwen3_tokenizer):
  
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "update_record",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "anyOf": [{"type": "object"}, {"type": "null"}],
                        },
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    # Deltas are pre-formed XML element chunks (one element per delta),
    # which is the same pattern used by speculative decoding.
    deltas = [
        "<tool_call>",
        "\n<function=update_record>",
        '\n<parameter=data>{"key": "value", "count": 42}</parameter>',
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert isinstance(args["data"], dict), (
        f"anyOf parameter was double-encoded in streaming: data={args['data']!r}"
    )


def test_qwen36_xml_streaming_double_close_brace(qwen3_tokenizer):
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    deltas = [
        "<tool_call>",
        "\n<function=get_weather>",
        "\n<parameter=city>\nDallas\n</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    full_args = reconstructor.tool_calls[0].function.arguments

    assert not full_args.endswith("}}"), (
        f"XML streaming parser emitted double closing brace: {full_args!r}. "
        "parse_single_streaming_chunks fallback called _end_element('function') twice."
    )
    args = json.loads(full_args)
    assert args == {"city": "Dallas"}


def test_xml_streaming_parallel_tool_calls_preformed_chunks(qwen3_tokenizer):
    """
    Note: in normal token-by-token streaming this rarely triggers because
    the tokenizer splits XML tags across multiple tokens.  It CAN trigger with
    speculative decoding multi-token flushes.
    """
    

    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    deltas = [
        "<tool_call>",
        "\n<function=get_weather>",
        "\n<parameter=city>Paris</parameter>",
        "\n</function>",
        "\n</tool_call>",
        "<tool_call>",
        "\n<function=get_weather>",
        "\n<parameter=city>London</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 2, (
        f"Expected 2 tool calls, got {len(reconstructor.tool_calls)}"
    )

    args0 = json.loads(reconstructor.tool_calls[0].function.arguments)
    args1 = json.loads(reconstructor.tool_calls[1].function.arguments)

    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert reconstructor.tool_calls[1].function.name == "get_weather"
    assert args0 == {"city": "Paris"}, f"First call args wrong: {args0!r}"
    assert args1 == {"city": "London"}, f"Second call args wrong: {args1!r}"
