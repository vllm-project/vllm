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


# ---------------------------------------------------------------------------
# XML-specific streaming bugs (Coder parser is not affected)
# ---------------------------------------------------------------------------


def test_xml_streaming_boolean_true_not_false(qwen3_tokenizer):
    """
    Bug B: In streaming mode, a boolean parameter with value "true" is
    streamed as "false".

    Root cause: When "true" arrives character by character:
      - 't' → _convert_param_value("t", "boolean") = False → emits "false"
      - 'r','u','e' → no new delta (output_data[len("false"):] = "")
    Final accumulated arguments contain "false" instead of "true".
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "set_flag",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    # Feed character-by-character to trigger the streaming accumulation bug.
    # Each chunk simulates a single-character token arriving in streaming.
    char_deltas = [
        "<tool_call>",
        "\n<function=set_flag>",
        "\n<parameter=enabled>",
        "t",   # ← first char triggers False → emits "false"
        "r",
        "u",
        "e",   # ← full "true" but delta = "true"[5:] = ""
        "</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        char_deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)

    assert args["enabled"] is True, (
        f"Boolean streaming bug: expected True, got {args['enabled']!r}. "
        f"First char 't' emits 'false'; subsequent chars emit nothing; "
        f"final value is 'false' even though the model said 'true'."
    )


def test_xml_streaming_string_null_last_char_not_dropped(qwen3_tokenizer):
    """
    Bug A (streaming variant): String parameter with value "null" loses
    the last character 'l' when tokens arrive one by one.

    Root cause: Accumulating 'n','u','l' emits correctly, but on the
    fourth char 'l' the full value is "null" →
    _convert_param_value("null", "string") → None →
    _convert_for_json_streaming(None, "string") → "" → delta = ""[3:] = "".
    The closing quote is then emitted, yielding "nul" not "null".
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        )
    ]

    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)

    char_deltas = [
        "<tool_call>",
        "\n<function=search>",
        "\n<parameter=query>",
        "n",
        "u",
        "l",
        "l",   # ← triggers _convert_param_value("null",…) = None → nothing emitted
        "</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ]

    reconstructor = run_tool_extraction_streaming(
        parser,
        char_deltas,
        request,
        assert_one_tool_per_delta=False,
    )

    assert len(reconstructor.tool_calls) == 1
    args = json.loads(reconstructor.tool_calls[0].function.arguments)

    assert "query" in args
    assert args["query"] == "null", (
        f"String 'null' streaming bug: last 'l' was dropped. "
        f"Got: {args['query']!r}. "
        f"When full value reaches 'null', _convert_param_value returns None "
        f"and _convert_for_json_streaming(None, 'string') returns '', "
        f"so the final delta is empty and the 'l' is never emitted."
    )
