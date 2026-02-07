# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.tool_parsers.minimax_m2_tool_parser import (
    MinimaxM2ToolParser,
)

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    """Minimal fake tokenizer that exposes the attributes used by the
    parser: a truthy model_tokenizer marker and a vocab mapping for the
    special tokens.
    """

    def __init__(self, base_tokenizer):
        self.model_tokenizer = True
        self.base_tokenizer = base_tokenizer
        self.model_tokenizer = base_tokenizer

        # Add special tokens
        special_tokens = ["<minimax:tool_call>", "</minimax:tool_call>"]
        base_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        # Get token IDs
        start_id = base_tokenizer.convert_tokens_to_ids("<minimax:tool_call>")
        end_id = base_tokenizer.convert_tokens_to_ids("</minimax:tool_call>")

        # Create vocab that includes our special tokens
        self._vocab = base_tokenizer.get_vocab()
        self._vocab["<minimax:tool_call>"] = start_id
        self._vocab["</minimax:tool_call>"] = end_id

        # Also make it accessible as .vocab attribute
        self.vocab = self._vocab

    def get_vocab(self):
        return self._vocab

    def __getattr__(self, name):
        # Delegate all other attributes to base tokenizer
        return getattr(self.base_tokenizer, name)


@pytest.fixture
def minimax_tokenizer():
    """Load MiniMax tokenizer and add special tool tokens."""
    from vllm.transformers_utils.tokenizer import get_tokenizer

    base_tokenizer = get_tokenizer("MiniMaxAI/MiniMax-Text-01")
    return FakeTokenizer(base_tokenizer)


@pytest.fixture
def minimax_m2_tool_parser(minimax_tokenizer):
    return MinimaxM2ToolParser(minimax_tokenizer)


def test_extract_tool_calls_streaming_incremental(minimax_m2_tool_parser):
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()
    chunks = [
        "<minimax:tool_call>",
        '<invoke name="get_weather">',
        '<parameter name="city">',
        "Seattle</parameter>",
        "</invoke></minimax:tool_call>",
    ]
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        delta = chunk
        parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous = current

    assert len(parser.prev_tool_call_arr) == 1
    entry = parser.prev_tool_call_arr[0]

    assert entry["name"] == "get_weather"
    args = entry["arguments"]
    assert args["city"] == "Seattle"


def test_streaming_minimax_m2_multiple_invokes(minimax_m2_tool_parser):
    parser = minimax_m2_tool_parser
    parser._reset_streaming_state()

    chunks = [
        "<minimax:tool_call>",
        '<invoke name="search_web">',
        '<parameter name="query_tag">',
        '["technology", "events"]</parameter>',
        '<parameter name="query_list">',
        '["OpenAI", "latest", "release"]</parameter>',
        "</invoke>",
        '<invoke name="search_web">',
        '<parameter name="query_tag">',
        '["technology", "events"]</parameter>',
        '<parameter name="query_list">',
        '["Gemini", "latest", "release"]</parameter>',
        "</invoke>",
        "</minimax:tool_call>",
    ]
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        delta = chunk
        parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous = current

    assert len(parser.prev_tool_call_arr) == 2

    for entry, expect_model in zip(parser.prev_tool_call_arr, ["OpenAI", "Gemini"]):
        assert entry["name"] == "search_web"
        args = json.dumps(entry["arguments"])
        assert "technology" in args and "events" in args
        assert expect_model in args

    # check streamed_args_for_tool for serving_chat.py
    for index in range(2):
        expected_call = parser.prev_tool_call_arr[index].get("arguments", {})
        expected_call = json.dumps(expected_call)
        actual_call = parser.streamed_args_for_tool[index]
        assert expected_call == actual_call


def test_minimax_m2_streaming_different_intervals_single_invoke(
    minimax_tokenizer,
    minimax_m2_tool_parser: MinimaxM2ToolParser,
):
    parser = minimax_m2_tool_parser

    text = """<minimax:tool_call>
<invoke name="get_weather"><parameter name="city">Seattle</parameter>
</invoke></minimax:tool_call>"""
    tokens = minimax_tokenizer.encode(text)

    interval_sizes = list(range(1, 21))
    interval_sizes.extend([100, 1000, 10_000, 100_000, 1_000_000, 10_000_000])
    for interval in interval_sizes:
        # Reset parser state for each interval test
        parser._reset_streaming_state()

        # The first token is processed separately to simulate server behavior
        previous_text = ""
        delta_messages = []

        # Process first token separately
        first_token = tokens[0:1]
        text_chunk = minimax_tokenizer.decode(first_token)
        current_text = text_chunk

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text_chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)

        # Process remaining tokens in chunks based on interval
        remaining_tokens = tokens[1:]
        for i in range(0, len(remaining_tokens), interval):
            # Get chunk of tokens (up to 'interval' tokens, or whatever's left)
            token_chunk = remaining_tokens[i : i + interval]

            # Decode the chunk
            text_chunk = minimax_tokenizer.decode(token_chunk)
            current_text = previous_text + text_chunk

            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=text_chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=None,
            )
            previous_text = current_text
            if delta is not None:
                delta_messages.append(delta)

        # Verify the results
        assert len(parser.prev_tool_call_arr) == 1
        entry = parser.prev_tool_call_arr[0]
        print(f"\tEntry: {entry}")

        assert entry["name"] == "get_weather"
        args = entry["arguments"]
        assert args["city"] == "Seattle"

        assert len(parser.streamed_args_for_tool) == 1
        streamed_args = parser.streamed_args_for_tool[0]
        assert streamed_args == '{"city": "Seattle"}'

        print("Delta messages:")
        for delta in delta_messages:
            print(delta)
        assert delta_messages[0].tool_calls[0].function.name == "get_weather"

        tool_call_args = "".join(
            delta.tool_calls[0].function.arguments or ""
            for delta in delta_messages
            if delta.tool_calls  # Skips if None or empty list
        )
        assert tool_call_args == '{"city": "Seattle"}'

        print(f"\tInterval: {interval} - PASSED\n")


def test_minimax_m2_streaming_different_intervals_multiple_invokes(
    minimax_tokenizer,
    minimax_m2_tool_parser: MinimaxM2ToolParser,
):
    parser = minimax_m2_tool_parser

    text = """<minimax:tool_call>
<invoke name="search_web"><parameter name="query_tag">["technology", "events"]
</parameter><parameter name="query_list">["OpenAI", "latest", "release"]
</parameter></invoke><invoke name="search_web"><parameter name="query_tag">
["technology", "events"]</parameter><parameter name="query_list">["Gemini", 
"latest", "release"]</parameter></invoke></minimax:tool_call>"""
    tokens = minimax_tokenizer.encode(text)

    interval_sizes = list(range(1, 21))
    interval_sizes.extend([100, 1000, 10_000, 100_000, 1_000_000, 10_000_000])
    for interval in interval_sizes:
        # Reset parser state for each interval test
        parser._reset_streaming_state()

        # The first token is processed separately to simulate server behavior
        previous_text = ""
        delta_messages = []

        # Process first token separately
        first_token = tokens[0:1]
        text_chunk = minimax_tokenizer.decode(first_token)
        current_text = text_chunk

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text_chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)

        # Process remaining tokens in chunks based on interval
        remaining_tokens = tokens[1:]
        for i in range(0, len(remaining_tokens), interval):
            # Get chunk of tokens (up to 'interval' tokens, or whatever's left)
            token_chunk = remaining_tokens[i : i + interval]

            # Decode the chunk
            text_chunk = minimax_tokenizer.decode(token_chunk)
            current_text = previous_text + text_chunk

            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=text_chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=None,
            )
            previous_text = current_text
            if delta is not None:
                delta_messages.append(delta)

        # Verify the results
        assert len(parser.prev_tool_call_arr) == 2

        entry1 = parser.prev_tool_call_arr[0]
        print(f"\tEntry1: {entry1}")
        assert entry1["name"] == "search_web"
        args = entry1["arguments"]
        assert args == {
            "query_tag": '["technology", "events"]',
            "query_list": '["OpenAI", "latest", "release"]',
        }

        entry2 = parser.prev_tool_call_arr[1]
        print(f"\tEntry2: {entry2}")
        assert entry2["name"] == "search_web"
        args = entry2["arguments"]
        assert args == {
            "query_tag": '["technology", "events"]',
            "query_list": '["Gemini", \n"latest", "release"]',
        }

        assert len(parser.streamed_args_for_tool) == 2
        for index in range(2):
            expected_call = parser.prev_tool_call_arr[index].get("arguments", {})
            expected_call = json.dumps(expected_call)
            actual_call = parser.streamed_args_for_tool[index]
            assert expected_call == actual_call

        print("Delta messages:")
        for delta in delta_messages:
            print(delta)
        assert delta_messages[0].tool_calls[0].function.name == "search_web"

        tool_call_args = "".join(
            tool_call.function.arguments or ""
            for delta in delta_messages
            if delta.tool_calls
            for tool_call in delta.tool_calls  # Iterate through all tool calls
        )

        expected_calls = [
            {
                "query_tag": '["technology", "events"]',
                "query_list": '["OpenAI", "latest", "release"]',
            },
            {
                "query_tag": '["technology", "events"]',
                "query_list": '["Gemini", \n"latest", "release"]',
            },
        ]
        expected = "".join(json.dumps(call) for call in expected_calls)
        assert tool_call_args == expected

        print(f"Interval: {interval} - PASSED\n")
