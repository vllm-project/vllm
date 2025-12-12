# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.tool_parsers.minimax_m2_tool_parser import (
    MinimaxM2ToolParser,
)

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    """Minimal fake tokenizer that exposes the attributes used by the
    parser: a truthy model_tokenizer marker and a vocab mapping for the
    special tokens.
    """

    def __init__(self):
        self.model_tokenizer = True
        # The parser will look up start/end tokens by their literal strings
        self.vocab = {
            "<minimax:tool_call>": 1,
            "</minimax:tool_call>": 2,
        }


@pytest.fixture
def minimax_m2_tool_parser():
    return MinimaxM2ToolParser(FakeTokenizer())


def test_extract_tool_calls_streaming_incremental(minimax_m2_tool_parser):
    """Simulate incremental streaming of a <minimax:tool_call> and ensure
    no IndexError is raised and fragments are returned as expected.
    """
    parser = minimax_m2_tool_parser

    # Reset parser streaming state to ensure a fresh start
    parser._reset_streaming_state()

    stages = [
        # header with invoke start
        '<minimax:tool_call><invoke name="get_weather">',
        # parameter appears
        '<parameter name="city">Seattle</parameter>',
        # close invoke and tool_call
        "</invoke></minimax:tool_call>",
    ]

    previous = ""
    saw_name = False
    saw_param = False
    saw_close = False

    for i, current in enumerate(stages):
        delta = current[len(previous) :]

        # Call the streaming extractor; it must not raise IndexError
        result = None
        try:
            result = parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=current if current is not None else "",
                delta_text=delta,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=None,
            )
        except Exception as e:
            pytest.fail(f"extract_tool_calls_streaming raised an exception: {e}")

        # Inspect returned DeltaMessage (if any) for expected fragments
        if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
            tc = result.tool_calls[0]
            # Stage 0: header should include function name
            if i == 0:
                assert tc.function is not None and tc.function.name == "get_weather"
                saw_name = True
            # Stage 1: parameter fragment should include the parameter name/value
            if i == 1:
                assert (
                    tc.function is not None
                    and tc.function.arguments == '{"city":"Seattle"}'
                )
                saw_param = True
            # Stage 2: closing brace/fragment expected
            if i == 2:
                # Parser returns closing '}' fragment when finishing
                assert tc.function is not None and tc.function.arguments == "}"
                saw_close = True

        previous = current

    assert saw_name, "Expected function name to be sent during streaming"
    assert saw_param, "Expected parameter fragment to be sent during streaming"
    assert saw_close, "Expected tool call close fragment to be sent"


def test_streaming_minimax_m2_multiple_invokes(minimax_m2_tool_parser):
    """Comprehensive streaming test that simulates the MiniMax M2
    tool_call output containing two <invoke> blocks. Verifies the parser
    does not raise, emits header/param fragments, and populates
    prev_tool_call_arr with parsed JSON arguments at completion.
    """
    parser = minimax_m2_tool_parser

    # Reset streaming state
    parser._reset_streaming_state()

    full = (
        "<minimax:tool_call>\n"
        '<invoke name="search_web">\n'
        '<parameter name="query_tag">["technology", "events"]</parameter>\n'
        '<parameter name="query_list">["\\"OpenAI\\" \\"latest\\" \\"release\\""]\n'
        "</parameter>\n"
        "</invoke>\n"
        '<invoke name="search_web">\n'
        '<parameter name="query_tag">["technology", "events"]</parameter>\n'
        '<parameter name="query_list">["\\"Gemini\\" \\"latest\\" \\"release\\""]\n'
        "</parameter>\n"
        "</invoke>\n"
        "</minimax:tool_call>"
    )

    # Stream in reasonable chunks to simulate real streaming
    chunks = []
    # split by lines to produce streaming-like deltas
    for line in full.split("\n"):
        chunks.append(line + "\n")

    previous = ""
    headers_seen = 0

    for chunk in chunks:
        current = previous + chunk
        delta = chunk
        try:
            res = parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=current,
                delta_text=delta,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=None,
            )
        except Exception as e:
            pytest.fail(f"extract_tool_calls_streaming raised an exception: {e}")

        # Count header messages (function name present)
        if res is not None and hasattr(res, "tool_calls") and res.tool_calls:
            for tc in res.tool_calls:
                if tc.function and getattr(tc.function, "name", None):
                    headers_seen += 1

        previous = current

    # After streaming all chunks, parser.prev_tool_call_arr should contain
    # two completed tool calls with parsed argument JSON strings.
    assert len(parser.prev_tool_call_arr) == 2, (
        f"Expected 2 completed tool calls, got {len(parser.prev_tool_call_arr)}"
    )

    import json

    for entry in parser.prev_tool_call_arr:
        assert entry["name"] == "search_web"
        assert "arguments" in entry
        args = json.loads(entry["arguments"])
        # query_tag should be a list
        assert isinstance(args.get("query_tag"), list)
        assert args["query_tag"] == ["technology", "events"]
        assert isinstance(args.get("query_list"), list)
        joined = args["query_list"][0]
        assert ("OpenAI" in joined) or ("Gemini" in joined)

    # Ensure we observed at least two headers (one per invoke)
    assert headers_seen >= 2, f"Expected at least 2 header events, saw {headers_seen}"
