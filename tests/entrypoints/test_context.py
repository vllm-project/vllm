# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.context import HarmonyContext
from vllm.outputs import CompletionOutput, RequestOutput


# Helper function for Python < 3.10 compatibility
async def async_next(async_iterator):
    """Compatibility function equivalent to Python 3.10's anext()."""
    return await async_iterator.__anext__()


def create_mock_request_output(
    prompt_token_ids=None,
    output_token_ids=None,
    num_cached_tokens=0,
):
    """Helper function to create a mock RequestOutput object for testing."""
    outputs = []
    token_ids = output_token_ids if output_token_ids is not None else []
    outputs = [
        CompletionOutput(
            index=0,
            text="Test output",
            token_ids=token_ids,
            cumulative_logprob=0.0,
            logprobs=None,
            finish_reason=None,
            stop_reason=None,
        )
    ]

    return RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        outputs=outputs,
        finished=True,
        num_cached_tokens=num_cached_tokens,
    )


async def generate_mock_outputs(num_turns,
                                prompt_token_counts,
                                output_token_counts,
                                cached_token_counts=None):
    """Generate a sequence of mock RequestOutput objects to simulate multiple
    turns."""
    if cached_token_counts is None:
        cached_token_counts = [0] * num_turns

    for i in range(num_turns):
        # Create mock prompt token IDs and output token IDs
        prompt_token_ids = list(range(1, prompt_token_counts[i] + 1))
        output_token_ids = list(range(1, output_token_counts[i] + 1))

        # Create and yield the RequestOutput
        yield create_mock_request_output(
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            num_cached_tokens=cached_token_counts[i],
        )


@pytest.fixture
def mock_parser():
    """Set up a mock parser for tests."""
    with patch("vllm.entrypoints.context.get_streamable_parser_for_assistant"
               ) as mock_parser_factory:
        # Create a mock parser object
        parser = MagicMock()
        parser.messages = []
        parser.current_channel = None
        mock_parser_factory.return_value = parser
        yield parser


def test_single_turn_token_counting():
    """Test token counting behavior for a single turn."""
    # Create a context
    context = HarmonyContext(messages=[], available_tools=[])

    # Create a mock RequestOutput with specific token counts
    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],  # 5 prompt tokens
        output_token_ids=[6, 7, 8],  # 3 output tokens
        num_cached_tokens=2,  # 2 cached tokens
    )

    # Append the output to the context
    context.append_output(mock_output)

    # Verify the token counts
    assert context.num_prompt_tokens == 5
    assert context.num_output_tokens == 3
    assert context.num_cached_tokens == 2
    assert context.num_tool_output_tokens == 0  # No tool tokens in first turn

    # Verify internal state tracking
    assert not context.first_turn
    assert context.num_last_turn_input_tokens == 5
    assert context.num_last_turn_output_tokens == 3


@pytest.mark.asyncio
async def test_multi_turn_token_counting():
    """Test token counting behavior across multiple turns with tool output."""
    # Create a context
    context = HarmonyContext(messages=[], available_tools=["browser"])

    # Simulate a conversation with 3 turns
    # Turn 1: prefill 5, decode 3, tool 7
    # Turn 2: prefill 15, cached 5, decode 4, tool 1
    # Turn 3: prefill 20, cached 15, decode 5
    prompt_token_counts = [5, 15, 20]
    output_token_counts = [3, 4, 5]
    cached_token_counts = [0, 5, 15]
    mock_generator = generate_mock_outputs(3, prompt_token_counts,
                                           output_token_counts,
                                           cached_token_counts)

    # First turn - initial prompt and response
    mock_output1 = await async_next(mock_generator)
    context.append_output(mock_output1)

    # At this point, we should have 5 prompt tokens and 3 output tokens
    assert context.num_prompt_tokens == 5
    assert context.num_output_tokens == 3
    assert context.num_tool_output_tokens == 0

    # Second turn - after tool output
    mock_output2 = await async_next(mock_generator)
    context.append_output(mock_output2)
    # Current prompt tokens (15) - last_turn_input_tokens (5) -
    # last_turn_output_tokens (3) = 7
    expected_tool_output = 7

    assert context.num_prompt_tokens == 5 + 15
    assert context.num_output_tokens == 3 + 4
    assert context.num_tool_output_tokens == expected_tool_output
    assert context.num_cached_tokens == 5

    # Third turn - final response
    mock_output3 = await async_next(mock_generator)
    context.append_output(mock_output3)
    # Additional tool output tokens from third turn:
    # Current prompt (20) - last_turn_input_tokens (15) -
    # last_turn_output_tokens (4) = 1
    expected_tool_output = 7 + 1

    assert context.num_prompt_tokens == 5 + 15 + 20
    assert context.num_output_tokens == 3 + 4 + 5
    assert context.num_tool_output_tokens == expected_tool_output
    assert context.num_cached_tokens == 5 + 15


def test_empty_output_tokens():
    """Test behavior when RequestOutput has empty output tokens."""
    context = HarmonyContext(messages=[], available_tools=[])

    # Create a RequestOutput with empty output tokens
    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3],  # 3 prompt tokens
        output_token_ids=[],  # Empty output tokens list
        num_cached_tokens=1,
    )

    context.append_output(mock_output)

    # Should handle empty outputs gracefully
    assert context.num_prompt_tokens == 3
    assert context.num_output_tokens == 0  # No output tokens
    assert context.num_cached_tokens == 1
    assert context.num_tool_output_tokens == 0


def test_missing_prompt_token_ids():
    """Test behavior when RequestOutput has None prompt_token_ids."""
    context = HarmonyContext(messages=[], available_tools=[])

    mock_output = create_mock_request_output(
        prompt_token_ids=None,  # No prompt token IDs
        output_token_ids=[1, 2],  # 2 output tokens
        num_cached_tokens=0,
    )

    # Logger.error will be called, but we don't need to check for warnings
    # here Just ensure it doesn't raise an exception
    context.append_output(mock_output)

    # Should handle missing prompt tokens gracefully
    assert context.num_prompt_tokens == 0
    assert context.num_output_tokens == 2
    assert context.num_cached_tokens == 0
    assert context.num_tool_output_tokens == 0


def test_reasoning_tokens_counting(mock_parser):
    """Test that reasoning tokens are counted correctly."""
    context = HarmonyContext(messages=[], available_tools=[])

    # Mock parser to simulate reasoning channel
    mock_parser.current_channel = "analysis"  # Reasoning channel

    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[4, 5, 6, 7],  # 4 tokens, all in reasoning
        num_cached_tokens=0,
    )

    context.append_output(mock_output)

    # All output tokens should be counted as reasoning
    assert context.num_reasoning_tokens == 4
    assert context.num_output_tokens == 4


def test_zero_tokens_edge_case():
    """Test behavior with all zero token counts."""
    context = HarmonyContext(messages=[], available_tools=[])

    # Create a request with empty lists (not None) for both prompt and
    # output tokens
    mock_output = create_mock_request_output(
        prompt_token_ids=[],  # Empty prompt tokens
        output_token_ids=[],  # Empty output tokens
        num_cached_tokens=0,
    )

    context.append_output(mock_output)

    # All counts should be zero
    assert context.num_prompt_tokens == 0
    assert context.num_output_tokens == 0
    assert context.num_cached_tokens == 0
    assert context.num_tool_output_tokens == 0
    assert context.num_reasoning_tokens == 0


@pytest.mark.asyncio
async def test_single_turn_no_tool_output():
    """Test that first turn never generates tool output tokens."""
    context = HarmonyContext(
        messages=[],
        available_tools=["browser"]  # Tools available
    )

    # Even with large prompt in first turn, no tool tokens should be counted
    mock_output = create_mock_request_output(
        prompt_token_ids=list(range(100)),  # 100 tokens
        output_token_ids=[1, 2, 3],
        num_cached_tokens=0,
    )

    context.append_output(mock_output)

    # First turn should never have tool output tokens
    assert context.num_tool_output_tokens == 0
    assert context.first_turn is False  # Should be updated after first turn


@pytest.mark.asyncio
async def test_negative_tool_tokens_edge_case():
    """Test edge case where calculation could result in negative tool
    tokens."""
    context = HarmonyContext(messages=[], available_tools=["browser"])

    # First turn
    mock_output1 = create_mock_request_output(
        prompt_token_ids=list(range(10)),  # 10 tokens
        output_token_ids=[1, 2, 3, 4, 5],  # 5 tokens
    )
    context.append_output(mock_output1)

    # Second turn with fewer new tokens than previous output
    # This could happen in edge cases with aggressive caching
    mock_output2 = create_mock_request_output(
        prompt_token_ids=list(range(12)),  # 12 tokens (only 2 new)
        output_token_ids=[6, 7],  # 2 tokens
    )
    context.append_output(mock_output2)

    # Tool tokens = 12 - 10 - 5 = -3, but should be handled gracefully
    # The implementation adds this to tool_output_tokens, so it would be
    # negative
    expected_tool_tokens = 12 - 10 - 5  # -3
    assert context.num_tool_output_tokens == expected_tool_tokens
    assert context.num_prompt_tokens == 10 + 12
    assert context.num_output_tokens == 5 + 2
