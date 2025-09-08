# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
from openai_harmony import StreamState

from vllm.entrypoints.context import HarmonyContext, StreamingHarmonyContext
from vllm.outputs import CompletionOutput, RequestOutput


# Helper function for Python < 3.10 compatibility
async def async_next(async_iterator):
    """Compatibility function equivalent to Python 3.10's anext()."""
    return await async_iterator.__anext__()


def create_mock_request_output(
    prompt_token_ids=None,
    output_token_ids=None,
    num_cached_tokens=0,
    finished=True,
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
        finished=finished,
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
        parser.state = StreamState.EXPECT_START
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
    assert not context.is_first_turn
    assert context.previous_turn.input_tokens == 5
    assert context.previous_turn.output_tokens == 3


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
    assert context.is_first_turn is False  # Should be updated after first turn


@pytest.mark.asyncio
async def test_negative_tool_tokens_edge_case():
    """Test edge case where calculation could result in negative tool
    tokens. We should log an error and clamp the value to 0."""
    # Use patch to check if logger.error was called
    with patch("vllm.entrypoints.context.logger.error") as mock_log:
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

        # Calculated negative tool tokens (12 - 10 - 5 = -3) should be clamped
        # to 0 and an error should be logged
        assert context.num_tool_output_tokens == 0
        assert context.num_prompt_tokens == 10 + 12
        assert context.num_output_tokens == 5 + 2

        # Verify the error was logged properly
        mock_log.assert_called_once()

        # Extract the actual log message and arguments from the call
        args, _ = mock_log.call_args
        log_message = args[0]

        # Check for key parts of the message
        assert "Negative tool output tokens calculated" in log_message
        assert "-3" in str(args)  # Check that -3 is in the arguments


@pytest.mark.asyncio
async def test_streaming_multi_turn_token_counting(mock_parser):
    """Test token counting for streaming multi-turn conversations.
    
    This test focuses on how StreamingHarmonyContext counts tokens in a 
    multi-turn conversation with streaming (token-by-token) outputs and 
    message boundaries.
    """
    # Create a streaming context
    context = StreamingHarmonyContext(messages=[], available_tools=["browser"])

    # Simulate three turns of conversation:
    # Turn 1: stream tokens one by one, then finish the message
    # Turn 2: new prompt, stream more tokens with a reasoning segment
    # Turn 3: new prompt with tool output and cached tokens

    # First turn: 3 tokens streamed one by one
    # First token of first turn
    context.append_output(
        create_mock_request_output(
            prompt_token_ids=[1, 2, 3],  # 3 prompt tokens
            output_token_ids=[101],  # Single token
            num_cached_tokens=0,
            finished=False,  # Not end of message yet
        ))

    # Second token of first turn
    context.append_output(
        create_mock_request_output(
            output_token_ids=[102],
            finished=False,
        ))

    # Last token of first turn (finished=True signals end of message)
    context.append_output(
        create_mock_request_output(
            output_token_ids=[103],
            finished=True,  # End of message
        ))

    # Check token counts after first turn
    assert context.num_prompt_tokens == 3  # Initial prompt tokens
    assert context.num_output_tokens == 3  # Three output tokens
    assert context.num_cached_tokens == 0
    assert context.num_tool_output_tokens == 0  # No tool output in first turn
    assert context.first_tok_of_message is True  # Ready for next message

    # Second turn: reasoning tokens in analysis channel
    mock_parser.current_channel = "analysis"  # Set to reasoning channel

    # First token of second turn
    context.append_output(
        create_mock_request_output(
            prompt_token_ids=[1, 2, 3, 101, 102, 103, 4,
                              5],  # 8 tokens (includes previous)
            output_token_ids=[201],
            num_cached_tokens=3,  # Some tokens cached
            finished=False,
        ))

    # More tokens in reasoning channel
    context.append_output(
        create_mock_request_output(
            output_token_ids=[202],
            finished=False,
        ))

    context.append_output(
        create_mock_request_output(
            output_token_ids=[203],
            finished=True,  # End of reasoning message
        ))

    # Check counts after second turn (reasoning message)
    assert context.num_prompt_tokens == 3 + 8  # Initial + second prompt
    assert context.num_output_tokens == 3 + 3  # First turn + second turn
    assert context.num_reasoning_tokens == 3  # All tokens in analysis channel
    assert context.num_cached_tokens == 3  # Cached tokens from second turn

    # Formula: this turn prompt tokens - last turn prompt - last turn output
    expected_tool_tokens = 8 - 3 - 3  # = 2
    assert context.num_tool_output_tokens == expected_tool_tokens

    # Third turn: regular output channel
    mock_parser.current_channel = "final"  # Switch back to regular channel

    # Third turn (with more cached tokens)
    context.append_output(
        create_mock_request_output(
            prompt_token_ids=[
                1, 2, 3, 101, 102, 103, 4, 5, 201, 202, 203, 6, 7
            ],  # 13 tokens
            output_token_ids=[301],
            num_cached_tokens=8,  # More cached tokens
            finished=False,
        ))

    context.append_output(
        create_mock_request_output(
            output_token_ids=[302],
            finished=True,
        ))

    # Final token counts check
    assert context.num_prompt_tokens == 3 + 8 + 13  # All prompts
    assert context.num_output_tokens == 3 + 3 + 2  # All outputs
    assert context.num_reasoning_tokens == 3  # Unchanged from second turn
    assert context.num_cached_tokens == 3 + 8  # Accumulated cached tokens

    # Additional tool tokens from third turn
    # Formula: this turn prompt - last turn prompt - last turn output
    additional_tool_tokens = 13 - 8 - 3  # = 2
    assert context.num_tool_output_tokens == expected_tool_tokens \
        + additional_tool_tokens
