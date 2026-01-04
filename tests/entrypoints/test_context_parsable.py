# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.context import ParsableContext, TurnMetrics
from vllm.entrypoints.openai.protocol import ResponsesRequest
from vllm.outputs import CompletionOutput, RequestOutput


def create_mock_request_output(
    prompt_token_ids=None,
    output_token_ids=None,
    num_cached_tokens=0,
    finished=True,
):
    """Helper function to create a mock RequestOutput object for testing."""
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


async def generate_mock_outputs(
    num_turns, prompt_token_counts, output_token_counts, cached_token_counts=None
):
    """Generate a sequence of mock RequestOutput objects to simulate multiple
    turns."""
    if cached_token_counts is None:
        cached_token_counts = [0] * num_turns

    for i in range(num_turns):
        prompt_token_ids = list(range(1, prompt_token_counts[i] + 1))
        output_token_ids = list(range(1, output_token_counts[i] + 1))

        yield create_mock_request_output(
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            num_cached_tokens=cached_token_counts[i],
        )


@pytest.fixture
def mock_tokenizer():
    """Set up a mock tokenizer for tests."""
    tokenizer = MagicMock()
    tokenizer.all_special_ids = set()
    tokenizer.decode.return_value = "test"
    return tokenizer


@pytest.fixture
def mock_reasoning_parser_cls():
    """Set up a mock reasoning parser class for tests."""
    mock_parser = MagicMock()
    mock_parser.extract_reasoning_streaming.return_value = MagicMock(
        reasoning=None, content="test"
    )
    mock_parser.is_reasoning_end_streaming.return_value = False

    def create_parser(tokenizer):
        return mock_parser

    return create_parser


@pytest.fixture
def mock_request():
    """Set up a mock ResponsesRequest for tests."""
    return ResponsesRequest(input="test", model="test-model")


@pytest.fixture
def mock_streamable_parser():
    """Set up a mock StreamableResponsesParser."""
    with patch(
        "vllm.entrypoints.context.get_streamable_responses_parser"
    ) as mock_factory:
        parser = MagicMock()
        parser.response_messages = []
        parser.current_channel = "final"
        parser.final_output = []
        mock_factory.return_value = parser
        yield parser


def create_parsable_context(
    mock_tokenizer,
    mock_reasoning_parser_cls,
    mock_request,
    available_tools=None,
):
    """Helper to create a ParsableContext with common mock dependencies."""
    return ParsableContext(
        response_messages=[],
        tokenizer=mock_tokenizer,
        reasoning_parser_cls=mock_reasoning_parser_cls,
        request=mock_request,
        available_tools=available_tools,
        tool_parser_cls=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def test_parsable_context_init(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test ParsableContext initialization."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    # Verify initial state
    assert context.num_prompt_tokens == 0
    assert context.num_output_tokens == 0
    assert context.num_cached_tokens == 0
    assert context.num_reasoning_tokens == 0
    assert context.num_tool_output_tokens == 0
    assert context.is_first_turn is True
    assert len(context.all_turn_metrics) == 0


def test_parsable_context_single_turn_token_counting(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test token counting behavior for a single turn."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],  # 5 prompt tokens
        output_token_ids=[6, 7, 8],  # 3 output tokens
        num_cached_tokens=2,  # 2 cached tokens
    )

    context.append_output(mock_output)

    # Verify the token counts
    assert context.num_prompt_tokens == 5
    assert context.num_output_tokens == 3
    assert context.num_cached_tokens == 2
    assert context.num_tool_output_tokens == 0  # No tool tokens in first turn

    # Verify internal state tracking
    assert context.is_first_turn is False
    assert len(context.all_turn_metrics) == 1
    previous_turn = context.all_turn_metrics[0]
    assert previous_turn.input_tokens == 5
    assert previous_turn.output_tokens == 3
    assert previous_turn.cached_input_tokens == 2
    assert previous_turn.tool_output_tokens == 0


@pytest.mark.asyncio
async def test_parsable_context_multi_turn_token_counting(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test token counting behavior across multiple turns with tool output."""
    context = create_parsable_context(
        mock_tokenizer,
        mock_reasoning_parser_cls,
        mock_request,
        available_tools=["browser"],
    )

    # Simulate a conversation with 3 turns
    # Turn 1: prefill 5, decode 3, tool 7
    # Turn 2: prefill 15, cached 5, decode 4, tool 1
    # Turn 3: prefill 20, cached 15, decode 5
    prompt_token_counts = [5, 15, 20]
    output_token_counts = [3, 4, 5]
    cached_token_counts = [0, 5, 15]
    mock_generator = generate_mock_outputs(
        3, prompt_token_counts, output_token_counts, cached_token_counts
    )

    # First turn - initial prompt and response
    mock_output1 = await anext(mock_generator)
    context.append_output(mock_output1)

    # At this point, we should have 5 prompt tokens and 3 output tokens
    # num_prompt_tokens is accumulated across all turns
    assert context.num_prompt_tokens == 5
    assert context.num_output_tokens == 3
    assert context.num_tool_output_tokens == 0

    # Second turn - after tool output
    mock_output2 = await anext(mock_generator)
    context.append_output(mock_output2)
    # Current prompt tokens (15) - last_turn_input_tokens (5) -
    # last_turn_output_tokens (3) = 7
    expected_tool_output = 7

    # num_prompt_tokens = 5 (turn1) + 15 (turn2) = 20
    assert context.num_prompt_tokens == 5 + 15
    assert context.num_output_tokens == 3 + 4
    assert context.num_tool_output_tokens == expected_tool_output
    # num_cached_tokens = 0 (turn1) + 5 (turn2) = 5
    assert context.num_cached_tokens == 0 + 5

    # Third turn - final response
    mock_output3 = await anext(mock_generator)
    context.append_output(mock_output3)
    # Additional tool output tokens from third turn:
    # Current prompt (20) - last_turn_input_tokens (15) -
    # last_turn_output_tokens (4) = 1
    expected_tool_output = 7 + 1

    # num_prompt_tokens = 5 (turn1) + 15 (turn2) + 20 (turn3) = 40
    assert context.num_prompt_tokens == 5 + 15 + 20
    assert context.num_output_tokens == 3 + 4 + 5
    assert context.num_tool_output_tokens == expected_tool_output
    # num_cached_tokens = 0 (turn1) + 5 (turn2) + 15 (turn3) = 20
    assert context.num_cached_tokens == 0 + 5 + 15

    # Validate all turn metrics
    assert len(context.all_turn_metrics) == 3
    for i, turn in enumerate(context.all_turn_metrics):
        assert turn.input_tokens == prompt_token_counts[i]
        assert turn.output_tokens == output_token_counts[i]
        assert turn.cached_input_tokens == cached_token_counts[i]
    assert context.all_turn_metrics[1].tool_output_tokens == 7
    assert context.all_turn_metrics[2].tool_output_tokens == 1


def test_parsable_context_empty_output_tokens(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test behavior when RequestOutput has empty output tokens."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3],  # 3 prompt tokens
        output_token_ids=[],  # Empty output tokens list
        num_cached_tokens=1,
    )

    context.append_output(mock_output)

    # Should handle empty outputs gracefully
    assert context.num_prompt_tokens == 3
    assert context.num_output_tokens == 0
    assert context.num_cached_tokens == 1
    assert context.num_tool_output_tokens == 0


def test_parsable_context_missing_prompt_token_ids(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test behavior when RequestOutput has None prompt_token_ids."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    mock_output = create_mock_request_output(
        prompt_token_ids=None,  # No prompt token IDs
        output_token_ids=[1, 2],  # 2 output tokens
        num_cached_tokens=0,
    )

    # Should handle missing prompt tokens gracefully
    context.append_output(mock_output)

    assert context.num_prompt_tokens == 0
    assert context.num_output_tokens == 2
    assert context.num_cached_tokens == 0
    assert context.num_tool_output_tokens == 0


def test_parsable_context_reasoning_tokens_counting(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test that reasoning tokens are counted correctly."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    # Mock parser to simulate reasoning channel
    mock_streamable_parser.current_channel = "analysis"

    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[4, 5, 6, 7],  # 4 tokens, all in reasoning
        num_cached_tokens=0,
    )

    context.append_output(mock_output)

    # All output tokens should be counted as reasoning
    assert context.num_reasoning_tokens == 4
    assert context.num_output_tokens == 4


def test_parsable_context_zero_tokens_edge_case(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test behavior with all zero token counts."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

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
async def test_parsable_context_single_turn_no_tool_output(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test that first turn never generates tool output tokens."""
    context = create_parsable_context(
        mock_tokenizer,
        mock_reasoning_parser_cls,
        mock_request,
        available_tools=["browser"],
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
    assert context.is_first_turn is False


@pytest.mark.asyncio
async def test_parsable_context_negative_tool_tokens_edge_case(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test edge case where calculation could result in negative tool
    tokens. We should log an error and clamp the value to 0."""
    with patch("vllm.entrypoints.context.logger.error") as mock_log:
        context = create_parsable_context(
            mock_tokenizer,
            mock_reasoning_parser_cls,
            mock_request,
            available_tools=["browser"],
        )

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
        # num_prompt_tokens = 10 (turn1) + 12 (turn2) = 22
        assert context.num_prompt_tokens == 10 + 12
        assert context.num_output_tokens == 5 + 2

        # Verify the error was logged properly
        mock_log.assert_called_once()

        # Extract the actual log message and arguments from the call
        args, _ = mock_log.call_args
        log_message = args[0]

        # Check for key parts of the message
        assert "Negative tool output tokens calculated" in log_message


def test_parsable_context_append_tool_output(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test append_tool_output method."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    # Create mock tool output items
    tool_output = [{"type": "function_call_output", "output": "result"}]

    context.append_tool_output(tool_output)

    # Verify tool output was added to parser response_messages
    assert tool_output == mock_streamable_parser.response_messages


def test_parsable_context_need_builtin_tool_call(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test need_builtin_tool_call method for different tool types."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    # Test code_interpreter tool
    mock_msg = MagicMock()
    mock_msg.type = "function_call"
    mock_msg.name = "code_interpreter"
    mock_streamable_parser.response_messages = [mock_msg]

    assert context.need_builtin_tool_call() is True

    # Test web_search_preview tool
    mock_msg.name = "web_search_preview"
    assert context.need_builtin_tool_call() is True

    # Test python tool
    mock_msg.name = "python"
    assert context.need_builtin_tool_call() is True

    # Test container tool
    mock_msg.name = "container.exec"
    assert context.need_builtin_tool_call() is True

    # Test non-builtin tool
    mock_msg.name = "custom_tool"
    assert context.need_builtin_tool_call() is False

    # Test non-function_call type
    mock_msg.type = "message"
    mock_msg.name = "code_interpreter"
    assert context.need_builtin_tool_call() is False


def test_parsable_context_parser_reset_on_finished(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test that parser is reset when output is finished."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    mock_streamable_parser.final_output = [{"type": "message", "content": "test"}]

    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[4, 5],
        finished=True,
    )

    context.append_output(mock_output)

    # Verify parser.reset() was called
    mock_streamable_parser.reset.assert_called_once()


def test_parsable_context_parser_not_reset_when_not_finished(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test that parser is not reset when output is not finished."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    mock_output = create_mock_request_output(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[4, 5],
        finished=False,
    )

    context.append_output(mock_output)

    # Verify parser.reset() was not called
    mock_streamable_parser.reset.assert_not_called()


def test_parsable_context_render_for_completion_not_implemented(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test that render_for_completion raises NotImplementedError."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    with pytest.raises(NotImplementedError):
        context.render_for_completion()


def test_parsable_context_reasoning_parser_required(mock_tokenizer, mock_request):
    """Test that reasoning_parser_cls is required."""
    with pytest.raises(ValueError, match="reasoning_parser_cls must be provided"):
        ParsableContext(
            response_messages=[],
            tokenizer=mock_tokenizer,
            reasoning_parser_cls=None,
            request=mock_request,
            available_tools=None,
            tool_parser_cls=None,
            chat_template=None,
            chat_template_content_format="auto",
        )


def test_turn_metrics_copy_and_reset():
    """Test TurnMetrics copy and reset methods work correctly."""
    original_metrics = TurnMetrics(
        input_tokens=10,
        output_tokens=20,
        cached_input_tokens=5,
        tool_output_tokens=3,
    )

    # Test copy functionality
    copied_metrics = original_metrics.copy()

    # Verify copy has same values
    assert copied_metrics.input_tokens == 10
    assert copied_metrics.output_tokens == 20
    assert copied_metrics.cached_input_tokens == 5
    assert copied_metrics.tool_output_tokens == 3

    # Verify they are separate objects
    assert copied_metrics is not original_metrics

    # Modify copy to ensure independence
    copied_metrics.input_tokens = 999
    assert original_metrics.input_tokens == 10
    assert copied_metrics.input_tokens == 999

    # Test reset functionality
    original_metrics.reset()

    # Verify all fields are reset to zero
    assert original_metrics.input_tokens == 0
    assert original_metrics.output_tokens == 0
    assert original_metrics.cached_input_tokens == 0
    assert original_metrics.tool_output_tokens == 0

    # Verify copied metrics are unaffected by reset
    assert copied_metrics.input_tokens == 999
    assert copied_metrics.output_tokens == 20
    assert copied_metrics.cached_input_tokens == 5
    assert copied_metrics.tool_output_tokens == 3


@pytest.mark.asyncio
async def test_parsable_context_call_tool_empty_messages(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test call_tool returns empty list when no messages."""
    context = create_parsable_context(
        mock_tokenizer, mock_reasoning_parser_cls, mock_request
    )

    mock_streamable_parser.response_messages = []

    result = await context.call_tool()
    assert result == []


@pytest.mark.asyncio
async def test_parsable_context_init_tool_sessions(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test init_tool_sessions with tool server."""
    context = create_parsable_context(
        mock_tokenizer,
        mock_reasoning_parser_cls,
        mock_request,
        available_tools=["browser"],
    )

    # Create mock tool server and exit stack
    mock_tool_server = MagicMock()
    mock_exit_stack = MagicMock()
    mock_session = MagicMock()
    mock_exit_stack.enter_async_context = MagicMock(return_value=mock_session)

    # Make enter_async_context return a coroutine
    async def mock_enter_context(*args, **kwargs):
        return mock_session

    mock_exit_stack.enter_async_context = mock_enter_context

    await context.init_tool_sessions(
        tool_server=mock_tool_server,
        exit_stack=mock_exit_stack,
        request_id="test-request-id",
        mcp_tools={},
    )

    # Verify session was added
    assert "browser" in context._tool_sessions


@pytest.mark.asyncio
async def test_parsable_context_cleanup_session(
    mock_tokenizer, mock_reasoning_parser_cls, mock_request, mock_streamable_parser
):
    """Test cleanup_session method."""
    context = create_parsable_context(
        mock_tokenizer,
        mock_reasoning_parser_cls,
        mock_request,
        available_tools=["browser"],
    )

    # Add mock session
    mock_session = MagicMock()

    async def mock_call_tool(*args, **kwargs):
        return None

    mock_session.call_tool = mock_call_tool
    context._tool_sessions["browser"] = mock_session
    context.called_tools.add("browser")

    # Should not raise
    await context.cleanup_session()
