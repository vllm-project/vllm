# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.entrypoints.serve.utils.request_logger import RequestLogger


def test_request_logger_log_outputs():
    """Test the new log_outputs functionality."""
    # Create a mock logger to capture log calls
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test basic output logging
        request_logger.log_outputs(
            request_id="test-123",
            outputs="Hello, world!",
            output_token_ids=[1, 2, 3, 4],
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-123"
        assert call_args[3] == "Hello, world!"
        assert call_args[4] == [1, 2, 3, 4]
        assert call_args[5] == "stop"


def test_request_logger_log_outputs_streaming_delta():
    """Test log_outputs with streaming delta mode."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test streaming delta logging
        request_logger.log_outputs(
            request_id="test-456",
            outputs="Hello",
            output_token_ids=[1],
            finish_reason=None,
            is_streaming=True,
            delta=True,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-456"
        assert call_args[2] == " (streaming delta)"
        assert call_args[3] == "Hello"
        assert call_args[4] == [1]
        assert call_args[5] is None


def test_request_logger_log_outputs_streaming_complete():
    """Test log_outputs with streaming complete mode."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test streaming complete logging
        request_logger.log_outputs(
            request_id="test-789",
            outputs="Complete response",
            output_token_ids=[1, 2, 3],
            finish_reason="length",
            is_streaming=True,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-789"
        assert call_args[2] == " (streaming complete)"
        assert call_args[3] == "Complete response"
        assert call_args[4] == [1, 2, 3]
        assert call_args[5] == "length"


def test_request_logger_log_outputs_with_truncation():
    """Test log_outputs respects max_log_len setting."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        # Set max_log_len to 10
        request_logger = RequestLogger(max_log_len=10)

        # Test output truncation
        long_output = "This is a very long output that should be truncated"
        long_token_ids = list(range(20))  # 20 tokens

        request_logger.log_outputs(
            request_id="test-truncate",
            outputs=long_output,
            output_token_ids=long_token_ids,
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args

        # Check that output was truncated to first 10 characters
        logged_output = call_args[0][3]
        assert logged_output == "This is a "
        assert len(logged_output) == 10

        # Check that token IDs were truncated to first 10 tokens
        logged_token_ids = call_args[0][4]
        assert logged_token_ids == list(range(10))
        assert len(logged_token_ids) == 10


def test_request_logger_log_outputs_none_values():
    """Test log_outputs handles None values correctly."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test with None output_token_ids
        request_logger.log_outputs(
            request_id="test-none",
            outputs="Test output",
            output_token_ids=None,
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-none"
        assert call_args[3] == "Test output"
        assert call_args[4] is None
        assert call_args[5] == "stop"


def test_request_logger_log_outputs_empty_output():
    """Test log_outputs handles empty output correctly."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=5)

        # Test with empty output
        request_logger.log_outputs(
            request_id="test-empty",
            outputs="",
            output_token_ids=[],
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-empty"
        assert call_args[3] == ""
        assert call_args[4] == []
        assert call_args[5] == "stop"


def test_request_logger_log_outputs_integration():
    """Test that log_outputs can be called alongside log_inputs."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test that both methods can be called without interference
        request_logger.log_inputs(
            request_id="test-integration",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            params=None,
            lora_request=None,
        )

        request_logger.log_outputs(
            request_id="test-integration",
            outputs="Test output",
            output_token_ids=[4, 5, 6],
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        # Should have been called twice - once for inputs, once for outputs
        assert mock_logger.info.call_count == 2

        # Check that the calls were made with correct patterns
        input_call = mock_logger.info.call_args_list[0][0]
        output_call = mock_logger.info.call_args_list[1][0]

        assert "Received request %s" in input_call[0]
        assert input_call[1] == "test-integration"

        assert "Generated response %s%s" in output_call[0]
        assert output_call[1] == "test-integration"


def test_streaming_complete_logs_full_text_content():
    """Test that streaming complete logging includes
    full accumulated text, not just token count."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.serve.utils.request_logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test with actual content instead of token count format
        full_response = "This is a complete response from streaming"
        request_logger.log_outputs(
            request_id="test-streaming-full-text",
            outputs=full_response,
            output_token_ids=None,
            finish_reason="streaming_complete",
            is_streaming=True,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args

        # Verify the logged output is the full text, not a token count format
        logged_output = call_args[3]
        assert logged_output == full_response
        assert "tokens>" not in logged_output
        assert "streaming_complete" not in logged_output

        # Verify other parameters
        assert call_args[1] == "test-streaming-full-text"
        assert call_args[2] == " (streaming complete)"
        assert call_args[5] == "streaming_complete"
