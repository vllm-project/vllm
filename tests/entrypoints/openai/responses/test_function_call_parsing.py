# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test function call parsing in ResponsesRequest."""

import json

import pytest
from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def test_function_call_dict_converted_to_object():
    """Test that function_call dictionaries are correctly parsed into
    ResponseFunctionToolCall objects."""
    # Create a request with function_call as dict
    request_data = {
        "model": "gpt-oss",
        "input": [
            {
                "type": "function_call",
                "call_id": "fc_123",
                "name": "get_weather",
                "arguments": '{"location": "Boston", "unit": "celsius"}',
            }
        ],
    }

    request = ResponsesRequest(**request_data)

    # Verify the input item is now a ResponseFunctionToolCall object
    assert len(request.input) == 1
    assert isinstance(request.input[0], ResponseFunctionToolCall)
    assert request.input[0].call_id == "fc_123"
    assert request.input[0].name == "get_weather"
    assert request.input[0].arguments == '{"location": "Boston", "unit": "celsius"}'


def test_direct_function_call_object_preservation():
    """Test that ResponseFunctionToolCall objects passed directly are preserved."""
    # Create a request with ResponseFunctionToolCall object
    function_call = ResponseFunctionToolCall(
        type="function_call",
        call_id="fc_456",
        name="get_stock_price",
        arguments='{"symbol": "AAPL"}',
    )

    request_data = {"model": "gpt-oss", "input": [function_call]}

    request = ResponsesRequest(**request_data)

    # Verify the object is preserved
    assert len(request.input) == 1
    assert request.input[0] is function_call


def test_mixed_input_types_with_function_calls():
    """Test parsing with mixed input types including function calls."""

    request_data = {
        "model": "gpt-oss",
        "input": [
            # Valid Message type
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What's the weather?"}],
            },
            # Function call that should be parsed
            {
                "type": "function_call",
                "call_id": "fc_789",
                "name": "check_weather",
                "arguments": '{"location": "NYC"}',
            },
            # Another function call
            {
                "type": "function_call",
                "call_id": "fc_790",
                "name": "get_time",
                "arguments": "{}",
            },
        ],
    }

    request = ResponsesRequest(**request_data)

    # Verify mixed types are handled correctly
    assert len(request.input) == 3
    # First item should be validated as Message
    assert request.input[0]["type"] == "message"
    # Second item should be parsed to ResponseFunctionToolCall
    assert isinstance(request.input[1], ResponseFunctionToolCall)
    assert request.input[1].call_id == "fc_789"
    assert request.input[1].name == "check_weather"
    # Third item should also be parsed to ResponseFunctionToolCall
    assert isinstance(request.input[2], ResponseFunctionToolCall)
    assert request.input[2].call_id == "fc_790"
    assert request.input[2].name == "get_time"


def test_function_call_with_complex_arguments():
    """Test parsing function calls with complex nested arguments."""
    complex_args = {
        "query": "weather forecast",
        "filters": {
            "location": {"city": "San Francisco", "state": "CA"},
            "timeRange": {"start": "2024-01-01", "end": "2024-01-07"},
            "metrics": ["temperature", "humidity", "precipitation"],
        },
        "options": {"format": "detailed", "includeAlerts": True},
    }

    request_data = {
        "model": "gpt-oss",
        "input": [
            {
                "type": "function_call",
                "call_id": "fc_complex",
                "name": "advanced_weather_query",
                "arguments": json.dumps(complex_args),
            }
        ],
    }

    request = ResponsesRequest(**request_data)

    # Verify complex arguments are preserved correctly
    assert len(request.input) == 1
    assert isinstance(request.input[0], ResponseFunctionToolCall)
    assert request.input[0].call_id == "fc_complex"
    assert request.input[0].name == "advanced_weather_query"

    # Parse the arguments back to verify they're intact
    parsed_args = json.loads(request.input[0].arguments)
    assert parsed_args == complex_args


def test_invalid_function_call_fallback():
    """Test that invalid function call dictionaries fall back gracefully."""
    # Missing required field 'call_id'
    request_data = {
        "model": "gpt-oss",
        "input": [
            {"type": "function_call", "name": "incomplete_function", "arguments": "{}"}
        ],
    }

    # This should not raise an error during model creation
    # The validator should keep the original dict and let Pydantic
    # handle validation
    with pytest.raises(ValueError):
        # Pydantic should raise a validation error for the invalid structure
        ResponsesRequest(**request_data)


def test_string_input_not_affected():
    """Test that string input is not affected by the validator."""
    request_data = {"model": "gpt-oss", "input": "This is a simple string input"}

    request = ResponsesRequest(**request_data)

    # Verify string input remains unchanged
    assert request.input == "This is a simple string input"


def test_empty_list_input():
    """Test that empty list input is handled correctly."""
    request_data = {"model": "gpt-oss", "input": []}

    request = ResponsesRequest(**request_data)

    # Verify empty list is preserved
    assert request.input == []


def test_function_call_output_not_affected():
    """Test that FunctionCallOutput is not affected by the function_call parsing."""

    # Test with FunctionCallOutput as dict (should not be parsed)
    request_data = {
        "model": "gpt-oss",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "fc_output_123",
                "output": "The weather in Boston is 72°F and sunny.",
            }
        ],
    }

    request = ResponsesRequest(**request_data)

    # FunctionCallOutput should remain as dict (not converted to an object)
    assert len(request.input) == 1
    assert isinstance(request.input[0], dict)
    assert request.input[0]["type"] == "function_call_output"
    assert request.input[0]["call_id"] == "fc_output_123"
    assert request.input[0]["output"] == "The weather in Boston is 72°F and sunny."


def test_mixed_function_call_and_output():
    """Test that function_call is parsed while function_call_output is preserved."""
    request_data = {
        "model": "gpt-oss",
        "input": [
            # This should be parsed to ResponseFunctionToolCall
            {
                "type": "function_call",
                "call_id": "fc_call_456",
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            },
            # This should remain as dict
            {
                "type": "function_call_output",
                "call_id": "fc_call_456",
                "output": "NYC weather is 68°F with light rain",
            },
        ],
    }

    request = ResponsesRequest(**request_data)

    assert len(request.input) == 2

    # First item should be parsed to ResponseFunctionToolCall
    assert isinstance(request.input[0], ResponseFunctionToolCall)
    assert request.input[0].call_id == "fc_call_456"
    assert request.input[0].name == "get_weather"

    # Second item should remain as dict (FunctionCallOutput)
    assert isinstance(request.input[1], dict)
    assert request.input[1]["type"] == "function_call_output"
    assert request.input[1]["call_id"] == "fc_call_456"
    assert request.input[1]["output"] == "NYC weather is 68°F with light rain"


def test_function_call_validation_failure_logs_debug(caplog):
    """Test that validation failures are logged at debug level."""
    from unittest.mock import patch

    request_data = {
        "model": "gpt-oss",
        "input": [
            {
                "type": "function_call",
                "name": "incomplete_function",
                "arguments": "{}",  # Missing call_id
            }
        ],
    }

    # Mock the logger to verify debug was called
    with patch("vllm.entrypoints.openai.responses.protocol.logger") as mock_logger:
        with pytest.raises(ValueError):
            ResponsesRequest(**request_data)

        # Verify debug was called with expected message
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Failed to parse function_call" in call_args


def test_validator_handles_iterator_input():
    """Test that validator can handle ValidatorIterator input (Pydantic internal)."""

    # This test simulates when Pydantic passes a ValidatorIterator instead of a list
    # This happened with complex nested structures containing reasoning + function_call

    # Create test data that would normally be a list
    test_input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Test"}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Test reasoning"}],
            "content": [{"type": "reasoning_text", "text": "Test content"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "test_function",
            "arguments": '{"test": "value"}',
            "id": "fc_1",
        },
    ]

    # Mock data where input is an iterator (simulates Pydantic ValidatorIterator)
    mock_data = {
        "model": "test-model",
        "input": iter(test_input_items),  # Iterator instead of list
    }

    # This should NOT raise an error with the fixed validator
    try:
        request = ResponsesRequest(**mock_data)

        # Verify the validator processed the data correctly
        assert len(request.input) == 3

        # Verify function_call was converted to ResponseFunctionToolCall object
        function_call_item = None
        for item in request.input:
            if isinstance(item, ResponseFunctionToolCall):
                function_call_item = item
                break

        assert function_call_item is not None
        assert function_call_item.call_id == "call_1"
        assert function_call_item.name == "test_function"

    except Exception as e:
        pytest.fail(f"Validator should handle iterator input, but failed with: {e}")


def test_validator_handles_empty_iterator():
    """Test validator handles empty iterator gracefully."""
    mock_data = {
        "model": "test-model",
        "input": iter([]),  # Empty iterator
    }

    request = ResponsesRequest(**mock_data)
    assert request.input == []
