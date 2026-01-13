# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ResponsesRequest function call parsing.

This test module ensures that function call dictionaries are properly parsed
as ResponseFunctionToolCall objects when creating ResponsesRequest instances,
preventing regression of the function call parsing issue.
"""

import pytest
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall)

from vllm.entrypoints.openai.protocol import ResponsesRequest


def test_function_call_dict_parsing():
    """Test that function call dictionaries are parsed as 
    ResponseFunctionToolCall objects."""
    # Function call data as it would come from JSON/API
    function_call_dict = {
        'arguments': '{"location": "Tokyo"}',
        'call_id': 'call_a90e65e81cd84553855dde1733e77749',
        'name': 'get_weather',
        'type': 'function_call',
        'id': 'ft_a90e65e81cd84553855dde1733e77749',
        'status': 'completed'
    }

    # Create ResponsesRequest with function call dictionary
    request_data = {'input': [function_call_dict], 'model': 'test-model'}

    request = ResponsesRequest(**request_data)
    parsed_function_call = request.input[0]

    # Verify that the dictionary was parsed as ResponseFunctionToolCall
    assert isinstance(parsed_function_call, ResponseFunctionToolCall), (
        f"Expected ResponseFunctionToolCall, got {type(parsed_function_call)}")

    # Verify all attributes are preserved
    assert parsed_function_call.arguments == '{"location": "Tokyo"}'
    assert (parsed_function_call.call_id ==
            'call_a90e65e81cd84553855dde1733e77749')
    assert parsed_function_call.name == 'get_weather'
    assert parsed_function_call.type == 'function_call'
    assert parsed_function_call.id == 'ft_a90e65e81cd84553855dde1733e77749'
    assert parsed_function_call.status == 'completed'


def test_direct_function_call_object_preservation():
    """Test that direct ResponseFunctionToolCall objects are preserved."""
    # Create ResponseFunctionToolCall object directly
    function_call_obj = ResponseFunctionToolCall(
        arguments='{"city": "New York"}',
        call_id='call_12345',
        name='get_temperature',
        type='function_call',
        id='ft_12345',
        status='completed')

    # Create ResponsesRequest with the object
    request_data = {'input': [function_call_obj], 'model': 'test-model'}

    request = ResponsesRequest(**request_data)
    parsed_function_call = request.input[0]

    # Verify that the object is preserved
    assert isinstance(parsed_function_call, ResponseFunctionToolCall)
    # Should be the same object
    assert parsed_function_call is function_call_obj


def test_mixed_input_types_with_function_calls():
    """Test ResponsesRequest with mixed input types including function calls."""
    # Mix of string input and function call
    function_call_dict = {
        'arguments': '{"query": "weather"}',
        'call_id': 'call_mixed_test',
        'name': 'search',
        'type': 'function_call',
        'id': 'ft_mixed_test',
        'status': 'completed'
    }

    request_data = {
        'input': [
            {
                'role': 'user',
                'content': 'Hello, how can I help?'
            },  # Regular dictionary input
            function_call_dict  # Function call dict
        ],
        'model':
        'test-model'
    }

    request = ResponsesRequest(**request_data)

    # Verify mixed types are handled correctly
    assert len(request.input) == 2
    assert isinstance(request.input[0], dict)
    assert request.input[0]['role'] == 'user'
    assert request.input[0]['content'] == 'Hello, how can I help?'

    assert isinstance(request.input[1], ResponseFunctionToolCall)
    assert request.input[1].name == 'search'
    assert request.input[1].arguments == '{"query": "weather"}'


@pytest.mark.parametrize("arguments", [
    '{"location": "Tokyo"}',
    '{"location": {"city": "Tokyo", "country": "Japan"}, "units": "celsius"}',
    '{"query": "weather", "filters": ["temperature", "humidity"], "count": 5}',
    '{"complex": {"nested": {"data": true}}, "array": [1, 2, 3]}'
])
def test_function_call_with_complex_arguments(arguments):
    """Test function call parsing with various argument complexities."""
    complex_function_call = {
        'arguments': arguments,
        'call_id': 'call_complex',
        'name': 'get_detailed_weather',
        'type': 'function_call',
        'id': 'ft_complex',
        'status': 'completed'
    }

    request_data = {'input': [complex_function_call], 'model': 'test-model'}

    request = ResponsesRequest(**request_data)
    parsed_function_call = request.input[0]

    assert isinstance(parsed_function_call, ResponseFunctionToolCall)
    assert parsed_function_call.name == 'get_detailed_weather'
    assert parsed_function_call.arguments == arguments
