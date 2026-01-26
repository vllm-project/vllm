# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tool_parsers/utils.py"""

import pytest
from json import JSONDecodeError

from vllm.tool_parsers.utils import safe_json_loads


class TestSafeJsonLoads:
    """Tests for the safe_json_loads function."""

    def test_single_json_object(self):
        """Test parsing a single valid JSON object."""
        input_str = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        result = safe_json_loads(input_str)
        assert result == {"name": "get_weather", "arguments": {"city": "NYC"}}

    def test_single_json_array(self):
        """Test parsing a single valid JSON array."""
        input_str = '[1, 2, 3]'
        result = safe_json_loads(input_str)
        assert result == [1, 2, 3]

    def test_multiple_concatenated_json_objects(self):
        """Test parsing multiple concatenated JSON objects (extracts first)."""
        input_str = '{"a": 1}{"b": 2}'
        result = safe_json_loads(input_str)
        assert result == {"a": 1}

    def test_multiple_concatenated_json_objects_with_whitespace(self):
        """Test parsing multiple JSON objects separated by whitespace."""
        input_str = '{"a": 1} {"b": 2}'
        result = safe_json_loads(input_str)
        assert result == {"a": 1}

    def test_multiple_tool_calls_scenario(self):
        """Test the actual scenario from issue #32638."""
        # This mimics what happens when multiple tool calls are parsed
        input_str = '{"city": "NYC"}{"unit": "celsius"}'
        result = safe_json_loads(input_str)
        assert result == {"city": "NYC"}

    def test_nested_json_object(self):
        """Test parsing nested JSON structures."""
        input_str = '{"outer": {"inner": {"deep": "value"}}}'
        result = safe_json_loads(input_str)
        assert result == {"outer": {"inner": {"deep": "value"}}}

    def test_json_with_special_characters(self):
        """Test parsing JSON with special characters in strings."""
        input_str = '{"text": "Hello\\nWorld\\t!"}'
        result = safe_json_loads(input_str)
        assert result == {"text": "Hello\nWorld\t!"}

    def test_empty_json_object(self):
        """Test parsing an empty JSON object."""
        input_str = '{}'
        result = safe_json_loads(input_str)
        assert result == {}

    def test_empty_json_array(self):
        """Test parsing an empty JSON array."""
        input_str = '[]'
        result = safe_json_loads(input_str)
        assert result == []

    def test_json_primitive_string(self):
        """Test parsing a JSON primitive string."""
        input_str = '"hello"'
        result = safe_json_loads(input_str)
        assert result == "hello"

    def test_json_primitive_number(self):
        """Test parsing a JSON primitive number."""
        input_str = '42'
        result = safe_json_loads(input_str)
        assert result == 42

    def test_json_primitive_boolean(self):
        """Test parsing JSON boolean values."""
        assert safe_json_loads('true') is True
        assert safe_json_loads('false') is False

    def test_json_null(self):
        """Test parsing JSON null."""
        input_str = 'null'
        result = safe_json_loads(input_str)
        assert result is None

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(JSONDecodeError):
            safe_json_loads('not valid json')

    def test_incomplete_json_raises_error(self):
        """Test that incomplete JSON raises JSONDecodeError."""
        with pytest.raises(JSONDecodeError):
            safe_json_loads('{"incomplete":')

    def test_empty_string_raises_error(self):
        """Test that empty string raises JSONDecodeError."""
        with pytest.raises(JSONDecodeError):
            safe_json_loads('')

    def test_complex_tool_arguments(self):
        """Test parsing complex tool call arguments."""
        input_str = '{"location": "San Francisco, CA", "units": "fahrenheit", "forecast_days": 5}'
        result = safe_json_loads(input_str)
        assert result == {
            "location": "San Francisco, CA",
            "units": "fahrenheit",
            "forecast_days": 5
        }
