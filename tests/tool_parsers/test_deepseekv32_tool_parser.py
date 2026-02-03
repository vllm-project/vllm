# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import pytest

from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.deepseekv32_tool_parser import (
    DeepSeekV32ToolParser,
)

MODEL = "deepseek-ai/DeepSeek-V3.2"


@pytest.fixture(scope="module")
def deepseekv32_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def parser(deepseekv32_tokenizer):
    return DeepSeekV32ToolParser(deepseekv32_tokenizer)


def test_convert_param_value_single_types(parser):
    """Test _convert_param_value with single type parameters."""
    # Test string type
    assert parser._convert_param_value("hello", "string") == "hello"
    assert parser._convert_param_value("123", "string") == "123"

    # Test integer type - valid integers
    assert parser._convert_param_value("123", "integer") == 123
    assert parser._convert_param_value("456", "int") == 456
    # Invalid integer should return original string (due to exception catch)
    assert parser._convert_param_value("abc", "integer") == "abc"

    # Test float/number type
    assert parser._convert_param_value("123.45", "float") == 123.45
    assert (
        parser._convert_param_value("123.0", "number") == 123
    )  # Should be int when whole number
    assert parser._convert_param_value("123.5", "number") == 123.5
    # Invalid float should return original string
    assert parser._convert_param_value("abc", "float") == "abc"

    # Test boolean type - valid boolean values
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value("false", "bool") is False
    assert parser._convert_param_value("1", "boolean") is True
    assert parser._convert_param_value("0", "boolean") is False
    # Invalid boolean should return original string
    assert parser._convert_param_value("yes", "boolean") == "yes"
    assert parser._convert_param_value("no", "bool") == "no"

    # Test null value
    assert parser._convert_param_value("null", "string") is None
    assert parser._convert_param_value("null", "integer") is None

    # Test object/array type (JSON)
    assert parser._convert_param_value('{"key": "value"}', "object") == {"key": "value"}
    assert parser._convert_param_value("[1, 2, 3]", "array") == [1, 2, 3]
    # Invalid JSON should return original string
    assert parser._convert_param_value("{invalid}", "object") == "{invalid}"

    # Test fallback for unknown type (tries json.loads, then returns original)
    assert parser._convert_param_value('{"key": "value"}', "unknown") == {
        "key": "value"
    }
    assert parser._convert_param_value("plain text", "unknown") == "plain text"


def test_convert_param_value_multi_typed_values(parser):
    """Test _convert_param_value with multi-typed values (list of types)."""
    # Test with list of types where first type succeeds
    assert parser._convert_param_value("123", ["integer", "string"]) == 123
    assert parser._convert_param_value("true", ["boolean", "string"]) is True
    assert parser._convert_param_value('{"x": 1}', ["object", "string"]) == {"x": 1}

    # Test with list of types where first type fails but second succeeds
    # "abc" is not a valid integer, so should try string next
    assert parser._convert_param_value("abc", ["integer", "string"]) == "abc"

    # Test with list of types where all fail - should return original value
    # "invalid json" is not valid JSON, last type is "object" which will fail JSON parse
    result = parser._convert_param_value("invalid json", ["integer", "object"])
    assert result == "invalid json"  # Returns original value after all types fail

    # Test with three types
    assert parser._convert_param_value("123.5", ["integer", "float", "string"]) == 123.5
    assert parser._convert_param_value("true", ["integer", "boolean", "string"]) is True

    # Test with null in multi-type list
    assert parser._convert_param_value("null", ["integer", "string"]) is None
    assert parser._convert_param_value("null", ["boolean", "object"]) is None

    # Test nested type conversion - boolean fails, integer succeeds
    value = parser._convert_param_value("123", ["boolean", "integer", "string"])
    assert value == 123  # Should be integer, not boolean

    # Test that order matters
    assert (
        parser._convert_param_value("123", ["string", "integer"]) == "123"
    )  # String first
    assert (
        parser._convert_param_value("123", ["integer", "string"]) == 123
    )  # Integer first

    # Test with all types failing - returns original value
    assert (
        parser._convert_param_value("not_a_number", ["integer", "float", "boolean"])
        == "not_a_number"
    )


def test_convert_param_value_stricter_type_checking(parser):
    """Test stricter type checking in the updated implementation."""
    # Boolean now has stricter validation
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value("false", "boolean") is False
    assert parser._convert_param_value("1", "boolean") is True
    assert parser._convert_param_value("0", "boolean") is False

    # These should return original string (not valid boolean values)
    assert parser._convert_param_value("yes", "boolean") == "yes"
    assert parser._convert_param_value("no", "boolean") == "no"
    assert (
        parser._convert_param_value("TRUE", "boolean") == "TRUE"
    )  # Note: uppercase not in allowed list
    assert parser._convert_param_value("FALSE", "boolean") == "FALSE"

    # Integer and float now raise exceptions for invalid values
    assert parser._convert_param_value("123abc", "integer") == "123abc"
    assert parser._convert_param_value("123.45.67", "float") == "123.45.67"

    # JSON parsing is stricter - invalid JSON returns original
    assert parser._convert_param_value("{invalid: json}", "object") == "{invalid: json}"
    assert parser._convert_param_value("[1, 2,", "array") == "[1, 2,"

    # Test multi-type with stricter checking
    # "yes" is not valid boolean, but string would accept it
    assert parser._convert_param_value("yes", ["boolean", "string"]) == "yes"

    # "123abc" is not valid integer or float, but string accepts it
    assert (
        parser._convert_param_value("123abc", ["integer", "float", "string"])
        == "123abc"
    )


def test_convert_param_value_edge_cases(parser):
    """Test edge cases for _convert_param_value."""
    # Empty string
    assert parser._convert_param_value("", "string") == ""
    assert (
        parser._convert_param_value("", "integer") == ""
    )  # Invalid int returns original

    # Whitespace - trimmed by conversion functions
    assert parser._convert_param_value("  123  ", "integer") == 123
    assert parser._convert_param_value("  true  ", "boolean") is True

    # Case sensitivity for boolean (now stricter - only lowercase allowed)
    assert (
        parser._convert_param_value("TRUE", "boolean") == "TRUE"
    )  # Not in allowed list
    assert parser._convert_param_value("FALSE", "boolean") == "FALSE"
    assert parser._convert_param_value("True", "boolean") == "True"
    assert parser._convert_param_value("False", "boolean") == "False"

    # Numeric strings with special characters
    assert parser._convert_param_value("123.45.67", "float") == "123.45.67"
    assert parser._convert_param_value("123abc", "integer") == "123abc"

    # JSON with whitespace - should parse correctly
    assert parser._convert_param_value('  { "key" : "value" }  ', "object") == {
        "key": "value"
    }

    # Invalid JSON returns original
    assert parser._convert_param_value("{invalid}", "object") == "{invalid}"
    assert parser._convert_param_value("[1, 2,", "array") == "[1, 2,"


def test_convert_param_value_checked_helper(parser):
    """Test the _convert_param_value_checked helper function indirectly."""
    # This tests the behavior through the main function
    # Valid conversions should work
    assert parser._convert_param_value("123", "integer") == 123
    assert parser._convert_param_value("123.45", "float") == 123.45
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value('{"x": 1}', "object") == {"x": 1}

    # Invalid conversions should return original value (exception caught)
    assert parser._convert_param_value("abc", "integer") == "abc"
    assert parser._convert_param_value("abc", "float") == "abc"
    assert parser._convert_param_value("yes", "boolean") == "yes"
    assert parser._convert_param_value("{invalid}", "object") == "{invalid}"

    # Test that null handling works in checked function
    assert parser._convert_param_value("null", "integer") is None
    assert parser._convert_param_value("null", "boolean") is None
    assert parser._convert_param_value("null", "object") is None
