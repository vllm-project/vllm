# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
)

ALL_JSON_TYPES = {
    "string",
    "number",
    "integer",
    "boolean",
    "null",
    "object",
    "array",
}


class TestCoerceToSchemaType:
    class TestNullHandling:
        def test_null_converted_when_type_is_null(self):
            assert coerce_to_schema_type("null", "null") is None

        def test_null_converted_when_null_in_type_list(self):
            assert coerce_to_schema_type("null", ["string", "null"]) is None

        def test_null_preserved_as_string_when_type_is_string(self):
            assert coerce_to_schema_type("null", "string") == "null"

        def test_null_case_insensitive(self):
            assert coerce_to_schema_type("NULL", "null") is None
            assert coerce_to_schema_type("Null", "null") is None

        def test_none_string_never_converted(self):
            assert coerce_to_schema_type("none", "null") == "none"
            assert coerce_to_schema_type("none", "string") == "none"
            assert coerce_to_schema_type("none", ["string", "null"]) == "none"

        def test_nil_string_never_converted(self):
            assert coerce_to_schema_type("nil", "string") == "nil"
            assert coerce_to_schema_type("nil", ["string", "null"]) == "nil"

        def test_non_null_value_with_null_type(self):
            assert coerce_to_schema_type("hello", ["null", "string"]) == "hello"

    class TestStringType:
        def test_string_type(self):
            assert coerce_to_schema_type("hello", "string") == "hello"

        def test_str_alias(self):
            assert coerce_to_schema_type("hello", "str") == "hello"

        def test_text_alias(self):
            assert coerce_to_schema_type("hello", "text") == "hello"

        def test_varchar_alias(self):
            assert coerce_to_schema_type("hello", "varchar") == "hello"

        def test_char_alias(self):
            assert coerce_to_schema_type("x", "char") == "x"

        def test_enum_alias(self):
            assert coerce_to_schema_type("option_a", "enum") == "option_a"

    class TestIntegerType:
        def test_integer_type(self):
            assert coerce_to_schema_type("42", "integer") == 42

        def test_int_alias(self):
            assert coerce_to_schema_type("42", "int") == 42

        def test_negative_integer(self):
            assert coerce_to_schema_type("-7", "integer") == -7

        def test_invalid_integer_fallback(self):
            assert coerce_to_schema_type("not_a_number", "integer") == "not_a_number"

        def test_uint32_alias(self):
            assert coerce_to_schema_type("5", "uint32") == 5

        def test_long_alias(self):
            assert coerce_to_schema_type("100", "long") == 100

    class TestNumberType:
        def test_number_type(self):
            assert coerce_to_schema_type("3.14", "number") == 3.14

        def test_float_alias(self):
            assert coerce_to_schema_type("2.5", "float") == 2.5

        def test_double_alias(self):
            assert coerce_to_schema_type("2.5", "double") == 2.5

        def test_whole_float_returns_int(self):
            assert coerce_to_schema_type("5.0", "number") == 5
            assert isinstance(coerce_to_schema_type("5.0", "number"), int)

        def test_invalid_number_fallback(self):
            assert coerce_to_schema_type("abc", "number") == "abc"

    class TestNonFiniteNumbers:
        """Non-finite numeric strings must not crash and must coerce to a
        JSON-serializable value.

        Regression: ``int(float("inf"))`` raised an uncaught ``OverflowError``
        (only ``ValueError``/``TypeError`` were handled), and ``"1e999"``
        round-tripped through ``json.loads`` to a float ``inf`` that
        ``json.dumps`` renders as invalid JSON ``Infinity``.
        """

        @pytest.mark.parametrize(
            "value", ["inf", "-inf", "Infinity", "1e999", "nan", "-nan"]
        )
        def test_non_finite_number_does_not_crash(self, value):
            # Must not raise (previously OverflowError for inf/1e999/Infinity).
            result = coerce_to_schema_type(value, "number")
            # Result must serialize to valid, finite JSON and round-trip.
            assert json.loads(json.dumps(result)) == result

        @pytest.mark.parametrize("value", ["inf", "-inf", "1e999"])
        def test_non_finite_number_preserved_as_string(self, value):
            assert coerce_to_schema_type(value, "number") == value

        @pytest.mark.parametrize("value", ["inf", "1e999", "Infinity"])
        def test_non_finite_integer_not_float_inf(self, value):
            result = coerce_to_schema_type(value, "integer")
            assert isinstance(result, str)
            assert result == value

    class TestNonFiniteContainers:
        """Non-finite floats nested in object/array values must not produce
        invalid JSON.

        Regression: the ``object``/``array`` branch returned
        ``json.loads(value)`` directly, so ``"[1e999]"`` became ``[inf]`` and
        ``'{"x": Infinity}'`` became ``{"x": inf}`` -- values that
        ``json.dumps`` later renders as invalid JSON (``Infinity``/``NaN``).
        """

        @pytest.mark.parametrize(
            "value", ["[1e999]", "[1, 2, 1e999]", "[NaN]", "[-Infinity]"]
        )
        def test_array_with_non_finite_preserved_as_string(self, value):
            result = coerce_to_schema_type(value, "array")
            assert result == value
            assert json.loads(json.dumps(result)) == result

        @pytest.mark.parametrize(
            "value", ['{"x": 1e999}', '{"x": Infinity}', '{"a": [1e999, 2]}']
        )
        def test_object_with_non_finite_preserved_as_string(self, value):
            result = coerce_to_schema_type(value, "object")
            assert result == value
            assert json.loads(json.dumps(result)) == result

        def test_finite_array_still_coerced(self):
            assert coerce_to_schema_type("[1, 2, 3]", "array") == [1, 2, 3]

        def test_finite_object_still_coerced(self):
            assert coerce_to_schema_type('{"a": 1}', "object") == {"a": 1}

        def test_unknown_type_non_finite_falls_back_to_string(self):
            # Exercises the final json.loads fallback path.
            assert coerce_to_schema_type("1e999", "unknown_type") == "1e999"

    class TestBooleanType:
        def test_true(self):
            assert coerce_to_schema_type("true", "boolean") is True

        def test_false(self):
            assert coerce_to_schema_type("false", "boolean") is False

        def test_bool_alias(self):
            assert coerce_to_schema_type("true", "bool") is True

        def test_one_is_true(self):
            assert coerce_to_schema_type("1", "boolean") is True

        def test_zero_is_false(self):
            assert coerce_to_schema_type("0", "boolean") is False

        def test_invalid_boolean_fallback(self):
            assert coerce_to_schema_type("maybe", "boolean") == "maybe"

    class TestObjectArrayType:
        def test_object_type(self):
            assert coerce_to_schema_type('{"a": 1}', "object") == {"a": 1}

        def test_array_type(self):
            assert coerce_to_schema_type("[1, 2, 3]", "array") == [1, 2, 3]

        def test_invalid_json_fallback(self):
            assert coerce_to_schema_type("not json", "object") == "not json"

        def test_dict_alias(self):
            assert coerce_to_schema_type('{"k": "v"}', "dict") == {"k": "v"}

        def test_list_alias(self):
            assert coerce_to_schema_type("[1]", "list") == [1]

    class TestMultiType:
        def test_null_takes_priority_over_string(self):
            assert coerce_to_schema_type("null", ["string", "null"]) is None

        def test_integer_tried_before_string(self):
            assert coerce_to_schema_type("42", ["integer", "string"]) == 42

        def test_falls_through_to_string(self):
            assert coerce_to_schema_type("hello", ["integer", "string"]) == "hello"

    class TestFallback:
        def test_unknown_type_returns_string(self):
            assert coerce_to_schema_type("hello", "unknown_type") == "hello"

        def test_json_fallback_for_unknown_type(self):
            assert coerce_to_schema_type('{"a": 1}', "unknown_type") == {"a": 1}

        @pytest.mark.parametrize("schema_type", ["string", "str", "text"])
        def test_string_types_preserve_value(self, schema_type):
            assert coerce_to_schema_type("anything", schema_type) == "anything"

        def test_unrecognized_type_falls_back_to_json(self):
            assert coerce_to_schema_type("42", "interval") == 42


class TestExtractTypesFromSchema:
    def test_direct_type_string(self):
        assert extract_types_from_schema({"type": "string"}) == ["string"]

    def test_direct_type_integer(self):
        assert extract_types_from_schema({"type": "integer"}) == ["integer"]

    def test_type_array(self):
        result = set(extract_types_from_schema({"type": ["string", "null"]}))
        assert result == {"string", "null"}

    def test_anyof(self):
        schema = {"anyOf": [{"type": "object"}, {"type": "null"}]}
        result = set(extract_types_from_schema(schema))
        assert result == {"object", "null"}

    def test_oneof(self):
        schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
        result = set(extract_types_from_schema(schema))
        assert result == {"integer", "string"}

    def test_allof(self):
        schema = {"allOf": [{"type": "object"}]}
        assert extract_types_from_schema(schema) == ["object"]

    def test_enum_infers_types(self):
        schema = {"enum": [1, "a", None]}
        result = set(extract_types_from_schema(schema))
        assert result == {"integer", "string", "null"}

    def test_enum_with_bool(self):
        schema = {"enum": [True, False]}
        assert extract_types_from_schema(schema) == ["boolean"]

    def test_enum_with_float(self):
        schema = {"enum": [1.5, 2.5]}
        assert extract_types_from_schema(schema) == ["number"]

    def test_enum_with_list_and_dict(self):
        schema = {"enum": [[1, 2], {"a": 1}]}
        result = set(extract_types_from_schema(schema))
        assert result == {"array", "object"}

    def test_none_schema_defaults_to_string(self):
        assert extract_types_from_schema(None) == ["string"]

    def test_non_dict_schema_defaults_to_string(self):
        assert extract_types_from_schema("string") == ["string"]

    def test_empty_dict_allows_any_type(self):
        assert set(extract_types_from_schema({})) == ALL_JSON_TYPES

    def test_untyped_property_schema_allows_any_type(self):
        schema = {"description": "Enter the requested value here."}
        assert set(extract_types_from_schema(schema)) == ALL_JSON_TYPES

    def test_nested_anyof(self):
        schema = {
            "anyOf": [
                {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                {"type": "string"},
            ]
        }
        result = set(extract_types_from_schema(schema))
        assert result == {"integer", "null", "string"}


class TestUntypedSchemaCoercion:
    """A property schema without a type constraint permits any JSON type,
    so values should be coerced to their natural JSON type instead of
    defaulting to string (https://github.com/vllm-project/vllm/issues/47557).
    """

    def coerce_untyped(self, value: str):
        return coerce_to_schema_type(value, extract_types_from_schema({}))

    def test_integer_value(self):
        assert self.coerce_untyped("5") == 5

    def test_float_value(self):
        assert self.coerce_untyped("3.14") == 3.14

    def test_boolean_value(self):
        assert self.coerce_untyped("true") is True

    def test_null_value(self):
        assert self.coerce_untyped("null") is None

    def test_object_value(self):
        assert self.coerce_untyped('{"a": 1}') == {"a": 1}

    def test_array_value(self):
        assert self.coerce_untyped("[1, 2]") == [1, 2]

    def test_plain_text_stays_string(self):
        assert self.coerce_untyped("hello") == "hello"

    def test_numeric_looking_quoted_json_string(self):
        assert self.coerce_untyped('"5"') == "5"
