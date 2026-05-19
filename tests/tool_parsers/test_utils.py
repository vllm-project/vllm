# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tool_parsers.utils import coerce_to_schema_type


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
