# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json

import pytest

from vllm.tool_parsers.utils import (
    UnexpectedAstError,
    coerce_to_schema_type,
    escape_ctrl_chars_in_strings,
    extract_types_from_schema,
    get_parameter_value,
    handle_single_tool,
    make_valid_python,
)


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

    def test_empty_dict_defaults_to_string(self):
        assert extract_types_from_schema({}) == ["string"]

    def test_nested_anyof(self):
        schema = {
            "anyOf": [
                {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                {"type": "string"},
            ]
        }
        result = set(extract_types_from_schema(schema))
        assert result == {"integer", "null", "string"}


class TestMakeValidPythonStringLiterals:
    def test_bracket_inside_string_is_literal(self):
        # A bracket inside a string argument must not be counted as a
        # structural bracket. Regression: `]` inside the string popped the
        # bracket stack and the whole call raised as mismatched.
        text = "[exec(command='grep -F \"]\" log.txt')]"
        assert make_valid_python(text) == (text, "")

    def test_open_bracket_inside_string_is_literal(self):
        # An unclosed `[` inside a string must not leave a phantom open
        # bracket on the stack.
        text = "[exec(command='grep [abc log.txt')]"
        assert make_valid_python(text) == (text, "")

    def test_partial_string_with_bracket_completes(self):
        # Streaming prefix ending mid-string after a literal bracket closes
        # with quote + paren + bracket.
        result = make_valid_python('[exec(command=\'grep -F "]" lo')
        assert result is not None
        completed, added = result
        assert added == "')]"
        assert completed == "[exec(command='grep -F \"]\" lo')]"

    def test_real_mismatched_bracket_still_raises(self):
        with pytest.raises(UnexpectedAstError):
            make_valid_python("[exec(command=data])")

    def test_multiline_string_argument_recovers(self):
        # A raw newline inside a string argument is invalid Python; the
        # escaped-retry path must recover the call instead of returning None,
        # and the escaped value must evaluate back to the original.
        text = "[exec(command='line1\nline2')]"
        result = make_valid_python(text)
        assert result is not None
        completed, added = result
        assert added == ""
        module = ast.parse(completed)
        call = module.body[0].value.elts[0]
        assert call.keywords[0].value.value == "line1\nline2"

    def test_value_ending_in_backslash_recovers(self):
        # A string value ending in a literal backslash: the closing quote follows
        # an escaped backslash (an *even* run), so it closes the string. Checking
        # only the single preceding char misread it as an escaped quote, left the
        # string open, and make_valid_python returned None — dropping calls whose
        # last argument ends in a backslash (common in regex like r'\b').
        text = "[write(path='x', content='pattern \\\\')]"
        assert make_valid_python(text) == (text, "")

    def test_escaped_quote_odd_backslashes_stays_open(self):
        # An escaped quote (an *odd* backslash run) must NOT close the string;
        # only the final unescaped quote does. Value round-trips to it's fine.
        text = "[say(msg='it\\'s fine')]"
        assert make_valid_python(text) == (text, "")
        module = ast.parse(text)
        assert module.body[0].value.elts[0].keywords[0].value.value == "it's fine"


class TestEscapeCtrlCharsInStrings:
    def test_newline_inside_string_escaped(self):
        assert escape_ctrl_chars_in_strings("f(cmd='a\nb')") == "f(cmd='a\\nb')"

    def test_ctrl_chars_outside_strings_untouched(self):
        assert escape_ctrl_chars_in_strings("f(a=1,\nb=2)") == "f(a=1,\nb=2)"

    def test_existing_escapes_pass_through(self):
        text = "f(cmd='a\\nb')"
        assert escape_ctrl_chars_in_strings(text) == text

    def test_escaped_quote_does_not_close_string(self):
        assert escape_ctrl_chars_in_strings("f(cmd='a\\'\nb')") == "f(cmd='a\\'\\nb')"

    def test_value_preserved_through_ast(self):
        # The escaped text parses and evaluates back to the original value.
        raw = "cat > f.py << EOF\nimport csv\nEOF\techo done"
        escaped = escape_ctrl_chars_in_strings(f"f(cmd='{raw}')")
        call = ast.parse(escaped).body[0].value
        assert call.keywords[0].value.value == raw


def _value_of(expr: str):
    """Parse a single Python expression and run get_parameter_value on it."""
    return get_parameter_value(ast.parse(expr, mode="eval").body)


def _first_call(text: str) -> ast.Call:
    """Parse ``[foo(...)]`` and return the single ast.Call node."""
    return ast.parse(text).body[0].value.elts[0]


class TestGetParameterValueNegativeNumbers:
    # A negative number is parsed by Python as UnaryOp(USub, Constant(n)), not
    # a plain Constant. Without explicit handling the entire tool call is
    # dropped. Negative longitudes/deltas/offsets are extremely common tool
    # arguments (e.g. every Western-hemisphere coordinate).
    def test_negative_int(self):
        assert _value_of("-1") == -1

    def test_negative_float(self):
        assert _value_of("-3.5") == -3.5

    def test_explicit_positive_int(self):
        assert _value_of("+7") == 7

    def test_negative_longitude(self):
        assert _value_of("-74.0046539") == -74.0046539

    def test_negative_in_list(self):
        assert _value_of("[-1, 2, -3]") == [-1, 2, -3]

    def test_negative_in_dict(self):
        assert _value_of('{"min": -5, "max": 5}') == {"min": -5, "max": 5}

    def test_nested_negative(self):
        assert _value_of('{"bbox": [-74.0, 40.7, -73.9]}') == {
            "bbox": [-74.0, 40.7, -73.9]
        }

    def test_non_numeric_unary_still_raises(self):
        # ``not x`` / ``~x`` are not literals and must still be rejected.
        with pytest.raises(UnexpectedAstError):
            _value_of("~5")
        with pytest.raises(UnexpectedAstError):
            _value_of("not True")


class TestHandleSingleToolNegativeNumbers:
    def test_negative_arg_end_to_end(self):
        call = _first_call("[searchWeather(latitude=40.84, longitude=-74.0046539)]")
        tool = handle_single_tool(call)
        assert tool.function.name == "searchWeather"
        assert json.loads(tool.function.arguments) == {
            "latitude": 40.84,
            "longitude": -74.0046539,
        }

    def test_negative_delta_end_to_end(self):
        call = _first_call("[updateInventory(quantity_delta=-20)]")
        tool = handle_single_tool(call)
        assert json.loads(tool.function.arguments) == {"quantity_delta": -20}


class TestGetParameterValueTuple:
    # JSON has no tuple type, so a tuple argument is decoded as a list rather
    # than dropping the whole call.
    def test_tuple_becomes_list(self):
        assert _value_of("(800, 600)") == [800, 600]

    def test_nested_tuple(self):
        assert _value_of("[(1, 2), (3, 4)]") == [[1, 2], [3, 4]]

    def test_tuple_with_negative(self):
        assert _value_of("(-74.0, 40.7)") == [-74.0, 40.7]

    def test_tuple_end_to_end(self):
        call = _first_call("[resize(size=(800, 600))]")
        tool = handle_single_tool(call)
        assert json.loads(tool.function.arguments) == {"size": [800, 600]}
