# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json

import pytest

from vllm.tool_parsers.utils import (
    UnexpectedAstError,
    coerce_to_schema_type,
    contains_broken_string_literal,
    escape_ctrl_chars_in_strings,
    escape_nested_quotes_in_strings,
    extract_types_from_schema,
    get_parameter_value,
    handle_single_tool,
    make_valid_python,
    normalize_leading_zero_ints,
    rename_reserved_kwargs,
    restore_reserved_kwarg_names,
    salvage_calls_from_unparsable_block,
    split_top_level_calls,
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


def _value_of(expr: str):
    """Parse a single Python expression and run get_parameter_value on it."""
    return get_parameter_value(ast.parse(expr, mode="eval").body)


def _first_call(text: str) -> ast.Call:
    """Parse ``[foo(...)]`` and return the single ast.Call node."""
    statement = ast.parse(text).body[0]
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.List)
    call = statement.value.elts[0]
    assert isinstance(call, ast.Call)
    return call


def _bare_call(text: str) -> ast.Call:
    """Parse ``foo(...)`` (no list wrapper) and return the ast.Call node."""
    statement = ast.parse(text).body[0]
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    return statement.value


def _kwarg_constant(call: ast.Call, index: int = 0):
    """Return the constant value of the call's ``index``-th keyword arg."""
    value = call.keywords[index].value
    assert isinstance(value, ast.Constant)
    return value.value


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

    def test_multiline_string_argument_recovers_after_escape(self):
        # A raw newline inside a string argument is invalid Python, so
        # make_valid_python alone returns None; callers pre-escape control
        # chars (as the lfm2 parser does) and the escaped value must evaluate
        # back to the original.
        text = "[exec(command='line1\nline2')]"
        assert make_valid_python(text) is None
        result = make_valid_python(escape_ctrl_chars_in_strings(text))
        assert result is not None
        completed, added = result
        assert added == ""
        assert _kwarg_constant(_first_call(completed)) == "line1\nline2"

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
        assert _kwarg_constant(_first_call(text)) == "it's fine"


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
        assert _kwarg_constant(_bare_call(escaped)) == raw

    def test_nul_byte_inside_string_escaped(self):
        # ast.parse raises ValueError (not SyntaxError) on NUL anywhere in
        # the source, so an unescaped NUL in a string arg is unrecoverable.
        raw = "printf a\x00b"
        escaped = escape_ctrl_chars_in_strings(f"f(cmd='{raw}')")
        assert "\x00" not in escaped
        assert _kwarg_constant(_bare_call(escaped)) == raw

    def test_nul_byte_outside_strings_untouched(self):
        text = "f(a=1,\x00b=2)"
        assert escape_ctrl_chars_in_strings(text) == text


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


class TestGetParameterValueSet:
    # JSON has no set type; a set argument is decoded as a list (preserving
    # source order) instead of dropping the whole call, mirroring the tuple
    # handling.
    def test_set_becomes_list(self):
        assert _value_of("{'a', 'b'}") == ["a", "b"]

    def test_set_of_numbers(self):
        assert _value_of("{1, 2, 3}") == [1, 2, 3]

    def test_set_nested_in_dict(self):
        assert _value_of("{'tags': {'x', 'y'}}") == {"tags": ["x", "y"]}

    def test_set_end_to_end(self):
        call = _first_call("[label(tags={'urgent', 'bug'})]")
        tool = handle_single_tool(call)
        assert json.loads(tool.function.arguments) == {"tags": ["urgent", "bug"]}


class TestMakeValidPythonSets:
    def test_complete_set_in_model_text_accepted(self):
        # A set the model wrote and closed itself is a genuine argument.
        text = "[label(tags={'urgent', 'bug'})]"
        assert make_valid_python(text) == (text, "")

    def test_truncated_dict_completed_to_set_still_rejected(self):
        # `{"k` closes to `{"k"}` — a truncated dict, not a set the model
        # wrote. The completion added the `}`, so it must keep waiting.
        assert make_valid_python('[f(x={"k') is None


class TestGetParameterValueFString:
    # A placeholder-free f-string is a plain string constant, but ast parses
    # it as JoinedStr; it must not drop the call. Real placeholders are not
    # literals and must still be rejected.
    def test_constant_fstring(self):
        assert _value_of("f'hello world'") == "hello world"

    def test_empty_fstring(self):
        assert _value_of("f''") == ""

    def test_constant_fstring_in_list(self):
        assert _value_of("[f'a', 'b']") == ["a", "b"]

    def test_fstring_with_placeholder_still_raises(self):
        with pytest.raises(UnexpectedAstError):
            _value_of("f'{x}'")

    def test_constant_fstring_end_to_end(self):
        call = _first_call("[send(msg=f'hello')]")
        tool = handle_single_tool(call)
        assert json.loads(tool.function.arguments) == {"msg": "hello"}


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


class TestEscapeNestedQuotesInStrings:
    # Unescaped same-style quotes nested in a string argument (shell
    # commands like sed -n '1,9p') are Python juxtaposition errors. When
    # exactly one quote can syntactically close the string, close there and
    # escape the interior quotes; anything ambiguous is left unchanged.
    def test_sed_command_recovers_exact_value(self):
        text = "[bash(command='sed -n '360,450p' /testbed/common.py')]"
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert changed
        assert _kwarg_constant(_first_call(rewritten)) == (
            "sed -n '360,450p' /testbed/common.py"
        )

    def test_double_quote_variant(self):
        text = '[bash(command="grep "foo" log.txt")]'
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert changed
        assert _kwarg_constant(_first_call(rewritten)) == 'grep "foo" log.txt'

    def test_mixed_with_already_escaped_quote(self):
        text = "[bash(command='it\\'s 'x' end')]"
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert changed
        assert _kwarg_constant(_first_call(rewritten)) == "it's 'x' end"

    def test_doubly_nested_python_c_payload(self):
        # command='python3 -c "...latex(x, mul_symbol='\,')..."' — two false
        # closers both followed by ')', but only the real one yields text
        # that parses; ast-validation disambiguates.
        text = "[bash(command='python3 -c \"print(latex(x, mul_symbol='\\,'))\"')]"
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert changed
        assert _kwarg_constant(_first_call(rewritten)) == (
            "python3 -c \"print(latex(x, mul_symbol='\\,'))\""
        )

    def test_multiline_python_c_with_inner_string(self):
        # Raw newlines beyond the phantom close plus a single-quoted inner
        # string; recovery must survive both (caller re-escapes ctrl chars).
        inner = "python3 -c \"\ntest_str = 'is <b>fine</b>'\nprint(test_str)\n\""
        text = f"[bash(command='{inner}')]"
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert changed
        recovered = escape_ctrl_chars_in_strings(rewritten)
        assert _kwarg_constant(_first_call(recovered)) == inner

    def test_nested_quotes_before_non_string_argument(self):
        # A later NON-string argument adds no candidate closer, so the
        # nested string is still unambiguous.
        text = "[run(cmd='awk '{print}' f', count=2)]"
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert changed
        call = _first_call(rewritten)
        assert _kwarg_constant(call, 0) == "awk '{print}' f"

    @pytest.mark.parametrize(
        "text",
        [
            "[f(a='x')]",
            "[f(a='x', b='y')]",
            "[f(a={'k': 'v'})]",
            "[f(a='it\\'s fine')]",
            "[a(x='p'), b(y='q')]",
        ],
    )
    def test_valid_text_unchanged(self, text):
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert not changed
        assert rewritten == text

    @pytest.mark.parametrize(
        "text",
        [
            "[f(a='echo 'hi', b='x')]",
            "[run(cmd='awk '{print}' f', mode='fast')]",
        ],
    )
    def test_ambiguous_nesting_left_unchanged(self, text):
        # A later string argument's closing quote is itself a plausible
        # closer, making the nesting formally ambiguous; no rewrite is
        # attempted rather than guessing.
        rewritten, changed = escape_nested_quotes_in_strings(text)
        assert not changed
        assert rewritten == text


class TestContainsBrokenStringLiteral:
    @pytest.mark.parametrize(
        "text, broken",
        [
            ("[bash(command='sed -n '360,450p' /x')]", True),
            ("[f(a='echo 'hi', b='x')]", True),
            # Closing quote at the very end of partial text: the follower
            # is unknown until the next chunk arrives, so hold.
            ("[f(a='x'", True),
            ("[f(a='x', b='y')]", False),
            # Mid-string (no closing quote yet) is normal streaming.
            ('[f(a=\'grep -F "]" log', False),
            ("[f(a={'k': 'v'})]", False),
            ("[f(a='it\\'s fine')]", False),
        ],
    )
    def test_detection(self, text, broken):
        assert contains_broken_string_literal(text) is broken


def _call_name(call: ast.Call) -> str:
    """Return the plain function name of a parsed call."""
    assert isinstance(call.func, ast.Name)
    return call.func.id


class TestSplitTopLevelCalls:
    def test_basic_split(self):
        assert split_top_level_calls("[a(x=1), b(y=2)]") == ["a(x=1)", "b(y=2)"]

    def test_comma_inside_string_not_a_separator(self):
        # String args sit at depth >= 1, so even the bracket-only strategy
        # never splits on their commas.
        text = "[exec(command='echo a, b'), f(x=1)]"
        for respect_strings in (True, False):
            assert split_top_level_calls(text, respect_strings=respect_strings) == [
                "exec(command='echo a, b')",
                "f(x=1)",
            ]

    def test_broken_quote_desyncs_string_aware_scan(self):
        # A broken quote flips the string state, so every later separator
        # looks like string content to the string-aware scan and the block
        # cannot be split at all. Counting brackets only is immune: string
        # arguments live at depth >= 1, so their commas still never split.
        # This asymmetry is why salvage runs both strategies.
        text = "[f(a='x 'y', b='z'), ls(path='/tmp')]"
        assert len(split_top_level_calls(text, respect_strings=True)) == 1
        assert split_top_level_calls(text, respect_strings=False) == [
            "f(a='x 'y', b='z')",
            "ls(path='/tmp')",
        ]

    def test_nested_brackets_not_split(self):
        text = "[f(x=[1, 2], y={'a': (3, 4)}), g(z=5)]"
        assert split_top_level_calls(text, respect_strings=False) == [
            "f(x=[1, 2], y={'a': (3, 4)})",
            "g(z=5)",
        ]


class TestSalvageCallsFromUnparsableBlock:
    def test_good_sibling_recovered_from_broken_block(self):
        # The whole block is a SyntaxError (genuinely ambiguous nested
        # quote), so the AST-level salvage never gets a call list and ls
        # died with the block.
        calls = salvage_calls_from_unparsable_block(
            "[ls(path='/tmp'), f(a='x 'y', b='z')]"
        )
        assert [_call_name(c) for c in calls] == ["ls"]

    def test_never_returns_a_call_swallowing_reading(self):
        # Closing the broken string late makes the text parse but swallows
        # the sibling call into the argument value, so the tool would run
        # with corrupted arguments — worse than dropping it. Any rewriting
        # returned must preserve the block's call count.
        text = "[f(a='x 'y'), g(b='p 'q')]"
        rewritten, changed = escape_nested_quotes_in_strings(text)
        if changed:
            statement = ast.parse(rewritten).body[0]
            assert isinstance(statement, ast.Expr)
            assert isinstance(statement.value, ast.List)
            assert len(statement.value.elts) == 2
        else:
            assert rewritten == text

    def test_both_calls_recovered_per_segment(self):
        # Each segment on its own has a unique closing quote, so splitting
        # first recovers both calls with the values the model intended.
        calls = salvage_calls_from_unparsable_block(
            "[f(a='x 'y'), g(b='p 'q')]",
            rewrite=lambda segment: [escape_nested_quotes_in_strings(segment)[0]],
        )
        assert [_call_name(c) for c in calls] == ["f", "g"]
        assert [_kwarg_constant(c) for c in calls] == ["x 'y", "p 'q"]

    def test_rewrites_apply_per_segment(self):
        # A recoverable quirk (month=07) in one segment is fixed by the
        # rewrite ladder even though the sibling segment is unrecoverable.
        def rewrite(text):
            return [normalize_leading_zero_ints(text)]

        calls = salvage_calls_from_unparsable_block(
            "[f(a='x 'y', b='z'), g(month=07)]", rewrite=rewrite
        )
        assert [_call_name(c) for c in calls] == ["g"]

    def test_no_fabrication_when_nothing_parses(self):
        assert (
            salvage_calls_from_unparsable_block("[f(a='x 'y' 'z), g(b='p 'q' 'r)]")
            == []
        )

    def test_single_segment_not_salvaged(self):
        # One segment equals the whole block, which the caller's rewrite
        # ladder already tried; re-parsing it here could only duplicate work.
        assert salvage_calls_from_unparsable_block("[f(a='x 'y', b='z')]") == []

    def test_order_preserved(self):
        calls = salvage_calls_from_unparsable_block(
            "[b(y=2), broken(a='x 'y', b='z'), a(x=1)]"
        )
        assert [_call_name(c) for c in calls] == ["b", "a"]


class TestHandleSingleToolPositionalArgs:
    # Positional values carry no parameter name; they used to be dropped
    # silently, emitting a successful-looking call with missing arguments
    # (worse than a visible failure).
    @pytest.mark.parametrize(
        "expr",
        [
            "[get_weather('Paris')]",
            "[get_weather('Paris', unit='celsius')]",
            "[f(*['a', 'b'])]",
        ],
    )
    def test_positional_arguments_raise(self, expr):
        with pytest.raises(UnexpectedAstError):
            handle_single_tool(_first_call(expr))


class TestHandleSingleToolKwargsUnpack:
    # **-unpacking is ast.keyword(arg=None); arguments[None] serialized as a
    # literal "null" key. Dict literals merge with later-binding-wins
    # semantics; non-dict operands are rejected.
    def test_unpacked_dict_merges(self):
        tool = handle_single_tool(_first_call("[f(**{'a': 1})]"))
        assert json.loads(tool.function.arguments) == {"a": 1}

    def test_unpacked_dict_merges_with_keywords(self):
        tool = handle_single_tool(_first_call("[f(x=1, **{'a': 2})]"))
        assert json.loads(tool.function.arguments) == {"x": 1, "a": 2}

    def test_later_binding_wins(self):
        tool = handle_single_tool(_first_call("[f(**{'x': 1}, x=2)]"))
        assert json.loads(tool.function.arguments) == {"x": 2}

    def test_non_dict_unpack_raises(self):
        with pytest.raises(UnexpectedAstError):
            handle_single_tool(_first_call("[f(**[1, 2])]"))


class TestHandleSingleToolNonFinite:
    # A numeric literal like 1e999 overflows to float inf at parse time and
    # json.dumps rendered it as Infinity -- arguments no JSON parser accepts.
    @pytest.mark.parametrize(
        "expr", ["[f(x=1e999)]", "[f(x=-1e999)]", "[f(x=[1, 1e999])]"]
    )
    def test_non_finite_arguments_raise(self, expr):
        with pytest.raises(UnexpectedAstError):
            handle_single_tool(_first_call(expr))


class TestGetParameterValueNonJsonConstants:
    # bytes/Ellipsis/complex are ast.Constant but have no JSON form; they
    # must raise UnexpectedAstError (like other unsupported nodes) instead
    # of surfacing later as a TypeError inside json.dumps.
    @pytest.mark.parametrize("expr", ["b'abc'", "...", "1j"])
    def test_non_json_constant_raises(self, expr):
        with pytest.raises(UnexpectedAstError):
            _value_of(expr)

    def test_handle_single_tool_raises_ast_error_not_type_error(self):
        call = _first_call("[f(x=b'abc')]")
        with pytest.raises(UnexpectedAstError):
            handle_single_tool(call)

    @pytest.mark.parametrize(
        "expr, expected",
        [("'s'", "s"), ("1", 1), ("1.5", 1.5), ("True", True), ("None", None)],
    )
    def test_json_constants_still_pass(self, expr, expected):
        assert _value_of(expr) == expected


class TestNormalizeLeadingZeroInts:
    # Zero-padded ints (month=07) are a SyntaxError no other recovery path
    # handles; the rewrite must strip the padding without touching tokens
    # that are already valid Python.
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("[f(month=07)]", "[f(month=7)]"),
            ("[f(x=007, y=05)]", "[f(x=7, y=5)]"),
            ("[f(x=0_7)]", "[f(x=7)]"),
            ("[f(x=-07)]", "[f(x=-7)]"),
        ],
    )
    def test_leading_zeros_stripped(self, text, expected):
        assert normalize_leading_zero_ints(text) == expected
        assert ast.parse(expected)

    @pytest.mark.parametrize(
        "text",
        [
            "[f(x=0)]",
            "[f(x=00)]",
            "[f(x=0.5)]",
            "[f(x=07.5)]",
            "[f(x=1.07)]",
            "[f(x=1e07)]",
            "[f(x=0x1F)]",
            "[f(x=0o17)]",
            "[f(x=0b101)]",
            "[f(s='id 007')]",
            '[f(s="v0.07")]',
        ],
    )
    def test_valid_tokens_and_strings_untouched(self, text):
        assert normalize_leading_zero_ints(text) == text

    def test_end_to_end(self):
        normalized = normalize_leading_zero_ints("[set_date(month=07, day=05)]")
        tool = handle_single_tool(_first_call(normalized))
        assert json.loads(tool.function.arguments) == {"month": 7, "day": 5}


class TestRenameReservedKwargs:
    # A parameter named after a Python keyword (`from=1`) is a SyntaxError
    # that no escape/retry can recover; rename_reserved_kwargs rewrites it to
    # a parseable name and restore_reserved_kwarg_names is its exact inverse.
    def test_reserved_kwarg_renamed(self):
        text, changed = rename_reserved_kwargs("[memory_get(from=1)]")
        assert changed
        assert text == "[memory_get(from_pyreservedkw_=1)]"
        assert ast.parse(text)

    def test_round_trip_restores_original_name(self):
        renamed, _ = rename_reserved_kwargs("[memory_get(path='M.md', from=1)]")
        tool = handle_single_tool(_first_call(renamed))
        restored = restore_reserved_kwarg_names(json.loads(tool.function.arguments))
        assert restored == {"path": "M.md", "from": 1}

    def test_multiple_reserved_kwargs(self):
        text, changed = rename_reserved_kwargs('[search(in="docs/", from=0)]')
        assert changed
        args = json.loads(handle_single_tool(_first_call(text)).function.arguments)
        assert restore_reserved_kwarg_names(args) == {"in": "docs/", "from": 0}

    def test_keyword_inside_string_untouched(self):
        text, changed = rename_reserved_kwargs('[f(cmd="import x from y")]')
        assert not changed
        assert text == '[f(cmd="import x from y")]'

    def test_keyword_with_from_eq_inside_string_untouched(self):
        text, changed = rename_reserved_kwargs('[f(cmd="SELECT from=1")]')
        assert not changed

    def test_keyword_value_untouched(self):
        # `x=True` has a keyword as *value*, not parameter name.
        text, changed = rename_reserved_kwargs("[f(x=True, y=None)]")
        assert not changed

    def test_double_equals_untouched(self):
        text, changed = rename_reserved_kwargs('[f(expr="a", cond=1)]')
        assert not changed
        # `in ==` comparison-like text is not kwarg position anyway, but the
        # `==` guard also protects string-free edge text.
        text, changed = rename_reserved_kwargs("[f(x=1)]")
        assert not changed

    def test_non_keyword_names_untouched(self):
        text, changed = rename_reserved_kwargs("[f(fromage=1, classic=2)]")
        assert not changed

    def test_spaces_around_equals(self):
        text, changed = rename_reserved_kwargs("[f( from = 1 )]")
        assert changed
        assert ast.parse(text)

    def test_restore_leaves_normal_names_alone(self):
        assert restore_reserved_kwarg_names({"path": "x", "from": 1}) == {
            "path": "x",
            "from": 1,
        }
