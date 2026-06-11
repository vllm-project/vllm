# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    resolve_tool_dicts,
    resolve_tool_schema,
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


class TestResolveToolSchema:
    """Tests for resolve_tool_schema (pre-template $ref + anyOf resolution)."""

    def test_simple_ref_resolution(self):
        schema = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                }
            },
            "properties": {
                "home_address": {"$ref": "#/$defs/Address"},
            },
        }
        result = resolve_tool_schema(schema)
        assert "$defs" not in result
        addr = result["properties"]["home_address"]
        assert addr["type"] == "object"
        assert "street" in addr["properties"]
        assert "city" in addr["properties"]

    def test_nested_ref_resolution(self):
        schema = {
            "type": "object",
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"val": {"type": "integer"}},
                },
                "Outer": {
                    "type": "object",
                    "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                },
            },
            "properties": {
                "data": {"$ref": "#/$defs/Outer"},
            },
        }
        result = resolve_tool_schema(schema)
        assert "$defs" not in result
        inner = result["properties"]["data"]["properties"]["inner"]
        assert inner["type"] == "object"
        assert inner["properties"]["val"]["type"] == "integer"

    def test_ref_in_array_items(self):
        schema = {
            "type": "object",
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Item"},
                }
            },
        }
        result = resolve_tool_schema(schema)
        item_schema = result["properties"]["items"]["items"]
        assert item_schema["type"] == "object"
        assert "name" in item_schema["properties"]

    def test_anyof_nullable_flattened(self):
        schema = {
            "type": "object",
            "properties": {
                "page": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": 10,
                }
            },
        }
        result = resolve_tool_schema(schema)
        page = result["properties"]["page"]
        assert page["type"] == "integer"
        assert "anyOf" not in page
        assert page["default"] == 10

    def test_oneof_nullable_flattened(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "oneOf": [{"type": "integer"}, {"type": "null"}],
                }
            },
        }
        result = resolve_tool_schema(schema)
        assert result["properties"]["count"]["type"] == "integer"
        assert "oneOf" not in result["properties"]["count"]

    def test_anyof_complex_not_flattened(self):
        """anyOf with multiple non-null types should be left alone."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "null"},
                    ]
                }
            },
        }
        result = resolve_tool_schema(schema)
        assert "anyOf" in result["properties"]["value"]

    def test_anyof_with_constraints_not_flattened(self):
        """anyOf where the non-null variant has extra keys like enum."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "anyOf": [
                        {"type": "string", "enum": ["active", "inactive"]},
                        {"type": "null"},
                    ]
                }
            },
        }
        result = resolve_tool_schema(schema)
        assert "anyOf" in result["properties"]["status"]

    def test_no_mutation_of_original(self):
        schema: dict[str, object] = {
            "type": "object",
            "$defs": {
                "Foo": {"type": "string"},
            },
            "properties": {
                "bar": {"$ref": "#/$defs/Foo"},
            },
        }
        resolve_tool_schema(schema)  # type: ignore[arg-type]
        assert "$defs" in schema
        props = schema["properties"]
        assert isinstance(props, dict)
        assert "$ref" in props["bar"]

    def test_remote_ref_left_unresolved(self):
        schema = {
            "type": "object",
            "properties": {
                "ext": {"$ref": "https://evil.com/schema.json"},
            },
        }
        result = resolve_tool_schema(schema)
        assert result["properties"]["ext"]["$ref"] == "https://evil.com/schema.json"

    def test_missing_def_left_unresolved(self):
        schema = {
            "type": "object",
            "$defs": {},
            "properties": {
                "x": {"$ref": "#/$defs/Missing"},
            },
        }
        result = resolve_tool_schema(schema)
        assert "$ref" in result["properties"]["x"]

    def test_circular_ref_does_not_crash(self):
        """A self-referencing $def should hit the depth limit, not infinite loop."""
        schema = {
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/Node"},
                    },
                },
            },
            "properties": {
                "root": {"$ref": "#/$defs/Node"},
            },
        }
        result = resolve_tool_schema(schema)
        assert result["properties"]["root"]["type"] == "object"

    def test_schema_without_refs_unchanged(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = resolve_tool_schema(schema)
        assert result == schema

    def test_definitions_key_supported(self):
        """Older schemas use 'definitions' instead of '$defs'."""
        schema = {
            "type": "object",
            "definitions": {
                "Color": {"type": "string", "enum": ["red", "blue"]},
            },
            "properties": {
                "fav": {"$ref": "#/$defs/Color"},
            },
        }
        result = resolve_tool_schema(schema)
        assert "definitions" not in result

    def test_pydantic_v2_style_schema(self):
        """Full Pydantic v2 schema like the one in issue #39108."""
        schema = {
            "type": "object",
            "$defs": {
                "SearchField": {
                    "type": "object",
                    "required": ["field"],
                    "properties": {
                        "field": {"type": "string"},
                        "value": {"title": "Value", "default": None},
                    },
                },
                "SearchInput": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/SearchField"},
                            "default": [],
                        },
                        "search_all": {"type": "boolean", "default": False},
                    },
                },
            },
            "required": ["input_model"],
            "properties": {
                "input_model": {"$ref": "#/$defs/SearchInput"},
                "page_size": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": 10,
                },
                "page_number": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": 0,
                },
            },
        }
        result = resolve_tool_schema(schema)

        assert "$defs" not in result

        inp = result["properties"]["input_model"]
        assert inp["type"] == "object"
        assert inp["properties"]["search_all"]["type"] == "boolean"

        fields_items = inp["properties"]["fields"]["items"]
        assert fields_items["type"] == "object"
        assert "field" in fields_items["properties"]

        assert result["properties"]["page_size"]["type"] == "integer"
        assert "anyOf" not in result["properties"]["page_size"]
        assert result["properties"]["page_number"]["type"] == "integer"


class TestResolveToolDicts:
    """Tests for resolve_tool_dicts (batch tool dict resolution)."""

    def test_resolves_function_parameters(self):
        tool_dicts = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search items",
                    "parameters": {
                        "type": "object",
                        "$defs": {
                            "Query": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                },
                            }
                        },
                        "properties": {
                            "query": {"$ref": "#/$defs/Query"},
                        },
                    },
                },
            }
        ]
        result = resolve_tool_dicts(tool_dicts)
        params = result[0]["function"]["parameters"]
        assert "$defs" not in params
        assert params["properties"]["query"]["type"] == "object"

    def test_does_not_mutate_original(self):
        tool_dicts: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "$defs": {"X": {"type": "string"}},
                        "properties": {"a": {"$ref": "#/$defs/X"}},
                    },
                },
            }
        ]
        resolve_tool_dicts(tool_dicts)  # type: ignore[arg-type]
        func = tool_dicts[0]["function"]
        assert isinstance(func, dict)
        assert "$defs" in func["parameters"]

    def test_tool_without_parameters(self):
        tool_dicts = [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "Does nothing",
                },
            }
        ]
        result = resolve_tool_dicts(tool_dicts)
        assert result[0]["function"]["name"] == "noop"
        assert "parameters" not in result[0]["function"]

    def test_multiple_tools(self):
        tool_dicts = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "anyOf": [
                                    {"type": "integer"},
                                    {"type": "null"},
                                ]
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "parameters": {
                        "type": "object",
                        "$defs": {
                            "Cfg": {
                                "type": "object",
                                "properties": {"v": {"type": "string"}},
                            }
                        },
                        "properties": {"config": {"$ref": "#/$defs/Cfg"}},
                    },
                },
            },
        ]
        result = resolve_tool_dicts(tool_dicts)
        assert (
            result[0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"
        )
        assert (
            result[1]["function"]["parameters"]["properties"]["config"]["type"]
            == "object"
        )
