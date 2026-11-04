# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.tool_parsers.schema_types import convert_value, parameter_types

_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_INT_OR_STR = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
_BODY = '{\n  "version": "1.4.0",\n  "build": 7\n}'
_DEFS = {
    "S": {"type": "string"},
    "C": {"allOf": [{"$ref": "#/$defs/C"}, {"type": "string"}]},
    # Each def references the next one twice: 2**60 paths without memoization.
    **{f"D{i}": {"anyOf": [{"$ref": f"#/$defs/D{i + 1}"}] * 2} for i in range(60)},
    "D60": {"type": "string"},
}


@pytest.mark.parametrize(
    ("schema", "raw", "expected"),
    [
        # plain string and non-string types: unchanged
        ({"type": "string"}, "1.10", "1.10"),
        ({"type": "integer"}, "1.10", 1.1),
        # schemas that admit a string keep the raw text...
        (_NULLABLE_STR, "1.10", "1.10"),
        (_NULLABLE_STR, "1984", "1984"),
        (_NULLABLE_STR, _BODY, _BODY),
        (_NULLABLE_STR, '"q"', '"q"'),
        ({"oneOf": [{"type": "string"}, {"type": "null"}]}, "true", "true"),
        ({"type": ["string", "null"]}, "1984", "1984"),
        ({"enum": ["1.10", "2.0"]}, "1.10", "1.10"),
        ({"const": "1.10"}, "1.10", "1.10"),
        ({"$ref": "#/$defs/S"}, "1984", "1984"),
        ({"allOf": [{"$ref": "#/$defs/S"}, {"description": "d"}]}, "1.10", "1.10"),
        ({"$ref": "#/$defs/C"}, "1.10", "1.10"),
        ({"allOf": [_NULLABLE_STR, {"type": "string"}]}, "null", "null"),
        ({"$ref": "#/$defs/D0"}, "1.10", "1.10"),
        (_INT_OR_STR, "1.10", "1.10"),
        ({"type": ["number", "string"]}, "true", "true"),
        # ...unless the value decodes to another admitted type
        (_NULLABLE_STR, "null", None),
        (_INT_OR_STR, "25", 25),
        ({"type": ["number", "string"]}, "25", 25),
        ({"enum": [1, "a"]}, "1", 1),
        ({"type": ["number", "string"], "enum": [1, "x"]}, "1", 1),
        ({"anyOf": [{"type": "number"}, {"type": "string"}], "enum": [1, "x"]}, "1", 1),
        ({"allOf": [{"type": ["number", "string"]}, {"enum": [1, "x"]}]}, "1", 1),
        # no string admitted, or no type information: decode as before
        ({"anyOf": [{"type": "number"}, {"type": "null"}]}, "1.10", 1.1),
        ({"description": "untyped"}, "25", 25),
        ({"anyOf": [{"type": "string"}, {}]}, "25", 25),
    ],
)
def test_convert_value(schema, raw, expected):
    params = {
        "$defs": _DEFS,
        "properties": {"v": schema},
        "additionalProperties": schema,
    }
    props, extra = parameter_types([{"name": "f", "parameters": params}])["f"]
    for allowed in props["v"], extra:
        assert convert_value(raw, allowed, json.loads) == expected
