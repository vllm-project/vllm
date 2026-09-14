# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import sys

import pytest

from vllm.v1.structured_output.backend_xgrammar import (
    flatten_allof_branches,
    has_xgrammar_unsupported_json_features,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def unsupported_string_schemas():
    return [
        {"type": "string", "format": "non_existing_format"},
        # pattern/format is compiled but length bounds are silently dropped,
        # so the combination must be rejected instead of producing quietly
        # wrong output
        {"type": "string", "pattern": "^a+$", "maxLength": 2},
        {"type": "string", "pattern": "^a+$", "minLength": 3},
        {"type": "string", "format": "email", "maxLength": 10},
    ]


@pytest.fixture
def unsupported_integer_schemas():
    return [
        {"type": "integer", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_number_schemas():
    return [
        {"type": "number", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_array_schemas():
    return [
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"type": "string"}},
        {"type": "array", "minContains": 1},
        {"type": "array", "maxContains": 5},
    ]


@pytest.fixture
def unsupported_object_schemas():
    return [
        {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}},
        {"type": "object", "patternProperties": {"^S": {"type": "string"}}},
    ]


@pytest.fixture
def unsupported_allof_schemas():
    return [
        {
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"x": {"type": "integer", "minimum": 10}},
                    "required": ["x"],
                }
            },
            "allOf": [
                {"$ref": "#/$defs/Base"},
                {
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                    "required": ["y"],
                },
            ],
        },
        {
            "allOf": [
                {"type": "string", "minLength": 1},
                {"type": "string", "maxLength": 5},
            ],
        },
    ]


@pytest.fixture
def supported_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"},
            "status": {"type": "string"},
            "scores": {"type": "array", "items": {"type": "number"}},
            "car_type": {"type": "string", "enum": ["sedan", "suv", "truck"]},
            "car_brand": {"type": "string", "pattern": "^[a-zA-Z]+$"},
            "short_description": {"type": "string", "maxLength": 50},
            "mileage": {"type": "number", "minimum": 0, "maximum": 1000000},
            "model_year": {
                "type": "integer",
                "exclusiveMinimum": 1900,
                "exclusiveMaximum": 2100,
            },
            "long_description": {"type": "string", "minLength": 50, "maxLength": 2000},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            },
        },
        "minProperties": 1,
        "maxProperties": 100,
    }


@pytest.mark.parametrize(
    "schema_type",
    [
        "unsupported_string_schemas",
        "unsupported_integer_schemas",
        "unsupported_number_schemas",
        "unsupported_array_schemas",
        "unsupported_object_schemas",
        "unsupported_allof_schemas",
    ],
)
def test_unsupported_json_features_by_type(schema_type, request):
    schemas = request.getfixturevalue(schema_type)
    for schema in schemas:
        assert has_xgrammar_unsupported_json_features(schema), (
            f"Schema should be unsupported: {schema}"
        )


def test_supported_json_features(supported_schema):
    assert not has_xgrammar_unsupported_json_features(supported_schema), (
        "Schema should be supported"
    )


def test_supported_single_branch_allof():
    # A single-branch `allOf` is enforced correctly by xgrammar and must not
    # be flagged as unsupported.
    schema = {
        "allOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        ],
    }
    assert not has_xgrammar_unsupported_json_features(schema)


def test_flatten_allof_ref_inheritance():
    schema = {
        "$defs": {
            "Animal": {
                "type": "object",
                "title": "Animal",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        },
        "title": "Dog",
        "allOf": [
            {"$ref": "#/$defs/Animal"},
            {
                "type": "object",
                "properties": {"breed": {"type": "string"}},
                "required": ["breed"],
            },
        ],
    }
    result = flatten_allof_branches(schema)
    assert result is not None
    assert "allOf" not in result
    assert result["type"] == "object"
    assert result["properties"] == {
        "name": {"type": "string"},
        "breed": {"type": "string"},
    }
    assert result["required"] == ["name", "breed"]
    assert result["$defs"] == schema["$defs"]


def test_flatten_allof_inline_object_branches():
    schema = {
        "allOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            {
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            },
        ]
    }
    result = flatten_allof_branches(schema)
    assert result is not None
    assert "allOf" not in result
    assert result["type"] == "object"
    assert result["properties"] == {
        "x": {"type": "integer"},
        "y": {"type": "string"},
    }
    assert result["required"] == ["x", "y"]


def test_flatten_allof_merged_schema_is_supported():
    mergeable = {
        "allOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            {
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            },
        ]
    }
    flattened = flatten_allof_branches(mergeable)
    assert flattened is not None
    assert not has_xgrammar_unsupported_json_features(flattened)
    assert has_xgrammar_unsupported_json_features(mergeable)

    unmergeable = {
        "allOf": [
            {"type": "string", "minLength": 1},
            {"type": "string", "maxLength": 5},
        ]
    }
    assert flatten_allof_branches(unmergeable) is None
    assert has_xgrammar_unsupported_json_features(unmergeable)


def test_flatten_allof_nested_in_properties():
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "allOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}},
                    {"type": "object", "properties": {"b": {"type": "integer"}}},
                ]
            }
        },
    }
    result = flatten_allof_branches(schema)
    assert result is not None
    nested = result["properties"]["nested"]
    assert "allOf" not in nested
    assert nested["type"] == "object"
    assert nested["properties"] == {
        "a": {"type": "string"},
        "b": {"type": "integer"},
    }


def test_flatten_allof_non_object_branches_returns_none():
    schema = {
        "allOf": [
            {"type": "string"},
            {"type": "string"},
        ]
    }
    assert flatten_allof_branches(schema) is None


def test_flatten_allof_conflicting_property_returns_none():
    schema = {
        "allOf": [
            {"type": "object", "properties": {"x": {"type": "string"}}},
            {
                "type": "object",
                "properties": {"x": {"type": "string", "maxLength": 5}},
            },
        ]
    }
    assert flatten_allof_branches(schema) is None


def test_flatten_allof_additional_properties_returns_none():
    schema = {
        "allOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "additionalProperties": False,
            },
            {"type": "object", "properties": {"y": {"type": "string"}}},
        ]
    }
    assert flatten_allof_branches(schema) is None


def test_flatten_allof_single_branch_unchanged():
    schema = {
        "allOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        ]
    }
    result = flatten_allof_branches(schema)
    assert result == schema
    assert not has_xgrammar_unsupported_json_features(result)


def test_flatten_allof_does_not_mutate_input():
    schema = {
        "$defs": {
            "Base": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
        },
        "allOf": [
            {"$ref": "#/$defs/Base"},
            {
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            },
        ],
    }
    original = copy.deepcopy(schema)
    flatten_allof_branches(schema)
    assert schema == original


def test_flatten_allof_ref_cycle_returns_none():
    """A $ref cycle through allOf must return None, not RecursionError."""
    schema = {
        "$defs": {
            "A": {
                "type": "object",
                "properties": {
                    "p": {"allOf": [{"$ref": "#/$defs/A"}, {"type": "object"}]}
                },
            }
        },
        "$ref": "#/$defs/A",
    }
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        assert flatten_allof_branches(schema) is None
    finally:
        sys.setrecursionlimit(old_limit)


def test_flatten_allof_shared_ref_is_mergeable():
    """Sibling allOf branches may share a $ref; that is not a cycle."""
    schema = {
        "$defs": {
            "Base": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            }
        },
        "allOf": [
            {"$ref": "#/$defs/Base"},
            {"$ref": "#/$defs/Base"},
        ],
    }
    result = flatten_allof_branches(schema)
    assert result is not None
    assert "allOf" not in result
    assert result["type"] == "object"
    assert result["properties"] == {"x": {"type": "string"}}
    assert result["$defs"] == schema["$defs"]
