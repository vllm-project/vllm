# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features)


@pytest.fixture
def unsupported_string_schemas():
    return [
        {
            "type": "string",
            "format": "email"
        },
    ]


@pytest.fixture
def unsupported_integer_schemas():
    return [
        {
            "type": "integer",
            "multipleOf": 120
        },
    ]


@pytest.fixture
def unsupported_number_schemas():
    return [
        {
            "type": "number",
            "multipleOf": 120
        },
    ]


@pytest.fixture
def unsupported_array_schemas():
    return [
        {
            "type": "array",
            "uniqueItems": True
        },
        {
            "type": "array",
            "contains": {
                "type": "string"
            }
        },
        {
            "type": "array",
            "minContains": 1
        },
        {
            "type": "array",
            "maxContains": 5
        },
        {
            "type": "array",
            "minItems": 1
        },
        {
            "type": "array",
            "maxItems": 10
        },
    ]


@pytest.fixture
def unsupported_object_schemas():
    return [
        {
            "type": "object",
            "minProperties": 1
        },
        {
            "type": "object",
            "maxProperties": 5
        },
        {
            "type": "object",
            "propertyNames": {
                "pattern": "^[a-z]+$"
            }
        },
        {
            "type": "object",
            "patternProperties": {
                "^S": {
                    "type": "string"
                }
            }
        },
    ]


@pytest.fixture
def supported_schema():
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "status": {
                "type": "string"
            },
            "scores": {
                "type": "array",
                "items": {
                    "type": "number"
                }
            },
            "car_type": {
                "type": "string",
                "enum": ["sedan", "suv", "truck"]
            },
            "car_brand": {
                "type": "string",
                "pattern": "^[a-zA-Z]+$"
            },
            "short_description": {
                "type": "string",
                "maxLength": 50
            },
            "mileage": {
                "type": "number",
                "minimum": 0,
                "maximum": 1000000
            },
            "model_year": {
                "type": "integer",
                "exclusiveMinimum": 1900,
                "exclusiveMaximum": 2100
            },
            "long_description": {
                "type": "string",
                "minLength": 50,
                "maxLength": 2000
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {
                        "type": "string"
                    },
                    "city": {
                        "type": "string"
                    }
                }
            }
        }
    }


@pytest.mark.parametrize("schema_type", [
    "unsupported_string_schemas", "unsupported_integer_schemas",
    "unsupported_number_schemas", "unsupported_array_schemas",
    "unsupported_object_schemas"
])
def test_unsupported_json_features_by_type(schema_type, request):
    schemas = request.getfixturevalue(schema_type)
    for schema in schemas:
        assert has_xgrammar_unsupported_json_features(
            schema), f"Schema should be unsupported: {schema}"


def test_supported_json_features(supported_schema):
    assert not has_xgrammar_unsupported_json_features(
        supported_schema), "Schema should be supported"
