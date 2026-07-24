# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
    validate_xgrammar_grammar,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def unsupported_string_schemas():
    return [
        {"type": "string", "format": "non_existing_format"},
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


@pytest.mark.parametrize(
    "schema",
    [
        # JSON Schema allows ``type`` to be a list of types, not just a
        # scalar string. The feature checks must fire for the list spelling
        # exactly as they do for the scalar one. The trigger is that ``type``
        # is a list at all -- not the presence of "null" -- so cover both a
        # single-element list, the common nullable form, and a list of two
        # real types.
        {"type": ["integer"], "multipleOf": 120},
        {"type": ["integer", "null"], "multipleOf": 120},
        {"type": ["number", "integer"], "multipleOf": 120},
        {"type": ["string", "null"], "format": "non_existing_format"},
        {"type": ["array", "null"], "uniqueItems": True},
        {"type": ["array", "null"], "contains": {"type": "string"}},
        {"type": ["object", "null"], "propertyNames": {"pattern": "^[a-z]+$"}},
        {"type": ["object", "null"], "patternProperties": {"^S": {"type": "string"}}},
    ],
)
def test_unsupported_json_features_with_list_type(schema):
    assert has_xgrammar_unsupported_json_features(schema), (
        f"Unsupported feature must be detected for list-type schema: {schema}"
    )


@pytest.mark.parametrize(
    "schema",
    [
        # Normalizing ``type`` to a set must not over-flag benign list-type
        # schemas: a list ``type`` without an unsupported feature is fine.
        {"type": ["string", "null"]},
        {"type": ["string", "null"], "format": "email"},
        {"type": ["integer", "null"]},
        {"type": ["integer", "string"]},
        {"type": ["array", "null"], "items": {"type": "string"}},
    ],
)
def test_supported_list_type_json_features(schema):
    assert not has_xgrammar_unsupported_json_features(schema), (
        f"Schema should be supported: {schema}"
    )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "integer", "multipleOf": 3},
        {"type": ["integer", "null"], "multipleOf": 3},
    ],
)
def test_validate_xgrammar_grammar_rejects_unsupported_schema(schema):
    """The gate feeds request validation: an unsupported schema must be
    rejected the same way whether ``type`` is scalar or a list, instead of
    slipping through to xgrammar which would silently drop the constraint."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    with pytest.raises(ValueError, match="not supported by xgrammar"):
        validate_xgrammar_grammar(params)
