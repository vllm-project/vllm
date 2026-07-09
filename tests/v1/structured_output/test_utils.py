# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output import backend_xgrammar
from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


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


def test_xgrammar_json_schema_validation_is_cached(monkeypatch):
    calls = 0

    class FakeGrammar:
        @staticmethod
        def from_json_schema(schema):
            nonlocal calls
            calls += 1

    class FakeXgrammar:
        Grammar = FakeGrammar

    cache = backend_xgrammar._validate_xgrammar_json_schema
    cache.cache_clear()
    monkeypatch.setattr(backend_xgrammar, "xgr", FakeXgrammar)

    try:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        backend_xgrammar.validate_xgrammar_grammar(
            SamplingParams(
                structured_outputs=StructuredOutputsParams(json=schema)
            )
        )
        backend_xgrammar.validate_xgrammar_grammar(
            SamplingParams(
                structured_outputs=StructuredOutputsParams(json=schema)
            )
        )

        assert calls == 1
    finally:
        cache.cache_clear()
