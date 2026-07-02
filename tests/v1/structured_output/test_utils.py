# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)
from vllm.v1.structured_output.utils import _apply_grammar_bitmask_torch

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


def _pack_allowed_tokens(token_ids: list[int]) -> int:
    value = 0
    for token_id in token_ids:
        value |= 1 << token_id
    return value


def test_apply_grammar_bitmask_torch_with_sparse_indices():
    logits = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    original = logits.clone()
    grammar_bitmask = torch.tensor(
        [
            [-1],
            [_pack_allowed_tokens([2, 5, 9])],
            [_pack_allowed_tokens([0])],
        ],
        dtype=torch.int32,
    )

    _apply_grammar_bitmask_torch(
        logits,
        grammar_bitmask,
        out_indices=[1],
        skip_out_indices=False,
    )

    assert torch.equal(logits[0], original[0])
    assert torch.equal(logits[2], original[2])
    assert torch.equal(logits[1, [2, 5, 9]], original[1, [2, 5, 9]])
    disallowed = [0, 1, 3, 4, 6, 7, 8]
    assert torch.isneginf(logits[1, disallowed]).all()


def test_apply_grammar_bitmask_torch_aligned_batch():
    logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    grammar_bitmask = torch.tensor(
        [
            [_pack_allowed_tokens([0, 1])],
            [_pack_allowed_tokens([8, 9])],
        ],
        dtype=torch.int32,
    )

    _apply_grammar_bitmask_torch(
        logits,
        grammar_bitmask,
        out_indices=[0, 1],
        skip_out_indices=True,
    )

    assert torch.isfinite(logits[0, [0, 1]]).all()
    assert torch.isneginf(logits[0, 2:]).all()
    assert torch.isneginf(logits[1, :8]).all()
    assert torch.isfinite(logits[1, [8, 9]]).all()
