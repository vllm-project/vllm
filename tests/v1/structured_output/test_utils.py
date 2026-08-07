# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.v1.structured_output as structured_output
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarGrammar,
    has_xgrammar_unsupported_json_features,
)

pytestmark = pytest.mark.cpu_test


class _FallbackGrammar(StructuredOutputGrammar):
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        raise NotImplementedError

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        raise NotImplementedError

    def rollback(self, num_tokens: int) -> None:
        raise NotImplementedError

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        bitmask[batch_index].fill_(7)

    def is_terminated(self) -> bool:
        return False

    def reset(self):
        raise NotImplementedError


def test_fill_bitmasks_batches_xgrammar_and_preserves_fallback(monkeypatch):
    manager = StructuredOutputManager.__new__(StructuredOutputManager)
    manager._grammar_bitmask = torch.zeros((4, 2), dtype=torch.int32)
    manager._full_mask = torch.tensor(-1, dtype=torch.int32)
    manager._xgr_batch_filler_local = threading.local()

    batch_filler = MagicMock()
    monkeypatch.setattr(
        structured_output,
        "xgr",
        SimpleNamespace(BatchGrammarMatcher=MagicMock(return_value=batch_filler)),
    )
    matcher = object()
    xgrammar = XgrammarGrammar(vocab_size=64, matcher=matcher, ctx=object())
    terminated = XgrammarGrammar(vocab_size=64, matcher=object(), ctx=object())
    terminated._is_terminated = True

    manager._fill_bitmasks(
        [
            (xgrammar, 0, True),
            (_FallbackGrammar(), 1, True),
            (terminated, 2, True),
            (xgrammar, 3, False),
        ]
    )

    batch_filler.batch_fill_next_token_bitmask.assert_called_once_with(
        [matcher], manager._grammar_bitmask, [0]
    )
    assert manager._grammar_bitmask[1].tolist() == [7, 7]
    assert manager._grammar_bitmask[2].tolist() == [-1, -1]
    assert manager._grammar_bitmask[3].tolist() == [-1, -1]


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
