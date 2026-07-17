# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that per-request whitespace_pattern is threaded through the V1
structured output path (regression from V0). See closed PR #34790."""

import outlines_core.json_schema as json_schema
import pytest

from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.request import get_structured_output_key

pytestmark = pytest.mark.cpu_test

_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
}


def test_whitespace_pattern_makes_key_cache_distinct():
    p1 = StructuredOutputsParams(json=_SCHEMA, whitespace_pattern=None)
    p2 = StructuredOutputsParams(json=_SCHEMA, whitespace_pattern=r"[\n ]*")

    k1 = get_structured_output_key(p1)
    k2 = get_structured_output_key(p2)

    # key is now a 3-tuple carrying whitespace_pattern
    assert len(k1) == 3
    assert k1[0] == StructuredOutputOptions.JSON
    assert k1[2] is None
    assert k2[2] == r"[\n ]*"
    # same schema, different whitespace -> distinct cache keys
    assert k1 != k2


def test_whitespace_pattern_changes_outlines_regex():
    _, spec, ws = get_structured_output_key(
        StructuredOutputsParams(json=_SCHEMA, whitespace_pattern=r"[\n ]*")
    )
    default_regex = json_schema.build_regex_from_schema(spec)
    custom_regex = json_schema.build_regex_from_schema(
        spec, whitespace_pattern=ws
    )
    assert default_regex != custom_regex


def test_non_json_key_has_none_whitespace():
    key = get_structured_output_key(StructuredOutputsParams(regex="[0-9]+"))
    assert key == (StructuredOutputOptions.REGEX, "[0-9]+", None)
