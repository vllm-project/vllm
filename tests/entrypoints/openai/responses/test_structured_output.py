# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for structured output helpers in the Responses API."""

import json

from vllm.entrypoints.openai.responses.serving import (
    _constraint_to_content_format,
)
from vllm.sampling_params import StructuredOutputsParams


class TestConstraintToContentFormat:
    """Test _constraint_to_content_format helper."""

    def test_json_schema_string_is_parsed(self):
        """JSON schema passed as a string gets json.loads'd into a dict."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        params = StructuredOutputsParams(json=json.dumps(schema))
        result = _constraint_to_content_format(params)

        assert result == {"type": "json_schema", "json_schema": schema}

    def test_structural_tag_only_returns_none(self):
        """structural_tag is not a content constraint — should return None."""
        params = StructuredOutputsParams(structural_tag='{"type": "structural_tag"}')
        result = _constraint_to_content_format(params)

        assert result is None
