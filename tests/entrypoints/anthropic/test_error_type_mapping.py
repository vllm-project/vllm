# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Anthropic ``error.type`` translation (issue #57997)."""

import json

import pytest

from vllm.entrypoints.anthropic.api_router import translate_error_response
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (418, "invalid_request_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (501, "api_error"),
        (503, "api_error"),
        (529, "overloaded_error"),
    ],
)
def test_error_type_is_translated_to_anthropic_vocabulary(
    status_code: int, expected_type: str
):
    response = translate_error_response(
        ErrorResponse(
            error=ErrorInfo(
                message="boom",
                # Deliberately use an internal type name to prove it is ignored.
                type="NotFoundError",
                code=status_code,
            )
        )
    )
    assert response.status_code == status_code
    assert json.loads(response.body) == {
        "type": "error",
        "error": {"type": expected_type, "message": "boom"},
    }
