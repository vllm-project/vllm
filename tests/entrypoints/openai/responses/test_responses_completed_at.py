# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for completed_at on ResponsesResponse (issue #63183)."""

import time

import pytest

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from vllm.sampling_params import SamplingParams


def _make_response(status: str) -> ResponsesResponse:
    request = ResponsesRequest(model="test-model", input="hello")
    created = int(time.time()) - 5
    return ResponsesResponse.from_request(
        request,
        SamplingParams(),
        model_name="test-model",
        created_time=created,
        output=[],
        status=status,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("status", ["completed"])
def test_completed_at_present_when_completed(status: str) -> None:
    resp = _make_response(status)
    assert resp.completed_at is not None
    assert resp.completed_at >= resp.created_at


@pytest.mark.parametrize("status", ["queued", "in_progress", "failed", "cancelled"])
def test_completed_at_absent_when_not_completed(status: str) -> None:
    resp = _make_response(status)
    assert resp.completed_at is None
