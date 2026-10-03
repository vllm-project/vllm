# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``validate_json_request`` FastAPI dependency.

These tests cover the body-type guard that rejects non-object JSON payloads
(e.g. ``[]``, ``true``, ``null``, ``123``) with HTTP 400 before they reach
pydantic model_validators that assume the input is a dict.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.exceptions import RequestValidationError

from vllm.entrypoints.serve.utils.api_utils import validate_json_request


def _make_request(body, content_type: str = "application/json"):
    request = MagicMock()
    request.headers = {"content-type": content_type}
    if isinstance(body, BaseException):
        request.json = AsyncMock(side_effect=body)
    else:
        request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, type_name",
    [
        ([], "list"),
        ([1, 2, 3], "list"),
        (True, "bool"),
        (False, "bool"),
        (None, "NoneType"),
        (123, "int"),
        (1.5, "float"),
        ("abc", "str"),
    ],
)
async def test_non_object_body_rejected(body, type_name):
    request = _make_request(body)
    with pytest.raises(RequestValidationError) as exc_info:
        await validate_json_request(request)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert f"got {type_name}" in errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"model": "x", "messages": []}])
async def test_object_body_accepted(body):
    # Empty dict and well-formed dict bodies must pass — downstream
    # pydantic validation is responsible for field-level checks.
    request = _make_request(body)
    await validate_json_request(request)


@pytest.mark.asyncio
async def test_invalid_json_passes_through():
    # JSON parse errors are intentionally not handled here; the existing
    # FastAPI/Pydantic path surfaces them as ``RequestValidationError``
    # with ``type='json_invalid'``. The dependency must not double-raise.
    request = _make_request(json.JSONDecodeError("Expecting value", "", 0))
    await validate_json_request(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/xml", "", "multipart/form-data"],
)
async def test_non_json_content_type_rejected(content_type):
    request = _make_request({}, content_type=content_type)
    with pytest.raises(RequestValidationError) as exc_info:
        await validate_json_request(request)
    errors = exc_info.value.errors()
    assert "Unsupported Media Type" in errors[0]


@pytest.mark.asyncio
async def test_json_content_type_with_charset_accepted():
    # ``application/json; charset=utf-8`` is the common form sent by many
    # HTTP clients and must be treated as JSON.
    request = _make_request({}, content_type="application/json; charset=utf-8")
    await validate_json_request(request)
