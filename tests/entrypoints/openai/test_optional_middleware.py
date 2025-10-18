# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for middleware that's off by default and can be toggled through
server arguments, mainly --api-key and --enable-request-id-headers.
"""

from http import HTTPStatus

import pytest
import requests

from ...utils import RemoteOpenAIServer

# Use a small embeddings model for faster startup and smaller memory footprint.
# Since we are not testing any chat functionality,
# using a chat capable model is overkill.
MODEL_NAME = "intfloat/multilingual-e5-small"


@pytest.fixture(scope="module")
def server(request: pytest.FixtureRequest):
    passed_params = []
    if hasattr(request, "param"):
        passed_params = request.param
    if isinstance(passed_params, str):
        passed_params = [passed_params]

    args = [
        "--runner",
        "pooling",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "2",
        *passed_params,
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_no_api_token(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("v1/models"))
    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_no_request_id_header(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))
    assert "X-Request-Id" not in response.headers


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_missing_api_token(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("v1/models"))
    assert response.status_code == HTTPStatus.UNAUTHORIZED


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_passed_api_token(server: RemoteOpenAIServer):
    response = requests.get(
        server.url_for("v1/models"), headers={"Authorization": "Bearer test"}
    )
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_not_v1_api_token(server: RemoteOpenAIServer):
    # Authorization check is skipped for any paths that
    # don't start with /v1 (e.g. /v1/chat/completions).
    response = requests.get(server.url_for("health"))
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    "server",
    ["--enable-request-id-headers"],
    indirect=True,
)
@pytest.mark.asyncio
async def test_enable_request_id_header(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))
    assert "X-Request-Id" in response.headers
    assert len(response.headers.get("X-Request-Id", "")) == 32


@pytest.mark.parametrize(
    "server",
    ["--enable-request-id-headers"],
    indirect=True,
)
@pytest.mark.asyncio
async def test_custom_request_id_header(server: RemoteOpenAIServer):
    response = requests.get(
        server.url_for("health"), headers={"X-Request-Id": "Custom"}
    )
    assert "X-Request-Id" in response.headers
    assert response.headers.get("X-Request-Id") == "Custom"
