# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dev.cache.api_router import attach_router

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    ("params", "expected_args", "reset_successful"),
    [
        ({}, (False, False), True),
        ({"reset_running_requests": "true"}, (True, False), True),
        ({"reset_external": "true"}, (False, True), True),
        (
            {"reset_running_requests": "true", "reset_external": "true"},
            (True, True),
            False,
        ),
    ],
)
def test_reset_prefix_cache_route(params, expected_args, reset_successful):
    app = FastAPI()
    app.state.engine_client = AsyncMock()
    reset = app.state.engine_client.reset_prefix_cache
    reset.return_value = reset_successful
    attach_router(app)

    with TestClient(app) as client:
        response = client.post("/reset_prefix_cache", params=params)

    assert response.status_code == 200
    assert response.json() == {"success": reset_successful}
    reset.assert_awaited_once_with(*expected_args)
