# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the step-barrier pause endpoints in vllm/entrypoints/serve/pause/.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.pause.api_router import attach_router


@pytest.fixture
def app():
    app = FastAPI()
    attach_router(app)
    return app


def test_pause_step_rejects_conflicting_params(app):
    app.state.engine_client = object()
    client = TestClient(app)
    resp = client.post("/pause/step?no_barrier=true&barrier=50")
    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "Cannot specify both no_barrier=true and barrier=<value>"
    )


def test_pause_step_requires_async_llm(app):
    app.state.engine_client = object()
    client = TestClient(app)
    resp = client.post("/pause/step")
    assert resp.status_code == 501
