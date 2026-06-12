# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.api_server import register_exception_handlers


@pytest.mark.parametrize(
    "exception",
    [
        ValueError("invalid value"),
        TypeError("invalid type"),
        OverflowError("invalid overflow"),
    ],
)
def test_validation_exception_handlers_do_not_reraise(exception: Exception):
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False)
    register_exception_handlers(app)

    @app.get("/")
    async def raise_exception():
        raise exception

    response = TestClient(app).get("/")

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "BadRequestError"
