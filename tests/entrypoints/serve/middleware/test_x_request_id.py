# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import Response

from vllm.entrypoints.serve.middleware.x_request_id import XRequestIdMiddleware


@pytest.mark.parametrize(
    "downstream_ids",
    [(), ("downstream",), ("first", "second")],
    ids=["absent", "one", "two"],
)
@pytest.mark.parametrize("request_id", [None, "caller"], ids=["generated", "provided"])
def test_response_has_one_request_id(
    downstream_ids: tuple[str, ...], request_id: str | None
):
    """Echo the caller ID or generate a UUID, replacing any downstream IDs."""
    app = FastAPI()
    app.add_middleware(XRequestIdMiddleware)

    @app.get("/")
    async def route():
        response = Response("ok")
        for value in downstream_ids:
            response.headers.append("X-Request-Id", value)
        return response

    headers = {"X-Request-Id": request_id} if request_id is not None else {}
    with TestClient(app) as client:
        response = client.get("/", headers=headers)

    values = response.headers.get_list("x-request-id")
    assert len(values) == 1
    if request_id is not None:
        assert values == [request_id]
    else:
        generated = UUID(values[0])
        assert generated.version == 4
        assert generated.hex == values[0]
