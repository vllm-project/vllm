# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from vllm.entrypoints.serve.elastic_ep import middleware as scaling


@pytest.fixture
def scaling_app():
    def _create(root_path: str = "") -> TestClient:
        app = FastAPI()
        app.add_middleware(scaling.ScalingMiddleware)
        app.root_path = root_path

        async def mock_endpoint():
            return JSONResponse({"status": "ok"})

        for path in ("/health", "/is_scaling_elastic_ep", "/v1/models"):
            app.add_api_route(path, mock_endpoint, methods=["GET"])
        return TestClient(app, root_path=root_path)

    scaling.set_scaling_elastic_ep(True)
    yield _create
    scaling.set_scaling_elastic_ep(False)


@pytest.mark.parametrize("root_path", ["", "/api"])
@pytest.mark.parametrize("path", ["/health", "/is_scaling_elastic_ep"])
def test_observability_paths_available_while_scaling(scaling_app, root_path, path):
    client = scaling_app(root_path)
    resp = client.get(root_path + path)
    assert resp.status_code == 200, f"{root_path + path} should stay up"


@pytest.mark.parametrize("root_path", ["", "/api"])
def test_inference_paths_paused_while_scaling(scaling_app, root_path):
    client = scaling_app(root_path)
    resp = client.get(root_path + "/v1/models")
    assert resp.status_code == 503
