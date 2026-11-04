# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from argparse import Namespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers
from vllm.entrypoints.scale_out.token_in_token_out.api_router import attach_router
from vllm.entrypoints.serve.middleware.authenticate import (
    GUARDED_PREFIX,
    AuthenticationMiddleware,
)
from vllm.tasks import SupportedTask

RENDER_PATHS = {
    "/v1/chat/completions/render",
    "/v1/chat/completions/derender",
    "/v1/completions/render",
    "/v1/completions/derender",
    "/v1/messages/render",
    "/v1/responses/render",
}
GENERATE_PATH = "/inference/v1/generate"
ABORT_PATH = "/inference/v1/abort_requests"
UNAUTHENTICATED_ABORT_PATH = "/abort_requests"
SCALE_OUT_PATHS = RENDER_PATHS | {GENERATE_PATH, ABORT_PATH}


def registered_paths(
    supported_tasks: tuple[SupportedTask, ...],
    *,
    tokens_only: bool = False,
    enable_scale_out: bool = False,
) -> set[str]:
    app = FastAPI()
    args = Namespace(
        tokens_only=tokens_only,
        enable_scale_out=enable_scale_out,
        enable_fault_tolerance=False,
    )
    app.state.args = args
    register_api_routers(args, app, supported_tasks)
    return {route.path for route in app.routes}


@pytest.mark.parametrize("path", sorted(SCALE_OUT_PATHS))
@pytest.mark.parametrize("authorized", [False, True])
def test_disabled_scale_out_routes_explain_opt_in_without_parsing(path, authorized):
    app = FastAPI()
    register_scale_out_api_routers(app, ("generate",))
    app.add_middleware(AuthenticationMiddleware, tokens=["secret"])
    headers = {"Authorization": "Bearer secret"} if authorized else {}
    client = TestClient(app)
    response = client.post(path, content="not JSON", headers=headers)

    assert response.status_code == (403 if authorized else 401)
    if authorized:
        assert "--enable-scale-out" in response.json()["error"]["message"]
        assert response.json()["error"]["type"] == "PermissionDeniedError"
        assert path not in app.openapi()["paths"]
        assert client.post("/unknown", headers=headers).status_code == 404


def test_disabled_scale_out_routes_are_logged_at_info(caplog):
    with caplog.at_level("INFO", logger="vllm.entrypoints.scale_out.factories"):
        registered_paths(("generate",))

    assert [
        record.levelno
        for record in caplog.records
        if "--enable-scale-out" in record.message
    ] == [logging.INFO]


@pytest.mark.parametrize("tokens_only", [False, True])
def test_enabled_generate_routes_use_native_request_validation(tokens_only):
    app = FastAPI()
    app.state.args = Namespace(
        tokens_only=tokens_only, enable_scale_out=not tokens_only
    )
    register_scale_out_api_routers(app, ("generate",))
    response = TestClient(app).post(
        GENERATE_PATH,
        content="not JSON",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


def test_scale_out_routes_can_be_enabled_for_generate_server():
    paths = registered_paths(("generate",), enable_scale_out=True)

    assert paths >= SCALE_OUT_PATHS


def test_render_routes_remain_enabled_for_render_server():
    paths = registered_paths(("render",))

    assert paths >= RENDER_PATHS
    assert GENERATE_PATH not in paths


def test_tokens_only_mode_enables_generate_routes_when_flag_is_unset():
    paths = registered_paths(("generate",), tokens_only=True)

    assert paths >= {GENERATE_PATH, ABORT_PATH, UNAUTHENTICATED_ABORT_PATH}


def test_abort_route_without_tokens_only_requires_api_key():
    """A decode pool that loads a tokenizer serves generate without
    --tokens-only and still needs to abort, but only behind --api-key."""
    paths = registered_paths(("generate",), enable_scale_out=True, tokens_only=False)

    assert ABORT_PATH in paths
    assert ABORT_PATH.startswith(GUARDED_PREFIX)
    assert UNAUTHENTICATED_ABORT_PATH not in paths


def test_unauthenticated_abort_route_is_not_registered_twice():
    """The RLHF dev router registers /abort_requests before the scale-out
    routers, so attaching the scale-out one too would add a shadowed duplicate."""
    app = FastAPI()
    app.state.args = Namespace(tokens_only=True)

    @app.post(UNAUTHENTICATED_ABORT_PATH)
    async def existing_abort_requests():
        return None

    attach_router(app)

    paths = [route.path for route in app.routes]
    assert paths.count(UNAUTHENTICATED_ABORT_PATH) == 1
    assert ABORT_PATH in paths
