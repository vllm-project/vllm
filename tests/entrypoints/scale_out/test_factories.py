# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from fastapi import FastAPI

from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers
from vllm.tasks import SupportedTask

RENDER_PATHS = {
    "/v1/chat/completions/render",
    "/v1/chat/completions/derender",
    "/v1/completions/render",
    "/v1/completions/derender",
    "/v1/messages/render",
}
GENERATE_PATH = "/inference/v1/generate"


def registered_paths(supported_tasks: tuple[SupportedTask, ...]) -> set[str]:
    app = FastAPI()
    app.state.args = Namespace(tokens_only=False)
    register_scale_out_api_routers(app, supported_tasks)
    return {route.path for route in app.routes}


def test_render_routes_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_RENDER_ENDPOINTS", raising=False)

    paths = registered_paths(("generate",))

    assert RENDER_PATHS.isdisjoint(paths)
    assert GENERATE_PATH in paths


def test_render_routes_can_be_enabled_for_generate_server(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_RENDER_ENDPOINTS", "1")

    paths = registered_paths(("generate",))

    assert paths >= RENDER_PATHS
    assert GENERATE_PATH in paths


def test_render_routes_remain_enabled_for_render_server(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_RENDER_ENDPOINTS", raising=False)

    paths = registered_paths(("render",))

    assert paths >= RENDER_PATHS
    assert GENERATE_PATH not in paths
