# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.tasks import SupportedTask

RENDER_PATHS = {
    "/v1/chat/completions/render",
    "/v1/chat/completions/derender",
    "/v1/completions/render",
    "/v1/completions/derender",
    "/v1/messages/render",
}
GENERATE_PATH = "/inference/v1/generate"
SCALE_OUT_PATHS = RENDER_PATHS | {GENERATE_PATH}


def registered_paths(
    supported_tasks: tuple[SupportedTask, ...], *, tokens_only: bool = False
) -> set[str]:
    app = FastAPI()
    args = Namespace(tokens_only=tokens_only, enable_fault_tolerance=False)
    app.state.args = args
    register_api_routers(args, app, supported_tasks)
    return {route.path for route in app.routes}


def test_scale_out_routes_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", raising=False)

    paths = registered_paths(("generate",))

    assert SCALE_OUT_PATHS.isdisjoint(paths)


def test_disabled_scale_out_routes_are_logged(monkeypatch, caplog):
    monkeypatch.delenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", raising=False)

    with caplog.at_level("INFO", logger="vllm.entrypoints.scale_out.factories"):
        registered_paths(("generate",))

    assert any(
        "VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1" in record.message
        for record in caplog.records
    )


def test_scale_out_routes_can_be_enabled_for_generate_server(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", "1")

    paths = registered_paths(("generate",))

    assert paths >= SCALE_OUT_PATHS


def test_render_routes_remain_enabled_for_render_server(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", raising=False)

    paths = registered_paths(("render",))

    assert paths >= RENDER_PATHS
    assert GENERATE_PATH not in paths


def test_render_server_rejects_explicitly_disabled_scale_out_routes(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", "0")

    with pytest.raises(ValueError, match="VLLM_ENABLE_SCALE_OUT_ENDPOINTS=0"):
        registered_paths(("render",))


def test_tokens_only_mode_enables_generate_routes_when_flag_is_unset(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", raising=False)

    paths = registered_paths(("generate",), tokens_only=True)

    assert paths >= {GENERATE_PATH, "/abort_requests"}


def test_tokens_only_mode_rejects_explicitly_disabled_scale_out_routes(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", "0")

    with pytest.raises(ValueError, match="--tokens-only"):
        registered_paths(("generate",), tokens_only=True)
