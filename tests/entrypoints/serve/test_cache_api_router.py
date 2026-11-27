# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from vllm.entrypoints.common.cli_args import FrontendArgs
from vllm.entrypoints.launchers.app import build_app

CACHE_PATHS = ("/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache")


@pytest.mark.parametrize(
    "enabled,dev_mode", [(False, "0"), (True, "0"), (True, "1"), (False, "1")]
)
def test_cache_reset_routes_are_independently_enabled(enabled, dev_mode, monkeypatch):
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", dev_mode)
    app = build_app(
        FrontendArgs(enable_cache_reset=enabled), supported_tasks=("generate",)
    )
    paths = [route.path for route in app.routes]
    for path in CACHE_PATHS:
        assert paths.count(path) == int(enabled or dev_mode == "1")
    assert ("/collective_rpc" in paths) == (dev_mode == "1")
    assert ("/sleep" in paths) == (dev_mode == "1")
    assert ("/server_info" in paths) == (dev_mode == "1")


@pytest.mark.parametrize("root_path", ["", "/proxy"])
def test_cache_reset_requires_api_key_before_calling_engine(root_path, monkeypatch):
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "0")
    app = build_app(
        FrontendArgs(enable_cache_reset=True, api_key=["secret"], root_path=root_path),
        supported_tasks=("generate",),
    )
    engine = AsyncMock()
    engine.reset_prefix_cache.return_value = True
    app.state.engine_client = engine
    client = TestClient(app, root_path=root_path)
    for path in CACHE_PATHS:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            assert client.post(root_path + path, headers=headers).status_code == 401
    assert engine.mock_calls == []

    headers = {"Authorization": "Bearer secret"}
    response = client.post(
        root_path
        + "/reset_prefix_cache?reset_running_requests=true&reset_external=true",
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json() == {"success": True}
    engine.reset_prefix_cache.assert_awaited_once_with(True, True)
    for path in CACHE_PATHS[1:]:
        assert client.post(root_path + path, headers=headers).status_code == 200
    engine.reset_mm_cache.assert_awaited_once_with()
    engine.reset_encoder_cache.assert_awaited_once_with()

    engine.reset_prefix_cache.return_value = False
    response = client.post(root_path + "/reset_prefix_cache", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"success": False}
