# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from fastapi import FastAPI
from starlette.testclient import TestClient

from vllm.entrypoints.serve.kv_offload.api_router import attach_router


class _FakeEngineCore:
    def __init__(self, error: Exception | None = None):
        self.calls: list[tuple] = []
        self._error = error

    async def call_utility_async(self, method: str, *args):
        self.calls.append((method, *args))
        if self._error is not None:
            raise self._error
        if method == "start_kv_peer_import":
            return {"import_id": "kv-import-0", "state": "connecting"}
        return [{"import_id": "kv-import-0", "state": "done"}]


def _client(engine_core: _FakeEngineCore) -> TestClient:
    app = FastAPI()
    attach_router(app)
    app.state.engine_client = SimpleNamespace(engine_core=engine_core)
    return TestClient(app)


def test_import_from_peer_forwards_arguments():
    core = _FakeEngineCore()
    response = _client(core).post(
        "/v1/kv_offload/import_from_peer",
        params={"host": "10.0.0.5", "port": 7777, "max_chunks": 100},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ranks": [{"import_id": "kv-import-0", "state": "connecting"}]
    }
    assert core.calls == [("start_kv_peer_import", "10.0.0.5", 7777, 100, 600.0)]


def test_import_status_lists_imports():
    core = _FakeEngineCore()
    response = _client(core).get(
        "/v1/kv_offload/import_status", params={"import_id": "kv-import-0"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "ranks": [[{"import_id": "kv-import-0", "state": "done"}]]
    }
    assert core.calls == [("kv_peer_import_status", "kv-import-0")]


def test_engine_error_is_a_bad_request():
    core = _FakeEngineCore(ValueError("KV peer import needs an OffloadingConnector"))
    response = _client(core).post(
        "/v1/kv_offload/import_from_peer", params={"host": "h", "port": 1}
    )
    assert response.status_code == 400
    assert "OffloadingConnector" in response.json()["error"]


class _FakeDPEngineCore(_FakeEngineCore):
    """Internal DP load balancing: one client drives several engines."""

    async def call_utility_all_async(self, method: str, *args):
        return [await self.call_utility_async(method, *args) for _ in range(2)]


def test_every_dp_rank_is_reported():
    core = _FakeDPEngineCore()
    response = _client(core).post(
        "/v1/kv_offload/import_from_peer", params={"host": "h", "port": 7777}
    )
    assert response.status_code == 200
    assert len(response.json()["ranks"]) == 2
    assert len(core.calls) == 2
