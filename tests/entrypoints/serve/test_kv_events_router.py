# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/entrypoints/serve/kv_events/api_router.py``.

Covers:

* Route registration: ``GET /kv_event_sources`` is attached unconditionally
  (not behind ``VLLM_SERVER_DEV_MODE``).
* Response shape: ``{"sources": [{data_parallel_rank, endpoint,
  replay_endpoint, topic}, ...]}`` sorted by DP rank, sourced from the
  engine client's ready-response aggregation.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.kv_events.api_router import attach_router


class _FakeEngineClient:
    """Stand-in for the EngineClient protocol surface the route uses."""

    def __init__(self, sources):
        self._sources = sources

    def get_kv_event_sources(self):
        return self._sources


def _build_app(sources) -> FastAPI:
    app = FastAPI()
    attach_router(app)
    app.state.engine_client = _FakeEngineClient(sources)
    return app


def test_route_registered_unconditionally(monkeypatch):
    """The discovery route must not be dev-mode gated (unlike
    /collective_rpc): consumers query it in production serving."""
    monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    app = FastAPI()
    attach_router(app)
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/kv_event_sources" in paths


def test_returns_sources_json():
    sources = [
        {
            "data_parallel_rank": 1,
            "endpoint": "tcp://0.0.0.0:41002",
            "replay_endpoint": None,
            "topic": "kv-events",
        },
        {
            "data_parallel_rank": 0,
            "endpoint": "tcp://0.0.0.0:41001",
            "replay_endpoint": "tcp://0.0.0.0:41003",
            "topic": "kv-events",
        },
    ]
    app = _build_app(sources)
    with TestClient(app) as client:
        r = client.get("/kv_event_sources")
    assert r.status_code == 200
    body = r.json()
    assert body == {"sources": sources}


def test_empty_sources():
    app = _build_app([])
    with TestClient(app) as client:
        r = client.get("/kv_event_sources")
    assert r.status_code == 200
    assert r.json() == {"sources": []}
