# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the steering API router using a mock engine."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vllm.envs as envs
from vllm.entrypoints.serve.steering.api_router import attach_router, router


def _make_app(engine_mock) -> FastAPI:
    """Build a FastAPI app with the steering router and a mock engine."""
    app = FastAPI()
    app.state.engine_client = engine_mock
    app.include_router(router)
    return app


@pytest.fixture(autouse=True)
def _reset_steering_lock():
    """Prevent cross-test lock state leakage."""
    import vllm.entrypoints.serve.steering.api_router as mod

    mod._steering_lock = asyncio.Lock()


@pytest.fixture
def engine():
    return AsyncMock()


@pytest.fixture
def client(engine):
    app = _make_app(engine)
    return TestClient(app)


# --- POST /v1/steering/set ---


class TestSetSteering:
    def test_set_basic(self, client, engine):
        """Set vectors on one layer."""
        engine.collective_rpc.side_effect = [
            # Phase 1 validate: worker returns matched layers
            [[0]],
            # Phase 2 apply: worker returns applied layers
            [[0]],
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0, 2.0, 3.0]}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["layers_updated"] == [0]

    def test_set_multiple_layers(self, client, engine):
        engine.collective_rpc.side_effect = [
            [[0, 3]],
            [[0, 3]],
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0], "3": [2.0]}},
        )
        assert resp.status_code == 200
        assert resp.json()["layers_updated"] == [0, 3]

    def test_set_with_scales(self, client, engine):
        """Scale factors are pre-multiplied before sending to workers."""
        engine.collective_rpc.side_effect = [[[0]], [[0]]]
        resp = client.post(
            "/v1/steering/set",
            json={
                "vectors": {"0": [1.0, 2.0]},
                "scales": {"0": 3.0},
            },
        )
        assert resp.status_code == 200
        # Check the vectors sent to the worker were scaled
        validate_call = engine.collective_rpc.call_args_list[0]
        scaled_vectors = validate_call.kwargs.get(
            "args", validate_call[1].get("args", (None,))
        )[0]
        assert scaled_vectors[0] == [3.0, 6.0]

    def test_set_missing_layer_returns_400(self, client, engine):
        """Layer not found on any worker -> 400."""
        engine.collective_rpc.side_effect = [
            # Validate returns empty (layer 999 not found)
            [[]],
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"999": [1.0]}},
        )
        assert resp.status_code == 400
        assert "999" in resp.json()["error"]

    def test_set_partial_missing_layers(self, client, engine):
        """Some layers valid, some missing -> 400 for the missing ones."""
        engine.collective_rpc.side_effect = [
            # Worker only recognizes layer 0, not 999
            [[0]],
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0], "999": [1.0]}},
        )
        assert resp.status_code == 400
        assert "999" in resp.json()["error"]

    def test_set_no_steerable_layers(self, client, engine):
        """Model has no steerable layers at all -> 400."""
        engine.collective_rpc.side_effect = [
            [[]],  # empty from worker
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0]}},
        )
        assert resp.status_code == 400
        # When requested layers don't match any worker, the "missing
        # layers" check fires before the "no steerable layers" check.
        assert "not found" in resp.json()["error"]

    def test_set_vector_size_mismatch_returns_400(self, client, engine):
        """Worker raises ValueError for wrong size -> 400."""
        engine.collective_rpc.side_effect = ValueError(
            "Layer 0: expected vector of size 8, got 3"
        )
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0, 2.0, 3.0]}},
        )
        assert resp.status_code == 400
        assert "expected vector of size" in resp.json()["error"]

    def test_set_nonfinite_returns_400(self, client, engine):
        """Worker raises ValueError for non-finite -> 400."""
        engine.collective_rpc.side_effect = ValueError(
            "Layer 0: steering vector contains non-finite values"
        )
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0]}},
        )
        assert resp.status_code == 400
        assert "non-finite" in resp.json()["error"]

    def test_set_runtime_error_returns_500(self, client, engine):
        """Unexpected engine error -> 500."""
        engine.collective_rpc.side_effect = RuntimeError("GPU exploded")
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0]}},
        )
        assert resp.status_code == 500
        assert "GPU exploded" in resp.json()["error"]

    def test_set_with_replace(self, client, engine):
        """replace=True triggers a clear before apply."""
        engine.collective_rpc.side_effect = [
            [[0]],  # Phase 1 validate
            None,  # clear_steering_vectors
            [[0]],  # Phase 2 apply
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0]}, "replace": True},
        )
        assert resp.status_code == 200
        # Should have 3 collective_rpc calls: validate, clear, apply
        assert engine.collective_rpc.call_count == 3
        clear_call = engine.collective_rpc.call_args_list[1]
        assert clear_call[0][0] == "clear_steering_vectors"

    def test_set_without_replace_no_clear(self, client, engine):
        """replace=False (default) does not call clear."""
        engine.collective_rpc.side_effect = [
            [[0]],  # validate
            [[0]],  # apply
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [1.0]}},
        )
        assert resp.status_code == 200
        assert engine.collective_rpc.call_count == 2

    def test_set_scale_default_1(self, client, engine):
        """Unscaled vectors are passed through as-is."""
        engine.collective_rpc.side_effect = [[[0]], [[0]]]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"0": [5.0, 10.0]}},
        )
        assert resp.status_code == 200
        validate_call = engine.collective_rpc.call_args_list[0]
        scaled_vectors = validate_call.kwargs.get(
            "args", validate_call[1].get("args", (None,))
        )[0]
        assert scaled_vectors[0] == [5.0, 10.0]

    def test_set_empty_vectors(self, client, engine):
        """Empty vectors dict -> no steerable layers -> 400."""
        engine.collective_rpc.side_effect = [
            [[]],
        ]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {}},
        )
        assert resp.status_code == 400


# --- POST /v1/steering/clear ---


class TestClearSteering:
    def test_clear(self, client, engine):
        engine.collective_rpc.return_value = None
        resp = client.post("/v1/steering/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        engine.collective_rpc.assert_called_once_with("clear_steering_vectors")

    def test_clear_engine_error(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("fail")
        resp = client.post("/v1/steering/clear")
        assert resp.status_code == 500
        assert "fail" in resp.json()["error"]


# --- GET /v1/steering ---


class TestGetSteering:
    def test_get_no_active(self, client, engine):
        engine.collective_rpc.return_value = [{}]
        resp = client.get("/v1/steering")
        assert resp.status_code == 200
        assert resp.json()["active_layers"] == {}

    def test_get_active_layers(self, client, engine):
        engine.collective_rpc.return_value = [
            {0: {"norm": 1.5}, 3: {"norm": 2.0}},
        ]
        resp = client.get("/v1/steering")
        assert resp.status_code == 200
        active = resp.json()["active_layers"]
        assert "0" in active
        assert "3" in active
        assert active["0"]["norm"] == 1.5

    def test_get_merges_multiple_workers(self, client, engine):
        """PP workers own disjoint layers — results are merged."""
        engine.collective_rpc.return_value = [
            {0: {"norm": 1.0}},  # worker 0
            {5: {"norm": 2.0}},  # worker 1
        ]
        resp = client.get("/v1/steering")
        assert resp.status_code == 200
        active = resp.json()["active_layers"]
        assert "0" in active
        assert "5" in active

    def test_get_engine_error(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("fail")
        resp = client.get("/v1/steering")
        assert resp.status_code == 500
        assert "fail" in resp.json()["error"]


# --- attach_router ---


class TestAttachRouter:
    def test_attached_in_dev_mode(self):
        app = FastAPI()
        with patch.object(envs, "VLLM_SERVER_DEV_MODE", True):
            attach_router(app)
        paths = {r.path for r in app.routes}
        assert "/v1/steering/set" in paths
        assert "/v1/steering/clear" in paths
        assert "/v1/steering" in paths

    def test_not_attached_without_dev_mode(self):
        app = FastAPI()
        with patch.object(envs, "VLLM_SERVER_DEV_MODE", False):
            attach_router(app)
        paths = {r.path for r in app.routes}
        assert "/v1/steering/set" not in paths
