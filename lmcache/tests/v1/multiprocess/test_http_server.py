# SPDX-License-Identifier: Apache-2.0
"""Tests for the POST /cache/checksums endpoint (and other MP HTTP routes)."""

# Standard
from unittest.mock import MagicMock, PropertyMock

# Third Party
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.http_apis.dependencies import build_context
from lmcache.v1.multiprocess.http_server import app
import lmcache.c_ops as lmc_ops


def _make_kv_tensors(
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Create deterministic CPU KV tensors for testing."""
    torch.manual_seed(42)
    return [
        torch.randn(
            2,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]


@pytest.fixture
def mock_gpu_ctx():
    """Create a mock GPUCacheContext with kv_tensors."""
    ctx = MagicMock()
    tensors = _make_kv_tensors()
    type(ctx).kv_tensors = PropertyMock(return_value=tensors)
    type(ctx).block_size = PropertyMock(return_value=4)
    # KV tensors are built as [2, NB, BS, NH, HS] -> NL_X_TWO_NB_BS_NH_HS;
    # one homogeneous kernel group, so every layer reports the same format.
    fmt = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    ctx.engine_kv_formats.return_value = [fmt]
    ctx.engine_kv_format_per_layer.return_value = [fmt] * len(tensors)
    return ctx


@pytest.fixture
def mock_mixed_engine():
    """Engine whose context mixes two KV formats (a key+value layer and a
    key-only layer), so the endpoint gathers each along its own axis."""
    ctx = MagicMock()
    kv_kv = torch.randn(2, 4, 4, 2, 8)
    kv_idx = torch.randn(4, 4, 8)
    type(ctx).kv_tensors = PropertyMock(return_value=[kv_kv, kv_idx])
    ctx.engine_kv_format_per_layer.return_value = [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
    ]
    engine = MagicMock()
    engine.cache_contexts = {0: ctx}
    return engine


@pytest.fixture
def mock_engine(mock_gpu_ctx):
    """Create a mock engine with cache_contexts."""
    engine = MagicMock()
    engine.cache_contexts = {0: mock_gpu_ctx}
    return engine


@pytest.fixture
def client_with_engine(mock_engine):
    """Create a test client with a mocked engine wired onto the per-app context.

    The cache handlers resolve the engine through ``get_context`` (i.e.
    ``app.state.context``), so the context must be built, not just
    ``app.state.engine``.
    """
    app.state.engine = mock_engine
    app.state.context = build_context(mock_engine)
    client = TestClient(app)
    yield client
    client.close()
    app.state.engine = None
    app.state.context = None


@pytest.fixture
def client_no_engine():
    """Create a test client without an engine/context."""
    app.state.engine = None
    app.state.context = None
    client = TestClient(app)
    yield client
    client.close()


class TestCacheChecksumsEndpoint:
    """Tests for POST /cache/checksums."""

    def test_success_basic(self, client_with_engine):
        """Basic successful checksum request."""
        # chunk_size is in blocks; 1 block + chunk_size=1 -> 1 chunk.
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 1}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["chunk_size"] == 1
        assert data["num_chunks"] == 1
        assert len(data["chunk_checksums"]) == 1
        assert "block_id_ranges" in data

    def test_success_layerwise(self, client_with_engine):
        """Layerwise mode returns per-layer checksums."""
        resp = client_with_engine.post(
            "/cache/checksums",
            json={"block_ids": [0], "chunk_size": 1, "layerwise": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["layerwise"] is True
        cks = data["chunk_checksums"]
        assert "layer_0" in cks
        assert "layer_1" in cks

    def test_deterministic(self, client_with_engine):
        """Same request produces identical checksums."""
        body = {"block_ids": [0], "chunk_size": 1}
        d1 = client_with_engine.post("/cache/checksums", json=body).json()
        d2 = client_with_engine.post("/cache/checksums", json=body).json()
        assert d1["chunk_checksums"] == d2["chunk_checksums"]

    def test_mixed_format_supported(self, mock_mixed_engine):
        """Two different KV formats are gathered per layer, not rejected."""
        app.state.engine = mock_mixed_engine
        app.state.context = build_context(mock_mixed_engine)
        client = TestClient(app)
        try:
            resp = client.post(
                "/cache/checksums",
                json={"block_ids": [0], "chunk_size": 1, "layerwise": True},
            )
        finally:
            client.close()
            app.state.engine = None
            app.state.context = None
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        # Both layers checksummed despite different formats / block axes.
        assert set(data["chunk_checksums"]) == {"layer_0", "layer_1"}

    def test_range_block_ids(self, client_with_engine):
        """Two block ids -> two single-block chunks."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0, 1], "chunk_size": 1}
        )
        assert resp.status_code == 200
        data = resp.json()
        # 2 blocks, chunk_size=1 block -> 2 chunks
        assert data["num_chunks"] == 2

    def test_multiple_block_ids(self, client_with_engine):
        """Three block ids -> three single-block chunks."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0, 1, 2], "chunk_size": 1}
        )
        assert resp.status_code == 200
        data = resp.json()
        # 3 blocks, chunk_size=1 block -> 3 chunks
        assert data["num_chunks"] == 3

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_no_engine(self, client_no_engine):
        """503 when the engine/context is not initialized."""
        resp = client_no_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 1}
        )
        assert resp.status_code == 503

    def test_no_cache_contexts(self, client_with_engine, mock_engine):
        """501 when engine has no cache_contexts attribute."""
        mock_engine.cache_contexts = None
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 1}
        )
        assert resp.status_code == 501

    def test_unknown_instance_id(self, client_with_engine):
        """404 when instance_id is not registered."""
        resp = client_with_engine.post(
            "/cache/checksums",
            json={"block_ids": [0], "chunk_size": 1, "instance_id": 99},
        )
        assert resp.status_code == 404

    def test_missing_block_ids(self, client_with_engine):
        """400 when block_ids is empty."""
        resp = client_with_engine.post("/cache/checksums", json={"chunk_size": 1})
        assert resp.status_code == 400
        assert "block_ids" in resp.json()["detail"]

    def test_missing_chunk_size(self, client_with_engine):
        """400 when chunk_size is missing (defaults to 0)."""
        resp = client_with_engine.post("/cache/checksums", json={"block_ids": [0]})
        assert resp.status_code == 400
        assert "chunk_size" in resp.json()["detail"]

    def test_zero_chunk_size(self, client_with_engine):
        """400 when chunk_size is zero."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 0}
        )
        assert resp.status_code == 400

    def test_negative_chunk_size(self, client_with_engine):
        """400 when chunk_size is negative."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": -1}
        )
        assert resp.status_code == 400

    def test_empty_kv_caches(self, client_with_engine, mock_gpu_ctx):
        """404 when kv_tensors is empty."""
        type(mock_gpu_ctx).kv_tensors = PropertyMock(
            return_value=[],
        )
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 1}
        )
        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # Chunk boundary edge cases
    # ------------------------------------------------------------------

    def test_partial_last_chunk(self, client_with_engine):
        """3 blocks with chunk_size=2 blocks -> 2 chunks (2+1)."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0, 1, 2], "chunk_size": 2}
        )
        data = resp.json()
        assert data["num_chunks"] == 2
        assert len(data["chunk_checksums"]) == 2

    def test_single_block_single_chunk(self, client_with_engine):
        """Single block with chunk_size=1 block produces one chunk."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 1}
        )
        data = resp.json()
        assert data["num_chunks"] == 1

    # ------------------------------------------------------------------
    # Checksum validity
    # ------------------------------------------------------------------

    def test_checksums_are_valid_md5(self, client_with_engine):
        """All checksums are 32-char hex strings."""
        # 2 blocks, chunk_size=1 block -> 2 md5 digests.
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0, 1], "chunk_size": 1}
        )
        data = resp.json()
        for cksum in data["chunk_checksums"]:
            assert len(cksum) == 32
            int(cksum, 16)  # must be valid hex

    def test_block_id_ranges_in_response(
        self,
        client_with_engine,
    ):
        """Response includes compressed block_id_ranges."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0], "chunk_size": 1}
        )
        data = resp.json()
        assert "block_id_ranges" in data
        # Single block id is kept as-is (no range compression).
        assert data["block_id_ranges"] == [0]

    # ------------------------------------------------------------------
    # block_ids addressing
    # ------------------------------------------------------------------

    def test_block_ids_basic(self, client_with_engine):
        """block_ids with a block-level chunk_size."""
        # 2 blocks, chunk_size=1 block -> 2 chunks.
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": [0, 1], "chunk_size": 1}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["num_chunks"] == 2
        assert data["block_id_ranges"] == [0, 1]

    def test_block_ids_invalid_type(self, client_with_engine):
        """422 when block_ids is not a list of ints (FastAPI validation)."""
        resp = client_with_engine.post(
            "/cache/checksums", json={"block_ids": "abc", "chunk_size": 1}
        )
        assert resp.status_code == 422


class TestHealthAndMiscEndpoints:
    """Smoke tests for other endpoints on the MP HTTP server."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        yield
        app.state.engine = None
        app.state.context = None

    def test_root(self, client_no_engine):
        """GET / returns ok."""
        resp = client_no_engine.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_healthcheck_no_engine(self, client_no_engine):
        """503 when engine is not set."""
        resp = client_no_engine.get("/healthcheck")
        assert resp.status_code == 503

    def test_healthcheck_with_engine(
        self,
        client_with_engine,
    ):
        """200 when engine is available."""
        resp = client_with_engine.get("/healthcheck")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_clear_cache_no_engine(self, client_no_engine):
        """503 when engine/context is not set."""
        resp = client_no_engine.post("/cache/clear", json={})
        assert resp.status_code == 503

    def test_clear_cache_success(
        self,
        client_with_engine,
        mock_engine,
    ):
        """200 and engine.clear() called."""
        resp = client_with_engine.post("/cache/clear", json={"tier": "l1"})
        assert resp.status_code == 200
        mock_engine.clear.assert_called_once()

    def test_status_no_engine(self, client_no_engine):
        """503 when engine is not set."""
        resp = client_no_engine.get("/status")
        assert resp.status_code == 503

    def test_status_success(
        self,
        client_with_engine,
        mock_engine,
    ):
        """200 and engine.report_status() called."""
        mock_engine.report_status.return_value = {"ok": True}
        resp = client_with_engine.get("/status")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
