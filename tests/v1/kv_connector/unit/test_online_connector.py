# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OnlineHiddenStatesConnector and supporting modules.

Tests cover:
  - PercentileTracker: sliding window, warmup, threshold, observe_and_check
  - HiddenStatesWriter: async write pipeline, compression, flush
  - OnlineHiddenStatesConnector: integration with predictable dummy model
    (mirrors the PR's test_extraction.py but uses the online connector)
"""

import gc
import os
import time

import numpy as np
import pytest
import torch

# ────────────────────────────────────────────────────────────────
# PercentileTracker tests
# ────────────────────────────────────────────────────────────────

from vllm.distributed.kv_transfer.kv_connector.online.percentile_tracker import (
    PercentileTracker,
)


class TestPercentileTracker:
    """Tests for the sliding-window percentile tracker."""

    def test_warmup_captures_everything(self):
        """During warmup (< min_samples), should_capture always returns True."""
        tracker = PercentileTracker(
            percentile=10.0, window_size=100, min_samples=50,
        )
        # Even a very high acceptance length should be captured during warmup
        for _ in range(49):
            assert tracker.should_capture(100.0) is True
            tracker.observe(100.0)

    def test_filtering_after_warmup(self):
        """After warmup, only worst-percentile values are captured."""
        tracker = PercentileTracker(
            percentile=10.0, window_size=1000, min_samples=100,
        )
        # Fill with values 1..200 (warmup + enough for stable percentile)
        for i in range(200):
            tracker.observe(float(i + 1))

        # 10th percentile of 1..200 is ~20.9
        # Values <= threshold should be captured
        assert tracker.should_capture(1.0) is True
        assert tracker.should_capture(10.0) is True
        # Values well above threshold should NOT be captured
        assert tracker.should_capture(100.0) is False
        assert tracker.should_capture(200.0) is False

    def test_observe_and_check_atomic(self):
        """observe_and_check should return decision AND record observation."""
        tracker = PercentileTracker(
            percentile=50.0, window_size=100, min_samples=10,
        )
        # Fill past warmup
        for i in range(20):
            tracker.observe(float(i))

        initial_samples = tracker.get_stats()["num_samples"]
        result = tracker.observe_and_check(5.0)
        after_samples = tracker.get_stats()["num_samples"]

        assert isinstance(result, bool)
        assert after_samples == initial_samples + 1

    def test_sliding_window_eviction(self):
        """Old observations should be evicted when window is full."""
        tracker = PercentileTracker(
            percentile=50.0, window_size=10, min_samples=5,
        )
        # Fill window with low values
        for _ in range(10):
            tracker.observe(1.0)
        # Now fill with high values — old low values get evicted
        for _ in range(10):
            tracker.observe(100.0)

        stats = tracker.get_stats()
        assert stats["num_samples"] == 10
        assert stats["min_acceptance"] == 100.0

    def test_get_stats_empty(self):
        tracker = PercentileTracker()
        stats = tracker.get_stats()
        assert stats["num_samples"] == 0
        assert stats["percentile_threshold"] is None
        assert stats["mean_acceptance"] is None

    def test_get_stats_populated(self):
        tracker = PercentileTracker(percentile=25.0, min_samples=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            tracker.observe(v)

        stats = tracker.get_stats()
        assert stats["num_samples"] == 10
        assert stats["mean_acceptance"] == pytest.approx(5.5)
        assert stats["min_acceptance"] == 1.0
        assert stats["max_acceptance"] == 10.0
        assert stats["percentile_threshold"] is not None
        # 25th percentile of 1..10 ≈ 3.25
        assert stats["percentile_threshold"] == pytest.approx(3.25)

    def test_threshold_updates_periodically(self):
        """Threshold cache invalidates every 10 observations."""
        tracker = PercentileTracker(
            percentile=50.0, window_size=100, min_samples=10,
        )
        for i in range(20):
            tracker.observe(float(i))

        t1 = tracker.get_stats()["percentile_threshold"]

        # Add 10 more high values to shift the distribution
        for _ in range(10):
            tracker.observe(1000.0)

        t2 = tracker.get_stats()["percentile_threshold"]
        assert t2 > t1


# ────────────────────────────────────────────────────────────────
# HiddenStatesWriter tests
# ────────────────────────────────────────────────────────────────

from vllm.distributed.kv_transfer.kv_connector.online.hidden_states_writer import (
    HiddenStatesWriter,
)


class TestHiddenStatesWriter:
    """Tests for the async safetensors writer."""

    def test_write_cpu_tensors(self, tmp_path):
        """Write CPU tensors (no CUDA needed)."""
        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=False,
        )
        hs = torch.randn(10, 4, 256)
        tids = torch.arange(10, dtype=torch.long)
        filename = str(tmp_path / "test_req.safetensors")

        writer.write_async(hs, tids, filename)
        writer.flush(timeout=5.0)

        assert os.path.exists(filename)
        import safetensors.torch
        loaded = safetensors.torch.load_file(filename)
        assert "hidden_states" in loaded
        assert "token_ids" in loaded
        assert loaded["hidden_states"].shape == (10, 4, 256)
        assert torch.equal(loaded["token_ids"], tids)

        writer.shutdown()

    def test_write_with_compression(self, tmp_path):
        """Write with zstd compression."""
        try:
            import zstandard  # noqa: F401
        except ImportError:
            pytest.skip("zstandard not installed")

        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=True,
            compression_level=3,
        )
        hs = torch.randn(20, 4, 128)
        tids = torch.arange(20, dtype=torch.long)
        filename = str(tmp_path / "compressed_req.safetensors")

        writer.write_async(hs, tids, filename)
        writer.flush(timeout=5.0)

        # Compressed file has .zst extension
        compressed_path = filename + ".zst"
        assert os.path.exists(compressed_path)

        # Verify we can decompress and read
        import zstandard as zstd
        import safetensors.torch
        dctx = zstd.ZstdDecompressor()
        with open(compressed_path, "rb") as f:
            raw = dctx.decompress(f.read())
        # Write decompressed to temp file and load
        tmp_st = str(tmp_path / "decompressed.safetensors")
        with open(tmp_st, "wb") as f:
            f.write(raw)
        loaded = safetensors.torch.load_file(tmp_st)
        assert loaded["hidden_states"].shape == (20, 4, 128)

        writer.shutdown()

    def test_multiple_writes(self, tmp_path):
        """Multiple async writes complete correctly."""
        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=False,
        )
        for i in range(5):
            hs = torch.randn(8, 2, 64)
            tids = torch.arange(8, dtype=torch.long)
            filename = str(tmp_path / f"req_{i}.safetensors")
            writer.write_async(hs, tids, filename)

        writer.flush(timeout=10.0)

        files = list(tmp_path.glob("req_*.safetensors"))
        assert len(files) == 5

        writer.shutdown()

    def test_shutdown_stats(self, tmp_path, capsys):
        """Shutdown logs write stats."""
        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=False,
        )
        hs = torch.randn(4, 2, 32)
        tids = torch.arange(4, dtype=torch.long)
        writer.write_async(hs, tids, str(tmp_path / "s.safetensors"))
        writer.flush(timeout=5.0)

        assert writer._total_writes == 1
        assert writer._total_bytes > 0

        writer.shutdown()

    def test_data_integrity_round_trip(self, tmp_path):
        """Verify tensor values survive the full async pipeline."""
        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=False,
        )
        # Use deterministic values so we can verify exact match
        hs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        tids = torch.tensor([100, 200], dtype=torch.long)
        filename = str(tmp_path / "integrity.safetensors")

        writer.write_async(hs, tids, filename)
        writer.flush(timeout=5.0)

        import safetensors.torch
        loaded = safetensors.torch.load_file(filename)
        assert torch.equal(loaded["hidden_states"], hs)
        assert torch.equal(loaded["token_ids"], tids)

        writer.shutdown()

    def test_data_integrity_with_compression(self, tmp_path):
        """Verify tensor values survive compression round-trip."""
        try:
            import zstandard as zstd  # noqa: F401
        except ImportError:
            pytest.skip("zstandard not installed")

        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=True,
            compression_level=3,
        )
        hs = torch.arange(120, dtype=torch.float32).reshape(5, 4, 6)
        tids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
        filename = str(tmp_path / "integrity_zst.safetensors")

        writer.write_async(hs, tids, filename)
        writer.flush(timeout=5.0)

        compressed_path = filename + ".zst"
        assert os.path.exists(compressed_path)

        import zstandard
        dctx = zstandard.ZstdDecompressor()
        with open(compressed_path, "rb") as f:
            raw = dctx.decompress(f.read())
        tmp_st = str(tmp_path / "decompressed.safetensors")
        with open(tmp_st, "wb") as f:
            f.write(raw)

        import safetensors.torch
        loaded = safetensors.torch.load_file(tmp_st)
        assert torch.equal(loaded["hidden_states"], hs)
        assert torch.equal(loaded["token_ids"], tids)

        writer.shutdown()

    def test_concurrent_writes_stress(self, tmp_path):
        """Stress test: many concurrent writes complete without data loss."""
        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=False,
        )
        num_writes = 50
        expected = {}
        for i in range(num_writes):
            seq_len = 4 + (i % 10)
            hs = torch.full((seq_len, 2, 32), fill_value=float(i))
            tids = torch.arange(seq_len, dtype=torch.long)
            filename = str(tmp_path / f"stress_{i}.safetensors")
            writer.write_async(hs, tids, filename)
            expected[filename] = (hs.clone(), tids.clone())

        writer.flush(timeout=30.0)

        import safetensors.torch
        for filename, (exp_hs, exp_tids) in expected.items():
            assert os.path.exists(filename), f"Missing: {filename}"
            loaded = safetensors.torch.load_file(filename)
            assert torch.equal(loaded["hidden_states"], exp_hs), (
                f"Data mismatch in {filename}"
            )
            assert torch.equal(loaded["token_ids"], exp_tids)

        assert writer._total_writes == num_writes
        writer.shutdown()

    def test_pinned_buffer_pool_reuse(self, tmp_path):
        """Verify pinned buffer pool works correctly."""
        from vllm.distributed.kv_transfer.kv_connector.online\
            .hidden_states_writer import _PinnedBufferPool

        pool = _PinnedBufferPool(max_per_key=4)

        # Get a buffer — should allocate fresh
        buf1 = pool.get((10, 4, 256), torch.float32)
        assert buf1.shape == (10, 4, 256)
        assert buf1.dtype == torch.float32

        # Return it
        pool.put(buf1)
        assert len(pool._pool[((10, 4, 256), torch.float32)]) == 1

        # Get again — should reuse the same buffer
        buf2 = pool.get((10, 4, 256), torch.float32)
        assert buf2.data_ptr() == buf1.data_ptr()
        assert len(pool._pool[((10, 4, 256), torch.float32)]) == 0

        # Different shape gets a new buffer
        buf3 = pool.get((5, 2, 128), torch.float32)
        assert buf3.shape == (5, 2, 128)
        assert buf3.data_ptr() != buf1.data_ptr()

        # Pool respects max_per_key
        for _ in range(10):
            pool.put(torch.empty(3, 3, dtype=torch.float32))
        assert len(pool._pool[((3, 3), torch.float32)]) == 4


# ────────────────────────────────────────────────────────────────
# request_finished cleanup tests
# ────────────────────────────────────────────────────────────────


class TestRequestFinishedCleanup:
    """Tests that request_finished properly cleans up internal state
    and performs deferred percentile filtering."""

    def _make_connector(self, tmp_path, capture_percentile=0.0):
        """Helper to create a minimally-configured connector with mocks."""
        from unittest.mock import MagicMock, patch
        from vllm.distributed.kv_transfer.kv_connector.online\
            .online_hidden_states_connector import (
            OnlineHiddenStatesConnector,
        )

        # Patch the parent __init__ to avoid full vllm init
        with patch.object(
            OnlineHiddenStatesConnector, '__init__', lambda self, *a, **k: None
        ):
            connector = OnlineHiddenStatesConnector.__new__(
                OnlineHiddenStatesConnector,
            )

        # Manually set the attributes that __init__ would set
        connector._block_size = 16
        connector._storage_path = str(tmp_path)
        connector._vllm_config = MagicMock()
        connector._kv_transfer_config = MagicMock()
        connector._use_compression = False
        connector._compression_level = 3
        connector._capture_percentile = capture_percentile
        connector._percentile_tracker = None
        connector._request_filenames = {}
        connector._active_requests = {}
        connector._req_blocks = {}
        connector._prev_step_info = {}
        connector._req_acceptance_lengths = {}
        connector._total_captured = 0
        connector._total_skipped = 0
        connector._writer = None
        connector.cache_layers = []
        connector.num_hidden_states = 3

        if capture_percentile > 0:
            from vllm.distributed.kv_transfer.kv_connector.online\
                .percentile_tracker import PercentileTracker
            connector._percentile_tracker = PercentileTracker(
                percentile=capture_percentile,
                window_size=1000,
                min_samples=5,  # low for testing
            )

        return connector

    def test_request_finished_cleans_all_state(self, tmp_path):
        """request_finished should remove req from all internal dicts."""
        connector = self._make_connector(tmp_path)

        req_id = "test-req-123"
        connector._request_filenames[req_id] = "/tmp/test.safetensors"
        connector._active_requests[req_id] = "mock_request_data"
        connector._req_blocks[req_id] = [0, 1, 2]
        connector._prev_step_info[req_id] = (100, 3)
        connector._req_acceptance_lengths[req_id] = [1.5, 2.0]

        from unittest.mock import MagicMock
        mock_request = MagicMock()
        mock_request.request_id = req_id

        freed, params = connector.request_finished(mock_request, [0, 1, 2])

        assert freed is False
        assert params == {"hidden_states_path": "/tmp/test.safetensors"}
        assert req_id not in connector._request_filenames
        assert req_id not in connector._active_requests
        assert req_id not in connector._req_blocks
        assert req_id not in connector._prev_step_info
        assert req_id not in connector._req_acceptance_lengths

    def test_request_finished_handles_missing_req(self, tmp_path):
        """request_finished should not crash for unknown request IDs."""
        connector = self._make_connector(tmp_path)

        from unittest.mock import MagicMock
        mock_request = MagicMock()
        mock_request.request_id = "nonexistent-req"

        freed, params = connector.request_finished(mock_request, [])

        assert freed is False
        assert params == {"hidden_states_path": None}

    def test_request_finished_isolates_requests(self, tmp_path):
        """Finishing one request should not affect others."""
        connector = self._make_connector(tmp_path)

        for rid in ["req-a", "req-b"]:
            connector._request_filenames[rid] = f"/tmp/{rid}.safetensors"
            connector._active_requests[rid] = f"data_{rid}"
            connector._req_blocks[rid] = [0, 1]
            connector._prev_step_info[rid] = (50, 2)

        from unittest.mock import MagicMock
        mock_req_a = MagicMock()
        mock_req_a.request_id = "req-a"

        connector.request_finished(mock_req_a, [0, 1])

        assert "req-a" not in connector._request_filenames
        assert "req-a" not in connector._active_requests
        assert "req-b" in connector._request_filenames
        assert "req-b" in connector._active_requests
        assert "req-b" in connector._req_blocks
        assert "req-b" in connector._prev_step_info


# ────────────────────────────────────────────────────────────────
# Deferred percentile filtering tests
# ────────────────────────────────────────────────────────────────


class TestDeferredPercentileFiltering:
    """Tests for the deferred write-then-delete filtering strategy.

    When percentile filtering is enabled:
      - Prefill always writes the file.
      - Acceptance lengths are accumulated per request during decode.
      - At request_finished, if the request's avg acceptance is above
        the threshold (drafter was good), the file is deleted.
      - If avg acceptance is in the worst percentile (drafter was bad),
        the file is kept for training.
    """

    def _make_connector(self, tmp_path, capture_percentile=10.0):
        """Create connector with percentile filtering enabled."""
        from unittest.mock import MagicMock, patch
        from vllm.distributed.kv_transfer.kv_connector.online\
            .online_hidden_states_connector import (
            OnlineHiddenStatesConnector,
        )
        from vllm.distributed.kv_transfer.kv_connector.online\
            .percentile_tracker import PercentileTracker

        with patch.object(
            OnlineHiddenStatesConnector, '__init__', lambda self, *a, **k: None
        ):
            connector = OnlineHiddenStatesConnector.__new__(
                OnlineHiddenStatesConnector,
            )

        connector._block_size = 16
        connector._storage_path = str(tmp_path)
        connector._vllm_config = MagicMock()
        connector._kv_transfer_config = MagicMock()
        connector._use_compression = False
        connector._compression_level = 3
        connector._capture_percentile = capture_percentile
        connector._percentile_tracker = PercentileTracker(
            percentile=capture_percentile,
            window_size=1000,
            min_samples=10,
        )
        connector._request_filenames = {}
        connector._active_requests = {}
        connector._req_blocks = {}
        connector._prev_step_info = {}
        connector._req_acceptance_lengths = {}
        connector._total_captured = 0
        connector._total_skipped = 0
        connector._writer = None
        connector.cache_layers = []
        connector.num_hidden_states = 3

        return connector

    def _create_dummy_file(self, path):
        """Create a dummy safetensors file at the given path."""
        import safetensors.torch
        tensors = {
            "hidden_states": torch.randn(4, 3, 256),
            "token_ids": torch.arange(4, dtype=torch.long),
        }
        safetensors.torch.save_file(tensors, path)
        assert os.path.exists(path)

    def test_bad_acceptance_keeps_file(self, tmp_path):
        """Requests with poor acceptance should keep their captured file."""
        connector = self._make_connector(tmp_path, capture_percentile=10.0)

        # Seed the tracker past warmup with mostly good values
        for _ in range(20):
            connector._percentile_tracker.observe(5.0)

        # Create a captured file
        req_id = "bad-req"
        filepath = str(tmp_path / f"{req_id}.safetensors")
        self._create_dummy_file(filepath)
        connector._request_filenames[req_id] = filepath
        connector._active_requests[req_id] = "mock"
        # Very bad acceptance — should be in worst 10%
        connector._req_acceptance_lengths[req_id] = [1.0, 1.0, 1.0]

        from unittest.mock import MagicMock
        mock_req = MagicMock()
        mock_req.request_id = req_id

        freed, params = connector.request_finished(mock_req, [])

        assert os.path.exists(filepath), "File should be kept for bad request"
        assert params["hidden_states_path"] == filepath
        assert connector._total_captured == 1
        assert connector._total_skipped == 0

    def test_good_acceptance_deletes_file(self, tmp_path):
        """Requests with good acceptance should have their file deleted."""
        connector = self._make_connector(tmp_path, capture_percentile=10.0)

        # Seed the tracker past warmup with mostly low values
        for _ in range(20):
            connector._percentile_tracker.observe(1.5)

        # Create a captured file
        req_id = "good-req"
        filepath = str(tmp_path / f"{req_id}.safetensors")
        self._create_dummy_file(filepath)
        connector._request_filenames[req_id] = filepath
        connector._active_requests[req_id] = "mock"
        # Very good acceptance — should NOT be in worst 10%
        connector._req_acceptance_lengths[req_id] = [5.0, 5.0, 5.0]

        from unittest.mock import MagicMock
        mock_req = MagicMock()
        mock_req.request_id = req_id

        freed, params = connector.request_finished(mock_req, [])

        assert not os.path.exists(filepath), \
            "File should be deleted for good request"
        assert params["hidden_states_path"] is None
        assert connector._total_skipped == 1
        assert connector._total_captured == 0

    def test_no_acceptance_data_keeps_file(self, tmp_path):
        """Requests with no decode steps (prefill-only) keep their file."""
        connector = self._make_connector(tmp_path, capture_percentile=10.0)

        req_id = "prefill-only"
        filepath = str(tmp_path / f"{req_id}.safetensors")
        self._create_dummy_file(filepath)
        connector._request_filenames[req_id] = filepath
        connector._active_requests[req_id] = "mock"
        # No acceptance data — request finished after prefill only

        from unittest.mock import MagicMock
        mock_req = MagicMock()
        mock_req.request_id = req_id

        freed, params = connector.request_finished(mock_req, [])

        assert os.path.exists(filepath), \
            "File should be kept when no acceptance data"
        assert params["hidden_states_path"] == filepath

    def test_filtering_disabled_always_keeps(self, tmp_path):
        """With capture_percentile=0, files are always kept."""
        from unittest.mock import MagicMock, patch
        from vllm.distributed.kv_transfer.kv_connector.online\
            .online_hidden_states_connector import (
            OnlineHiddenStatesConnector,
        )

        with patch.object(
            OnlineHiddenStatesConnector, '__init__', lambda self, *a, **k: None
        ):
            connector = OnlineHiddenStatesConnector.__new__(
                OnlineHiddenStatesConnector,
            )

        connector._block_size = 16
        connector._storage_path = str(tmp_path)
        connector._percentile_tracker = None  # disabled
        connector._request_filenames = {}
        connector._active_requests = {}
        connector._req_blocks = {}
        connector._prev_step_info = {}
        connector._req_acceptance_lengths = {}
        connector._total_captured = 0
        connector._total_skipped = 0

        req_id = "always-keep"
        filepath = str(tmp_path / f"{req_id}.safetensors")
        self._create_dummy_file(filepath)
        connector._request_filenames[req_id] = filepath
        connector._active_requests[req_id] = "mock"

        mock_req = MagicMock()
        mock_req.request_id = req_id

        freed, params = connector.request_finished(mock_req, [])

        assert os.path.exists(filepath)
        assert params["hidden_states_path"] == filepath

    def test_compressed_file_deleted(self, tmp_path):
        """Deletion should handle .zst compressed files."""
        connector = self._make_connector(tmp_path, capture_percentile=10.0)

        for _ in range(20):
            connector._percentile_tracker.observe(1.5)

        req_id = "compressed-good"
        filepath = str(tmp_path / f"{req_id}.safetensors")
        # Create the .zst version (as the writer would)
        zst_path = filepath + ".zst"
        with open(zst_path, "wb") as f:
            f.write(b"fake compressed data")
        assert os.path.exists(zst_path)

        connector._request_filenames[req_id] = filepath
        connector._active_requests[req_id] = "mock"
        connector._req_acceptance_lengths[req_id] = [5.0, 5.0]

        from unittest.mock import MagicMock
        mock_req = MagicMock()
        mock_req.request_id = req_id

        connector.request_finished(mock_req, [])

        assert not os.path.exists(zst_path), \
            "Compressed file should be deleted for good request"


# ────────────────────────────────────────────────────────────────
# LoRA subdirectory layout tests
# ────────────────────────────────────────────────────────────────


class TestLoraSubdirectoryLayout:
    """Tests that output files are organized by LoRA adapter."""

    def _make_connector(self, tmp_path):
        """Create connector with mocked init."""
        from unittest.mock import MagicMock, patch
        from vllm.distributed.kv_transfer.kv_connector.online\
            .online_hidden_states_connector import (
            OnlineHiddenStatesConnector,
        )

        with patch.object(
            OnlineHiddenStatesConnector, '__init__', lambda self, *a, **k: None
        ):
            connector = OnlineHiddenStatesConnector.__new__(
                OnlineHiddenStatesConnector,
            )

        connector._block_size = 16
        connector._storage_path = str(tmp_path)
        connector._vllm_config = MagicMock()
        connector._kv_transfer_config = MagicMock()
        connector._use_compression = False
        connector._compression_level = 3
        connector._capture_percentile = 0.0
        connector._percentile_tracker = None
        connector._request_filenames = {}
        connector._active_requests = {}
        connector._req_blocks = {}
        connector._prev_step_info = {}
        connector._req_acceptance_lengths = {}
        connector._total_captured = 0
        connector._total_skipped = 0
        connector._writer = None
        connector.cache_layers = []
        connector.num_hidden_states = 3

        return connector

    def _make_scheduler_output(self, new_reqs):
        """Build a minimal SchedulerOutput with the given new requests."""
        from unittest.mock import MagicMock
        sched_out = MagicMock()
        sched_out.scheduled_new_reqs = new_reqs
        sched_out.scheduled_spec_decode_tokens = {}
        sched_out.num_scheduled_tokens = {
            r.req_id: 1 for r in new_reqs
        }
        # Empty cached reqs
        cached = MagicMock()
        cached.req_ids = []
        cached.num_computed_tokens = []
        cached.new_block_ids = []
        sched_out.scheduled_cached_reqs = cached
        return sched_out

    def _make_new_req(self, req_id, lora_name=None):
        """Build a minimal NewRequestData-like object."""
        from unittest.mock import MagicMock
        req = MagicMock()
        req.req_id = req_id
        req.prompt_token_ids = [1, 2, 3, 4]
        req.block_ids = ([0, 1],)
        req.num_computed_tokens = 0
        if lora_name is not None:
            lora_req = MagicMock()
            lora_req.lora_name = lora_name
            req.lora_request = lora_req
        else:
            req.lora_request = None
        return req

    def test_base_model_goes_to_base_subdir(self, tmp_path):
        """Requests without LoRA should go to base/ subdirectory."""
        connector = self._make_connector(tmp_path)
        req = self._make_new_req("req-base-1")
        sched_out = self._make_scheduler_output([req])

        meta = connector.build_connector_meta(sched_out)

        filename = connector._request_filenames["req-base-1"]
        assert "/base/" in filename
        assert filename == os.path.join(
            str(tmp_path), "base", "req-base-1.safetensors"
        )

    def test_lora_goes_to_adapter_subdir(self, tmp_path):
        """Requests with LoRA should go to <lora_name>/ subdirectory."""
        connector = self._make_connector(tmp_path)
        req = self._make_new_req("req-lora-1", lora_name="my-adapter")
        sched_out = self._make_scheduler_output([req])

        meta = connector.build_connector_meta(sched_out)

        filename = connector._request_filenames["req-lora-1"]
        assert "/my-adapter/" in filename
        assert filename == os.path.join(
            str(tmp_path), "my-adapter", "req-lora-1.safetensors"
        )

    def test_multiple_adapters_separate_dirs(self, tmp_path):
        """Different LoRA adapters should go to different subdirectories."""
        connector = self._make_connector(tmp_path)
        reqs = [
            self._make_new_req("req-1", lora_name=None),
            self._make_new_req("req-2", lora_name="adapter-a"),
            self._make_new_req("req-3", lora_name="adapter-b"),
            self._make_new_req("req-4", lora_name="adapter-a"),
        ]
        sched_out = self._make_scheduler_output(reqs)

        meta = connector.build_connector_meta(sched_out)

        assert "/base/" in connector._request_filenames["req-1"]
        assert "/adapter-a/" in connector._request_filenames["req-2"]
        assert "/adapter-b/" in connector._request_filenames["req-3"]
        assert "/adapter-a/" in connector._request_filenames["req-4"]


    def test_writer_creates_subdirectory_files(self, tmp_path):
        """Verify the writer can write to LoRA subdirectories."""
        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=False,
        )

        # Create subdirectories (as save_kv_layer would)
        base_dir = tmp_path / "base"
        lora_dir = tmp_path / "my-adapter"
        base_dir.mkdir()
        lora_dir.mkdir()

        hs = torch.randn(4, 3, 64)
        tids = torch.arange(4, dtype=torch.long)

        writer.write_async(hs, tids, str(base_dir / "req-1.safetensors"))
        writer.write_async(hs, tids, str(lora_dir / "req-2.safetensors"))
        writer.flush(timeout=5.0)

        assert (base_dir / "req-1.safetensors").exists()
        assert (lora_dir / "req-2.safetensors").exists()
        assert writer._total_writes == 2

        writer.shutdown()

    def test_writer_compressed_to_subdirectory(self, tmp_path):
        """Verify compressed writes work in subdirectories."""
        try:
            import zstandard  # noqa: F401
        except ImportError:
            pytest.skip("zstandard not installed")

        writer = HiddenStatesWriter(
            shared_storage_path=str(tmp_path),
            use_compression=True,
            compression_level=3,
        )

        subdir = tmp_path / "adapter-x"
        subdir.mkdir()

        hs = torch.randn(6, 2, 32)
        tids = torch.arange(6, dtype=torch.long)
        filename = str(subdir / "req-zst.safetensors")

        writer.write_async(hs, tids, filename)
        writer.flush(timeout=5.0)

        assert (subdir / "req-zst.safetensors.zst").exists()
        assert writer._total_writes == 1

        writer.shutdown()
