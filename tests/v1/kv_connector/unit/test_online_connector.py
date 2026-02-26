# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OnlineHiddenStatesConnector."""
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.online.hidden_states_writer import HiddenStatesWriter
from vllm.distributed.kv_transfer.kv_connector.online.online_hidden_states_connector import OnlineHiddenStatesConnector
from vllm.distributed.kv_transfer.kv_connector.online.percentile_tracker import PercentileTracker


@pytest.fixture
def writer(tmp_path):
    w = HiddenStatesWriter(str(tmp_path), use_compression=False)
    yield w
    w.shutdown()


def _make_connector(tmp_path, capture_percentile=0.0):
    with patch.object(OnlineHiddenStatesConnector, "__init__",
                      lambda self, *a, **k: None):
        c = OnlineHiddenStatesConnector.__new__(OnlineHiddenStatesConnector)
    c._block_size, c._storage_path = 16, str(tmp_path)
    c._vllm_config = c._kv_transfer_config = MagicMock()
    c._active_requests, c._req_blocks, c._request_filenames = {}, {}, {}
    c.cache_layers, c.num_hidden_states = [], 3
    c.use_compression, c.compression_level = False, 3
    c.percentile_tracker, c.writer = None, None
    c.decode_filenames, c.request_filenames = {}, {}
    c.prev_step_info, c.acceptance_lengths = {}, {}
    c.total_captured = c.total_skipped = 0
    if capture_percentile > 0:
        c.percentile_tracker = PercentileTracker(
            percentile=capture_percentile, window_size=1000, min_samples=5)
    return c


@pytest.fixture
def connector(tmp_path):
    return _make_connector(tmp_path)


def _mock_request(req_id):
    mock = MagicMock()
    mock.request_id = req_id
    return mock


def _setup_request(connector, tmp_path, req_id, lora=None):
    subdir = lora or "base"
    prefill_path = str(tmp_path / subdir / f"{req_id}.safetensors")
    decode_path = str(tmp_path / subdir / f"{req_id}_decode.safetensors")
    connector.request_filenames[req_id] = (prefill_path, decode_path)
    connector._active_requests[req_id] = "mock"
    return prefill_path, decode_path


def _mock_scheduler_output(reqs):
    output = MagicMock()
    output.scheduled_new_reqs = reqs
    output.scheduled_spec_decode_tokens = {}
    output.num_scheduled_tokens = {r.req_id: 1 for r in reqs}
    cached = MagicMock()
    cached.req_ids, cached.num_computed_tokens, cached.new_block_ids = [], [], []
    output.scheduled_cached_reqs = cached
    return output


def _mock_new_request(req_id, lora=None):
    req = MagicMock()
    req.req_id = req_id
    req.prompt_token_ids = [1, 2, 3]
    req.block_ids = ([0, 1],)
    req.num_computed_tokens = 0
    req.lora_request = None
    if lora:
        req.lora_request = MagicMock()
        req.lora_request.lora_name = lora
    return req


class TestPercentileTracker:
    def test_warmup_captures_everything(self):
        t = PercentileTracker(percentile=10.0, window_size=100, min_samples=50)
        for _ in range(49):
            assert t.should_capture(100.0) is True
            t.observe(100.0)

    def test_filtering_after_warmup(self):
        t = PercentileTracker(percentile=10.0, window_size=1000, min_samples=100)
        for i in range(200):
            t.observe(float(i + 1))
        assert t.should_capture(1.0) is True
        assert t.should_capture(100.0) is False

    def test_observe_and_check_atomic(self):
        t = PercentileTracker(percentile=50.0, window_size=100, min_samples=10)
        for i in range(20):
            t.observe(float(i))
        n = t.get_stats()["num_samples"]
        t.observe_and_check(5.0)
        assert t.get_stats()["num_samples"] == n + 1

    def test_sliding_window_eviction(self):
        t = PercentileTracker(percentile=50.0, window_size=10, min_samples=5)
        for _ in range(10):
            t.observe(1.0)
        for _ in range(10):
            t.observe(100.0)
        assert t.get_stats()["min_acceptance"] == 100.0

    def test_uniform_distribution_filters(self):
        """When all values are identical, should_capture respects percentile."""
        t = PercentileTracker(percentile=5.0, window_size=1000, min_samples=10)
        for _ in range(200):
            t.observe(2.0)
        # All values are 2.0, threshold is 2.0, fraction_at_threshold ~ 0.05
        # Run many trials to check statistical behavior
        import random
        random.seed(42)
        captured = sum(t.should_capture(2.0) for _ in range(1000))
        # With p5, expect ~5% capture rate (50 out of 1000)
        assert 10 < captured < 120, f"Expected ~50 captures, got {captured}"

    def test_get_stats(self):
        assert PercentileTracker().get_stats()["num_samples"] == 0
        t = PercentileTracker(percentile=25.0, min_samples=5)
        for v in range(1, 11):
            t.observe(float(v))
        assert t.get_stats()["mean_acceptance"] == pytest.approx(5.5)


class TestHiddenStatesWriter:
    def test_write_and_read(self, writer, tmp_path):
        hs = torch.randn(10, 4, 256)
        tids = torch.arange(10, dtype=torch.long)
        f = str(tmp_path / "test.safetensors")
        writer.write_async(hs, tids, f)
        writer.flush(timeout=5.0)
        import safetensors.torch
        loaded = safetensors.torch.load_file(f)
        assert loaded["hidden_states"].shape == hs.shape
        assert torch.equal(loaded["token_ids"], tids)

    def test_compression(self, tmp_path):
        pytest.importorskip("zstandard")
        w = HiddenStatesWriter(str(tmp_path), use_compression=True)
        f = str(tmp_path / "c.safetensors")
        w.write_async(torch.randn(20, 4, 128), torch.arange(20, dtype=torch.long), f)
        w.flush(timeout=5.0)
        assert os.path.exists(f + ".zst")
        w.shutdown()

    def test_data_integrity(self, writer, tmp_path):
        hs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        tids = torch.tensor([100, 200], dtype=torch.long)
        f = str(tmp_path / "int.safetensors")
        writer.write_async(hs, tids, f)
        writer.flush(timeout=5.0)
        import safetensors.torch
        assert torch.equal(safetensors.torch.load_file(f)["hidden_states"], hs)

    def test_creates_subdirs(self, writer, tmp_path):
        hs, tids = torch.randn(4, 3, 64), torch.arange(4, dtype=torch.long)
        writer.write_async(hs, tids, str(tmp_path / "base" / "r1.safetensors"))
        writer.write_async(hs, tids, str(tmp_path / "lora" / "r2.safetensors"))
        writer.flush(timeout=5.0)
        assert (tmp_path / "base" / "r1.safetensors").exists()
        assert (tmp_path / "lora" / "r2.safetensors").exists()


class TestWriterAccumulate:
    def test_accumulate_and_flush(self, writer, tmp_path):
        for i in range(3):
            writer.accumulate_async("r1", torch.randn(1, 3, 256),
                                    torch.tensor([i], dtype=torch.long))
        f = str(tmp_path / "base" / "r1_decode.safetensors")
        writer.flush_request("r1", f)
        writer.flush(timeout=5.0)
        import safetensors.torch
        assert safetensors.torch.load_file(f)["hidden_states"].shape == (3, 3, 256)

    def test_flush_empty_is_noop(self, writer, tmp_path):
        f = str(tmp_path / "empty.safetensors")
        writer.flush_request("nonexistent", f)
        writer.flush(timeout=5.0)
        assert not os.path.exists(f)

    def test_discard_cleans_buffers(self, writer):
        writer.accumulate_async("r1", torch.randn(1, 3, 64),
                                torch.tensor([1], dtype=torch.long))
        writer.discard_request("r1")
        assert "r1" not in writer.decode_hidden_states

    def test_shutdown_clears_and_joins(self, tmp_path):
        w = HiddenStatesWriter(str(tmp_path), use_compression=False)
        w.accumulate_async("r1", torch.randn(1, 3, 64),
                           torch.tensor([1], dtype=torch.long))
        w.shutdown()
        assert len(w.decode_hidden_states) == 0
        assert not w.writer_thread.is_alive()


class TestGetFinished:
    def test_flushes_decode(self, connector, tmp_path):
        w = connector.get_writer()
        for i in range(3):
            w.accumulate_async("r1", torch.randn(1, 3, 256),
                               torch.tensor([i], dtype=torch.long))
        connector.decode_filenames["r1"] = str(tmp_path / "base" / "r1_decode.safetensors")
        connector.get_finished({"r1"})
        w.flush(timeout=5.0)
        assert os.path.exists(str(tmp_path / "base" / "r1_decode.safetensors"))
        w.shutdown()

    def test_no_decode_no_file(self, connector, tmp_path):
        connector.decode_filenames["r1"] = str(tmp_path / "base" / "r1_decode.safetensors")
        connector.get_finished({"r1"})
        assert not os.path.exists(str(tmp_path / "base" / "r1_decode.safetensors"))

    def test_cleans_state(self, connector):
        connector.decode_filenames.update({"r1": "/tmp/x", "r2": "/tmp/y"})
        connector.get_finished({"r1", "r2"})
        assert len(connector.decode_filenames) == 0


class TestRequestFinished:
    def test_returns_both_paths(self, connector, tmp_path):
        prefill_path, decode_path = _setup_request(connector, tmp_path, "r1")
        _, params = connector.request_finished(_mock_request("r1"), [])
        assert params["hidden_states_prefill"] == prefill_path
        assert params["hidden_states_decode"] == decode_path

    def test_cleans_all_state(self, connector, tmp_path):
        _setup_request(connector, tmp_path, "r1")
        connector.prev_step_info["r1"] = (10, 2)
        connector.acceptance_lengths["r1"] = [1.5]
        connector.request_finished(_mock_request("r1"), [])
        for d in (connector.request_filenames, connector._active_requests,
                  connector.prev_step_info, connector.acceptance_lengths):
            assert "r1" not in d

    def test_missing_req(self, connector):
        assert connector.request_finished(_mock_request("nope"), [])[1]["hidden_states_prefill"] is None

    def test_isolates_requests(self, connector, tmp_path):
        _setup_request(connector, tmp_path, "a")
        _setup_request(connector, tmp_path, "b")
        connector.request_finished(_mock_request("a"), [])
        assert "a" not in connector.request_filenames and "b" in connector.request_filenames

    def test_shutdown_flushes(self, connector, tmp_path):
        w = connector.get_writer()
        w.write_async(torch.randn(4, 3, 64), torch.arange(4, dtype=torch.long),
                      str(tmp_path / "base" / "t.safetensors"))
        connector.shutdown()
        assert os.path.exists(str(tmp_path / "base" / "t.safetensors"))


class TestPercentileFiltering:
    def test_bad_acceptance_keeps(self, tmp_path):
        c = _make_connector(tmp_path, capture_percentile=10.0)
        for _ in range(20):
            c.percentile_tracker.observe(5.0)
        prefill_path, _ = _setup_request(c, tmp_path, "bad")
        c.acceptance_lengths["bad"] = [1.0, 1.0]
        assert c.request_finished(_mock_request("bad"), [])[1]["hidden_states_prefill"] == prefill_path

    def test_good_acceptance_deletes(self, tmp_path):
        c = _make_connector(tmp_path, capture_percentile=10.0)
        for _ in range(20):
            c.percentile_tracker.observe(1.5)
        prefill_path, decode_path = _setup_request(c, tmp_path, "good")
        os.makedirs(os.path.dirname(prefill_path), exist_ok=True)
        for path in (prefill_path, decode_path):
            open(path, "wb").write(b"x")
        c.acceptance_lengths["good"] = [5.0, 5.0]
        assert c.request_finished(_mock_request("good"), [])[1]["hidden_states_prefill"] is None
        assert not os.path.exists(prefill_path) and not os.path.exists(decode_path)

    def test_no_acceptance_keeps(self, tmp_path):
        c = _make_connector(tmp_path, capture_percentile=10.0)
        _setup_request(c, tmp_path, "p")
        assert c.request_finished(_mock_request("p"), [])[1]["hidden_states_prefill"] is not None

    def test_disabled_always_keeps(self, connector, tmp_path):
        _setup_request(connector, tmp_path, "r")
        assert connector.request_finished(_mock_request("r"), [])[1]["hidden_states_prefill"] is not None


class TestLoraLayout:
    def test_base_subdir(self, connector):
        connector.build_connector_meta(_mock_scheduler_output([_mock_new_request("r1")]))
        prefill_path, decode_path = connector.request_filenames["r1"]
        assert "/base/" in prefill_path and prefill_path.endswith("r1.safetensors")
        assert decode_path.endswith("r1_decode.safetensors")

    def test_lora_subdir(self, connector):
        connector.build_connector_meta(_mock_scheduler_output([_mock_new_request("r1", "my-lora")]))
        assert "/my-lora/" in connector.request_filenames["r1"][0]

    def test_multiple_adapters(self, connector):
        connector.build_connector_meta(
            _mock_scheduler_output([_mock_new_request("r1"), _mock_new_request("r2", "a"), _mock_new_request("r3", "b")]))
        assert "/base/" in connector.request_filenames["r1"][0]
        assert "/a/" in connector.request_filenames["r2"][0]
        assert "/b/" in connector.request_filenames["r3"][0]
