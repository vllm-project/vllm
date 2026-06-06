# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CapturingEagleProposer and capture_hidden_states."""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.online.online_hidden_states_connector import OnlineHiddenStatesConnector


def _make_connector(tmp_path):
    with patch.object(OnlineHiddenStatesConnector, "__init__",
                      lambda self, *a, **k: None):
        connector = OnlineHiddenStatesConnector.__new__(
            OnlineHiddenStatesConnector)
    connector._block_size, connector._storage_path = 16, str(tmp_path)
    connector._vllm_config = connector._kv_transfer_config = MagicMock()
    connector._active_requests, connector._req_blocks = {}, {}
    connector._request_filenames = {}
    connector.cache_layers, connector.num_hidden_states = [], 3
    connector.use_compression, connector.compression_level = False, 3
    connector.percentile_tracker = None
    connector.decode_filenames, connector.request_filenames = {}, {}
    connector.prev_step_info, connector.acceptance_lengths = {}, {}
    connector.total_captured = connector.total_skipped = 0
    connector.writer = None
    return connector


def _flush_and_shutdown(connector):
    """Flush accumulated buffers via get_finished and shut down writer."""
    req_ids = set(connector.decode_filenames.keys())
    connector.get_finished(req_ids)
    writer = connector.writer
    if writer is not None:
        writer.flush(timeout=5.0)
        writer.shutdown()


class TestCaptureHiddenStates:
    def test_splits_per_request(self, tmp_path):
        connector = _make_connector(tmp_path)
        hidden_states = torch.randn(40, 4096)
        token_ids = torch.arange(40, dtype=torch.long)
        query_start_loc = torch.tensor([0, 10, 25, 40], dtype=torch.int32)

        connector.capture_hidden_states(
            hidden_states, token_ids, query_start_loc,
            req_ids=["r1", "r2", "r3"],
            lora_mapping=np.array([0, 0, 0]),
            lora_lookup={})

        _flush_and_shutdown(connector)

        import safetensors.torch
        for req_id, expected_len in [("r1", 10), ("r2", 15), ("r3", 15)]:
            filepath = str(tmp_path / "base" / f"{req_id}.safetensors")
            assert os.path.exists(filepath), f"Missing {filepath}"
            loaded = safetensors.torch.load_file(filepath)
            assert loaded["hidden_states"].shape[0] == expected_len
            assert loaded["token_ids"].shape[0] == expected_len

    def test_accumulates_across_steps(self, tmp_path):
        connector = _make_connector(tmp_path)
        mapping = np.array([0])
        # Step 1: 4 tokens
        connector.capture_hidden_states(
            torch.randn(4, 256), torch.tensor([1, 2, 3, 4]),
            torch.tensor([0, 4], dtype=torch.int32),
            req_ids=["r1"], lora_mapping=mapping, lora_lookup={})
        # Step 2: 4 more tokens
        connector.capture_hidden_states(
            torch.randn(4, 256), torch.tensor([5, 6, 7, 8]),
            torch.tensor([0, 4], dtype=torch.int32),
            req_ids=["r1"], lora_mapping=mapping, lora_lookup={})

        _flush_and_shutdown(connector)

        import safetensors.torch
        loaded = safetensors.torch.load_file(
            str(tmp_path / "base" / "r1.safetensors"))
        assert loaded["hidden_states"].shape[0] == 8
        assert loaded["token_ids"].shape[0] == 8
        assert torch.equal(loaded["token_ids"],
                           torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_lora_subdirs(self, tmp_path):
        connector = _make_connector(tmp_path)
        mock_lora = MagicMock()
        mock_lora.lora_name = "my-adapter"

        connector.capture_hidden_states(
            torch.randn(30, 256), torch.arange(30, dtype=torch.long),
            torch.tensor([0, 10, 20, 30], dtype=torch.int32),
            req_ids=["r1", "r2", "r3"],
            lora_mapping=np.array([0, 5, 5]),
            lora_lookup={5: mock_lora})

        _flush_and_shutdown(connector)

        assert os.path.exists(str(tmp_path / "base" / "r1.safetensors"))
        assert os.path.exists(
            str(tmp_path / "my-adapter" / "r2.safetensors"))
        assert os.path.exists(
            str(tmp_path / "my-adapter" / "r3.safetensors"))

    def test_skips_empty_requests(self, tmp_path):
        connector = _make_connector(tmp_path)
        connector.capture_hidden_states(
            torch.randn(10, 256), torch.arange(10, dtype=torch.long),
            torch.tensor([0, 0, 10], dtype=torch.int32),
            req_ids=["r1", "r2"],
            lora_mapping=np.array([0, 0]),
            lora_lookup={})

        _flush_and_shutdown(connector)

        assert not os.path.exists(str(tmp_path / "base" / "r1.safetensors"))
        assert os.path.exists(str(tmp_path / "base" / "r2.safetensors"))

    def test_data_integrity(self, tmp_path):
        connector = _make_connector(tmp_path)
        hidden_states = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        token_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)

        connector.capture_hidden_states(
            hidden_states, token_ids,
            torch.tensor([0, 2, 4], dtype=torch.int32),
            req_ids=["r1", "r2"],
            lora_mapping=np.array([0, 0]),
            lora_lookup={})

        _flush_and_shutdown(connector)

        import safetensors.torch
        r1 = safetensors.torch.load_file(
            str(tmp_path / "base" / "r1.safetensors"))
        assert torch.equal(r1["hidden_states"], hidden_states[:2])
        assert torch.equal(r1["token_ids"], token_ids[:2])

        r2 = safetensors.torch.load_file(
            str(tmp_path / "base" / "r2.safetensors"))
        assert torch.equal(r2["hidden_states"], hidden_states[2:4])
        assert torch.equal(r2["token_ids"], token_ids[2:4])

    def test_no_files_before_flush(self, tmp_path):
        """Data is only written to disk after get_finished flushes."""
        connector = _make_connector(tmp_path)
        connector.capture_hidden_states(
            torch.randn(10, 256), torch.arange(10, dtype=torch.long),
            torch.tensor([0, 10], dtype=torch.int32),
            req_ids=["r1"],
            lora_mapping=np.array([0]),
            lora_lookup={})

        # Before flush: no files on disk
        assert not os.path.exists(str(tmp_path / "base" / "r1.safetensors"))

        _flush_and_shutdown(connector)

        # After flush: file exists
        assert os.path.exists(str(tmp_path / "base" / "r1.safetensors"))

    def test_discard_drops_accumulated(self, tmp_path):
        """Discarding a request drops accumulated buffers without writing."""
        connector = _make_connector(tmp_path)
        connector.capture_hidden_states(
            torch.randn(10, 256), torch.arange(10, dtype=torch.long),
            torch.tensor([0, 10], dtype=torch.int32),
            req_ids=["r1"],
            lora_mapping=np.array([0]),
            lora_lookup={})

        # Manually discard instead of flushing
        writer = connector.get_writer()
        writer.discard_request("r1")
        connector.decode_filenames.pop("r1", None)
        writer.flush(timeout=5.0)
        writer.shutdown()

        assert not os.path.exists(str(tmp_path / "base" / "r1.safetensors"))
