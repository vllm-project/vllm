# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF3FS KV Connector high-level components:
- TestHf3fsMockClient      : file-backed mock client I/O correctness
- TestHF3FSKVConnectorStats: metric collection, aggregation, serialisation
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnector,
    HF3FSKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient as MockHf3fsClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def hf3fs_stats():
    """Fresh HF3FSKVConnectorStats instance."""
    return HF3FSKVConnectorStats()


def _make_cuda_event():
    """Return a real CUDA event when available, otherwise a MagicMock."""
    if torch.cuda.is_available():
        return torch.cuda.Event()
    return MagicMock()


# ===========================================================================
# TestHf3fsMockClient
# ===========================================================================


class TestHf3fsMockClient:
    """Tests for hf3fs_mock_client.Hf3fsClient (file-backend mock)."""

    def test_init_creates_file(self, tmp_path):
        """Initializing the client should create the backing file."""
        path = str(tmp_path / "test_file")
        client = MockHf3fsClient(path=path, size=4096, bytes_per_page=512, entries=4)
        assert os.path.exists(path), "Backing file should be created on init"
        assert os.path.getsize(path) == 4096
        client.close()

    @pytest.mark.parametrize(
        "dtype, bytes_per_page",
        [
            (torch.float32, 512),
            (torch.float16, 256),
            (torch.bfloat16, 256),
        ],
        ids=["float32", "float16", "bfloat16"],
    )
    def test_batch_write_and_read_dtype(self, tmp_path, dtype, bytes_per_page):
        """Write a tensor of the given dtype and verify round-trip correctness."""
        path = str(tmp_path / f"rw_{dtype}")
        client = MockHf3fsClient(
            path=path, size=bytes_per_page * 8, bytes_per_page=bytes_per_page, entries=4
        )
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = bytes_per_page // elem_size
        tensor_write = torch.arange(numel, dtype=dtype)
        event = _make_cuda_event()

        results = client.batch_write([0], [tensor_write], event)
        assert results == [bytes_per_page], f"Write should succeed, got {results}"

        tensor_read = torch.zeros(numel, dtype=dtype)
        results = client.batch_read([0], [tensor_read])
        assert results == [bytes_per_page], f"Read should succeed, got {results}"
        assert torch.equal(tensor_write, tensor_read), (
            "Read tensor should match written tensor"
        )
        client.close()

    def test_batch_read_empty_file_returns_error(self, tmp_path):
        """Reading out-of-bounds offset should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "empty_read")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor_read = torch.zeros(numel, dtype=torch.float32)
        results = client.batch_read([size], [tensor_read])  # offset == size => OOB
        assert results[0] == -1, "Out-of-bounds read should return -1"
        client.close()

    def test_batch_write_out_of_bounds_returns_error(self, tmp_path):
        """Writing at an offset beyond file size should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "oob_write")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor = torch.ones(numel, dtype=torch.float32)
        event = _make_cuda_event()
        results = client.batch_write([size], [tensor], event)  # OOB offset
        assert results[0] == -1, "Out-of-bounds write should return -1"
        client.close()

    def test_multiple_tensors_rw(self, tmp_path):
        """Write multiple tensors at different offsets, then read all back."""
        bytes_per_page = 128
        n = 4
        path = str(tmp_path / "multi_rw")
        client = MockHf3fsClient(
            path=path,
            size=bytes_per_page * n * 2,
            bytes_per_page=bytes_per_page,
            entries=8,
        )
        tensors_write = [
            torch.full((bytes_per_page // 4,), float(i), dtype=torch.float32)
            for i in range(n)
        ]
        offsets = [i * bytes_per_page for i in range(n)]
        event = _make_cuda_event()

        results = client.batch_write(offsets, tensors_write, event)
        assert all(r == bytes_per_page for r in results)

        tensors_read = [
            torch.zeros(bytes_per_page // 4, dtype=torch.float32) for _ in range(n)
        ]
        results = client.batch_read(offsets, tensors_read)
        assert all(r == bytes_per_page for r in results)

        for i, (tw, tr) in enumerate(zip(tensors_write, tensors_read)):
            assert torch.allclose(tw, tr), f"Tensor {i} mismatch after round-trip"
        client.close()

    def test_flush_and_close_no_error(self, tmp_path):
        """flush() and close() should not raise exceptions."""
        path = str(tmp_path / "flush_close")
        client = MockHf3fsClient(path=path, size=1024, bytes_per_page=128, entries=4)
        client.flush()
        client.close()


# ===========================================================================
# TestHF3FSKVConnectorStats
# ===========================================================================


class TestHF3FSKVConnectorStats:
    """Tests for HF3FSKVConnectorStats metric collection and aggregation."""

    def test_initial_is_empty(self, hf3fs_stats):
        """Fresh stats object should report is_empty() == True."""
        assert hf3fs_stats.is_empty() is True

    @pytest.mark.parametrize(
        "task_type, duration_key",
        [
            ("Saved", "save_duration"),
            ("Loaded", "load_duration"),
        ],
        ids=["save", "load"],
    )
    def test_record_success_duration(self, hf3fs_stats, task_type, duration_key):
        """Recording a successful task should update duration list and total count."""
        hf3fs_stats.record_success_task_duration(task_type, 0.5)
        assert not hf3fs_stats.is_empty()
        assert len(hf3fs_stats.data[duration_key]) == 1
        assert hf3fs_stats.data[duration_key][0] == pytest.approx(0.5)
        assert hf3fs_stats.data["num_transfer_task"] == 1

    @pytest.mark.parametrize(
        "task_type, failed_key",
        [
            ("Saved", "num_failed_save"),
            ("Loaded", "num_failed_load"),
        ],
        ids=["save", "load"],
    )
    def test_record_failed_task(self, hf3fs_stats, task_type, failed_key):
        """Recording a failed task should increment the corresponding counter."""
        hf3fs_stats.record_failed_task_count(task_type)
        assert hf3fs_stats.data[failed_key] == 1
        assert hf3fs_stats.data["num_transfer_task"] == 1

    def test_aggregate_two_stats(self):
        """aggregate() should merge save/load duration lists and sum counters."""
        stats1 = HF3FSKVConnectorStats()
        stats1.record_success_task_duration("Saved", 0.1)
        stats1.record_success_task_duration("Loaded", 0.2)

        stats2 = HF3FSKVConnectorStats()
        stats2.record_success_task_duration("Saved", 0.3)
        stats2.record_failed_task_count("Loaded")

        stats1.aggregate(stats2)
        assert stats1.data["save_duration"] == pytest.approx([0.1, 0.3])
        assert stats1.data["load_duration"] == pytest.approx([0.2])
        assert stats1.data["num_failed_load"] == 1
        assert stats1.data["num_transfer_task"] == 4

    def test_reduce_with_data(self):
        """reduce() computes correct averages when data is present."""
        stats = HF3FSKVConnectorStats()
        stats.record_success_task_duration("Saved", 1.0)
        stats.record_success_task_duration("Saved", 3.0)
        result = stats.reduce()
        assert result["Num save task success"] == pytest.approx(2.0, rel=0.01)
        assert result["Num save task failed"] == pytest.approx(0.0, rel=0.01)
        assert result["Avg save duration (ms)"] == pytest.approx(2000.0, rel=0.01)
        # Reduced values must be plain Python scalars so CLI logging renders
        # them without numpy reprs (eg np.float64(...)).
        assert all(not isinstance(v, np.generic) for v in result.values())

    def test_clone_and_reset(self, hf3fs_stats):
        """clone_and_reset() returns a copy with data and resets the original."""
        hf3fs_stats.record_success_task_duration("Saved", 0.7)
        hf3fs_stats.record_success_task_duration("Loaded", 0.4)

        clone = hf3fs_stats.clone_and_reset()
        assert clone.data["num_transfer_task"] == 2
        assert hf3fs_stats.is_empty()


# ===========================================================================
# TestHF3FSKVConnectorMetadataServerUrl
# ===========================================================================


class TestHF3FSKVConnectorMetadataServerUrl:
    """Tests that HF3FSKVConnector passes hf3fs_metadata_server_url
    to Hf3fsMetadataClient.
    """

    def _make_mock_config(self, extra_config: dict | None = None):
        mock_vllm_config = MagicMock()
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.model_config.use_mla = False
        mock_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="HF3FSKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config or {},
        )
        return mock_vllm_config

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector.Hf3fsMetadataClient"
    )
    def test_scheduler_custom_metadata_server_url(self, mock_client_cls):
        custom_url = "http://custom-scheduler-meta:18000"
        vllm_config = self._make_mock_config({"hf3fs_metadata_server_url": custom_url})
        connector = HF3FSKVConnector(
            vllm_config, KVConnectorRole.SCHEDULER, MagicMock()
        )
        assert connector._metadata_server_url == custom_url
        mock_client_cls.assert_called_once_with(custom_url)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector.Hf3fsMetadataClient"
    )
    def test_scheduler_default_metadata_server_url(self, mock_client_cls):
        vllm_config = self._make_mock_config()
        connector = HF3FSKVConnector(
            vllm_config, KVConnectorRole.SCHEDULER, MagicMock()
        )
        assert connector._metadata_server_url == "http://localhost:18000"
        mock_client_cls.assert_called_once_with("http://localhost:18000")

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector.Hf3fsMetadataClient"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector.Hf3fsClient"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("os.makedirs")
    def test_worker_custom_metadata_server_url(
        self, mock_makedirs, mock_rank, mock_hf3fs_client, mock_meta_client_cls
    ):
        custom_url = "http://custom-worker-meta:18000"
        vllm_config = self._make_mock_config(
            {
                "hf3fs_metadata_server_url": custom_url,
                "hf3fs_storage_path": "./dummy_storage",
            }
        )
        connector = HF3FSKVConnector(vllm_config, KVConnectorRole.WORKER, MagicMock())
        connector._bytes_per_page = 1024
        connector._setup_storage_clients()
        mock_meta_client_cls.assert_called_once_with(custom_url)
