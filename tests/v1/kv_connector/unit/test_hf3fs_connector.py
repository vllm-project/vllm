# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for HF3FS KV Connector high-level components:
  - TestHf3fsMockClient      : file-backed mock client I/O correctness
  - TestHF3FSKVConnectorStats: metric collection, aggregation, serialisation
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnector,
    HF3FSKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.common import (
    HF3FSConnectorMetadata,
    HF3FSRequestMetadata,
    LoadBlockInfo,
    RequestSchedulingState,
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


def _make_hash_connector() -> HF3FSKVConnector:
    connector = object.__new__(HF3FSKVConnector)
    connector._block_size = 4
    return connector


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

    def test_clone_and_reset(self, hf3fs_stats):
        """clone_and_reset() returns a copy with data and resets the original."""
        hf3fs_stats.record_success_task_duration("Saved", 0.7)
        hf3fs_stats.record_success_task_duration("Loaded", 0.4)

        clone = hf3fs_stats.clone_and_reset()
        assert clone.data["num_transfer_task"] == 2
        assert hf3fs_stats.is_empty()


# ===========================================================================
# TestHF3FSCacheSaltIsolation
# ===========================================================================


def _request_metadata(
    cache_salt: str, load_op: LoadBlockInfo | None = None
) -> HF3FSRequestMetadata:
    state = RequestSchedulingState(
        request_id=cache_salt,
        token_ids=list(range(8)),
        allocated_block_ids=[10, 11],
        cache_salt=cache_salt,
    )
    request_metadata = HF3FSRequestMetadata.from_scheduling_state(
        state, 4, load_op=load_op
    )
    assert request_metadata is not None
    return request_metadata


def _worker_hashes(
    request_metadata: HF3FSRequestMetadata, *, load: bool = False
) -> list[str]:
    metadata = HF3FSConnectorMetadata()
    metadata.add_request(request_metadata)

    connector = _make_hash_connector()
    connector._max_device_buffer_count = 8
    connector._connector_metadata = metadata
    connector._async_manager = MagicMock()

    if load:
        connector.start_load_kv(None)
        submit = connector._async_manager.submit_load_operation
    else:
        connector.wait_for_save()
        submit = connector._async_manager.submit_save_operation

    submit.assert_called_once()
    assert metadata.requests[0].cache_salt == request_metadata.cache_salt
    return submit.call_args.args[2]


class TestHF3FSCacheSaltIsolation:
    """Verify cache_salt isolation through scheduler and worker paths."""

    def test_scheduler_lookup_and_hash_chain_are_isolated_by_salt(self):
        connector = _make_hash_connector()
        prompt_token_ids = list(range(9))
        tenant_a_keys = connector._generate_block_hashes(
            prompt_token_ids[:8], 0, cache_salt="tenant-A"
        )
        tenant_b_keys = connector._generate_block_hashes(
            prompt_token_ids[:8], 0, cache_salt="tenant-B"
        )
        assert connector._generate_block_hashes(list(range(4)), 0)[0] == (
            "44aad182b5a8a40926162f81a84f091f"
        )
        assert all(a != b for a, b in zip(tenant_a_keys, tenant_b_keys))

        stored_keys = set(tenant_a_keys)
        connector._scheduling_states = {}
        connector._metadata_client = MagicMock()
        connector._metadata_client.batch_key_exists.side_effect = lambda hashes: [
            block_hash in stored_keys for block_hash in hashes
        ]

        tenant_a = SimpleNamespace(
            request_id="request-a",
            prompt_token_ids=prompt_token_ids,
            cache_salt="tenant-A",
        )
        tenant_b = SimpleNamespace(
            request_id="request-b",
            prompt_token_ids=prompt_token_ids,
            cache_salt="tenant-B",
        )

        assert connector.get_num_new_matched_tokens(tenant_a, 0) == (8, True)
        assert connector.get_num_new_matched_tokens(tenant_b, 0) == (0, False)

    def test_salt_survives_worker_save_and_load_handoff(self):
        tenant_a_save = _worker_hashes(_request_metadata("tenant-A"))
        expected_hashes = _make_hash_connector()._generate_block_hashes(
            list(range(8)), 0, cache_salt="tenant-A"
        )
        assert tenant_a_save == expected_hashes

        load_op = LoadBlockInfo(
            num_computed_blocks=1,
            num_blocks_to_load=1,
            need_fetch_block_ids=[11],
        )
        tenant_a_load = _worker_hashes(
            _request_metadata("tenant-A", load_op), load=True
        )

        assert tenant_a_load == expected_hashes[1:]
