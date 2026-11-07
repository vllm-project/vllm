# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF3FS KV Connector high-level components:
- TestHf3fsMockClient      : file-backed mock client I/O correctness
- TestHF3FSKVConnectorStats: metric collection, aggregation, serialisation
"""

import os
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    DEFAULT_MAX_IO_ENTRIES,
    AsyncOperationManager,
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


@pytest.mark.cpu_test
class TestHF3FSWriterCompletion:
    @pytest.mark.parametrize(
        "results, expected_count, bytes_per_page, succeeds",
        [
            ([16, 16], 2, 16, True),
            ([], 1, 16, False),
            ([16], 2, 16, False),
            ([16, 16], 1, 16, False),
            (None, 1, 16, False),
            ([True], 1, 1, False),
            ([15], 1, 16, False),
            (RuntimeError("writer failed"), 1, 16, False),
        ],
    )
    def test_only_exact_full_integer_results_authorize_publish(
        self, results, expected_count: int, bytes_per_page: int, succeeds: bool
    ):
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._bytes_per_page = bytes_per_page
        writer: Future = Future()
        if isinstance(results, Exception):
            writer.set_exception(results)
        else:
            writer.set_result(results)
        assert manager._drain_writers([(writer, expected_count)]) is succeeds

    def test_cancelled_writer_does_not_block_drain(self):
        """A cancelled queued writer fails without waiting for executor pickup."""
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._bytes_per_page = 16
        writer: Future = Future()
        assert writer.cancel()
        outcome: Future[bool] = Future()

        def drain():
            outcome.set_result(manager._drain_writers([(writer, 1)]))

        worker = threading.Thread(target=drain, daemon=True)
        worker.start()
        try:
            worker.join(timeout=5)
            assert not worker.is_alive()
            assert outcome.result(timeout=1) is False
        finally:
            if worker.is_alive():
                writer.set_running_or_notify_cancel()
                worker.join(timeout=5)

    @pytest.mark.parametrize(
        "allocation_results",
        [
            {},
            [],
            [("wrong-hash", 0)],
            [("hash-0", True)],
            [("hash-0", "0")],
            [("hash-0", -1)],
            [("hash-0", 0), ("extra", 1)],
        ],
    )
    def test_invalid_allocation_reply_fails_before_gather(self, allocation_results):
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._rank, manager._bytes_per_page = 0, 16
        manager.hf3fs_stats = MagicMock()
        manager._save_buffer_allocator = MagicMock()
        manager._io_executor = MagicMock()
        manager._connector = MagicMock()
        allocate = manager._connector._metadata_client.allocate_pages_for_keys
        allocate.return_value = allocation_results
        save: Future[bool] = Future()

        manager._handle_save_task(("request", [0], ["hash-0"], save, MagicMock()))

        assert save.result() is False
        manager._save_buffer_allocator.alloc_buffer.assert_not_called()
        manager._connector._gather_or_scatter_kv_caches.assert_not_called()
        manager._io_executor.submit.assert_not_called()
        manager._connector._metadata_client.confirm_write_for_keys.assert_not_called()

    def test_mismatched_block_and_key_counts_fail_before_allocation(self):
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._rank = 0
        manager.hf3fs_stats = MagicMock()
        manager._save_buffer_allocator = MagicMock()
        manager._connector = MagicMock()
        save: Future[bool] = Future()

        manager._handle_save_task(("request", [0, 1], ["hash-0"], save, MagicMock()))

        assert save.result() is False
        manager._connector._metadata_client.allocate_pages_for_keys.assert_not_called()
        manager._save_buffer_allocator.alloc_buffer.assert_not_called()

    def test_duplicate_allocation_pages_fail_before_gather(self):
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._rank, manager._bytes_per_page = 0, 16
        manager.hf3fs_stats = MagicMock()
        manager._save_buffer_allocator = MagicMock()
        manager._connector = MagicMock()
        allocate = manager._connector._metadata_client.allocate_pages_for_keys
        allocate.return_value = [("hash-0", 7), ("hash-1", 7)]
        save: Future[bool] = Future()

        manager._handle_save_task(
            ("request", [0, 1], ["hash-0", "hash-1"], save, MagicMock())
        )

        assert save.result() is False
        manager._save_buffer_allocator.alloc_buffer.assert_not_called()
        manager._connector._gather_or_scatter_kv_caches.assert_not_called()
        manager._connector._metadata_client.confirm_write_for_keys.assert_not_called()

    @pytest.mark.parametrize("writer_results", [[], [16]])
    def test_incomplete_writer_result_is_not_published(self, writer_results):
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._rank, manager._bytes_per_page = 0, 16
        manager._save_stream = MagicMock()
        manager._save_buffer_allocator = MagicMock()
        manager._save_buffer_allocator.alloc_buffer.return_value = [object(), object()]
        manager._io_executor = MagicMock()
        manager.hf3fs_stats = MagicMock()
        manager._connector = MagicMock()
        manager._connector._metadata_client.allocate_pages_for_keys.return_value = [
            ("hash-0", 0),
            ("hash-1", 1),
        ]
        manager._connector._clients = [MagicMock()]
        manager._connector._ac.next.return_value = 0
        writer: Future[list[int]] = Future()
        writer.set_result(writer_results)
        manager._io_executor.submit.return_value = writer
        save: Future[bool] = Future()

        with (
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "Event", return_value=MagicMock()),
        ):
            manager._handle_save_task(
                (
                    "request",
                    [0, 1],
                    ["hash-0", "hash-1"],
                    save,
                    MagicMock(),
                )
            )

        assert save.result() is False
        manager._connector._metadata_client.confirm_write_for_keys.assert_not_called()

    def test_save_failure_drains_later_writer_before_cleanup(self):
        block_count = DEFAULT_MAX_IO_ENTRIES + 1
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._rank, manager._bytes_per_page = 0, 16
        manager._save_stream = MagicMock()
        manager._save_buffer_allocator = MagicMock()
        buffers = [object()] * block_count
        manager._save_buffer_allocator.alloc_buffer.return_value = buffers
        manager._io_executor = MagicMock()
        manager.hf3fs_stats = MagicMock()
        manager._connector = MagicMock()
        manager._connector._metadata_client.allocate_pages_for_keys.return_value = [
            (f"hash-{i}", i) for i in range(block_count)
        ]
        manager._connector._clients = [MagicMock(), MagicMock()]
        manager._connector._ac.next.side_effect = [0, 1]
        block_ids = list(range(block_count))
        save: Future[bool] = Future()
        task = (
            "request",
            block_ids,
            [f"hash-{i}" for i in block_ids],
            save,
            MagicMock(),
        )
        first_writer: Future[list[int]] = Future()
        second_writer: Future[list[int]] = Future()
        first_writer.set_result([15] * DEFAULT_MAX_IO_ENTRIES)
        submitted_second = threading.Event()

        def submit_writer(*_args):
            if manager._io_executor.submit.call_count == 1:
                return first_writer
            submitted_second.set()
            return second_writer

        manager._io_executor.submit.side_effect = submit_writer
        with (
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "Event", return_value=MagicMock()),
        ):
            worker = threading.Thread(target=manager._handle_save_task, args=(task,))
            worker.start()
            try:
                assert submitted_second.wait(timeout=5)
                worker.join(timeout=0.05)
                manager._save_buffer_allocator.free_buffer.assert_not_called()
            finally:
                second_writer.set_result([16])
                worker.join(timeout=5)

        assert not worker.is_alive()
        assert save.result() is False
        manager._save_buffer_allocator.free_buffer.assert_called_once_with(buffers)
        manager._connector._metadata_client.confirm_write_for_keys.assert_not_called()

    def test_submission_failure_drains_earlier_writer_before_buffer_reuse(self):
        manager = AsyncOperationManager.__new__(AsyncOperationManager)
        manager._bytes_per_page = 16
        manager._io_executor = MagicMock()
        manager._connector = MagicMock()
        manager._connector._clients = [MagicMock(), MagicMock()]
        manager._connector._ac.next.side_effect = [0, 1]
        first_writer: Future[list[int]] = Future()
        submitted_second = threading.Event()

        def submit_writer(*_args):
            if manager._io_executor.submit.call_count == 1:
                return first_writer
            submitted_second.set()
            raise RuntimeError("executor rejected writer")

        manager._io_executor.submit.side_effect = submit_writer
        released = MagicMock()

        def submit_then_free():
            with pytest.raises(RuntimeError, match="executor rejected"):
                manager._submit_and_drain_writers(
                    list(range(DEFAULT_MAX_IO_ENTRIES + 1)),
                    [MagicMock()] * (DEFAULT_MAX_IO_ENTRIES + 1),
                    MagicMock(),
                )
            released()

        worker = threading.Thread(target=submit_then_free)
        worker.start()
        try:
            assert submitted_second.wait(timeout=5)
            worker.join(timeout=0.05)
            released.assert_not_called()
        finally:
            first_writer.set_result([16] * DEFAULT_MAX_IO_ENTRIES)
            worker.join(timeout=5)

        assert not worker.is_alive()
        released.assert_called_once()
