# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.distributed.ec_transfer.ec_connector.tiering.executor import (
    ECFileIOExecutor,
)


def test_executor_round_trip_reports_one_completion_per_job(
    tmp_path: Path,
) -> None:
    executor = ECFileIOExecutor(n_read_threads=1, n_write_threads=1)
    entry_path = tmp_path / "entry.ec"
    temporary_path = tmp_path / ".entry.ec.write.tmp"
    destination = bytearray(8)
    try:
        write_id = executor.submit_write(
            "write-key",
            entry_path,
            temporary_path,
            [memoryview(b"abcd"), memoryview(b"efXY")],
            8,
        )
        assert write_id is not None
        write_stats = executor.stats()
        assert write_stats.pending_write_jobs == 1
        assert write_stats.pending_write_bytes == 8
        executor.drain()

        write_completions = executor.poll_completed()
        assert len(write_completions) == 1
        assert write_completions[0].job_id == write_id
        assert write_completions[0].success
        assert write_completions[0].operation == "write"
        assert write_completions[0].file_size_bytes == 8
        assert write_completions[0].submit_time > 0
        assert write_completions[0].queue_time >= 0
        assert write_completions[0].transfer_time >= 0
        assert executor.poll_completed() == []

        read_id = executor.submit_read(
            "read-key",
            entry_path,
            [memoryview(destination)],
            8,
        )
        assert read_id is not None
        read_stats = executor.stats()
        assert read_stats.pending_read_jobs == 1
        assert read_stats.pending_read_bytes == 8
        executor.drain()

        read_completions = executor.poll_completed()
        assert len(read_completions) == 1
        assert read_completions[0].job_id == read_id
        assert read_completions[0].success
        assert read_completions[0].operation == "read"
        assert destination == b"abcdefXY"
        assert executor.stats().pending_read_jobs == 0
        assert executor.stats().pending_read_bytes == 0

        delete_id = executor.submit_delete("delete-key", entry_path, 8)
        assert delete_id is not None
        delete_stats = executor.stats()
        assert delete_stats.pending_delete_jobs == 1
        assert delete_stats.pending_delete_bytes == 8
        executor.drain()

        delete_completions = executor.poll_completed()
        assert len(delete_completions) == 1
        assert delete_completions[0].job_id == delete_id
        assert delete_completions[0].success
        assert delete_completions[0].operation == "delete"
        assert not entry_path.exists()
        assert executor.stats().pending_delete_jobs == 0
        assert executor.stats().pending_delete_bytes == 0
    finally:
        executor.shutdown()


def test_executor_reports_io_failure_once(tmp_path: Path) -> None:
    executor = ECFileIOExecutor(n_read_threads=1, n_write_threads=1)
    destination = bytearray(4)
    try:
        job_id = executor.submit_read(
            "missing-key",
            tmp_path / "missing.ec",
            [memoryview(destination)],
            4,
        )
        assert job_id is not None
        executor.drain()

        completions = executor.poll_completed()
        assert len(completions) == 1
        assert completions[0].job_id == job_id
        assert not completions[0].success
        assert completions[0].error is not None
        assert "FileNotFoundError" in completions[0].error
        assert executor.poll_completed() == []
    finally:
        executor.shutdown()


def test_shutdown_drains_then_rejects_new_jobs(tmp_path: Path) -> None:
    executor = ECFileIOExecutor(n_read_threads=1, n_write_threads=1)
    entry_path = tmp_path / "entry.ec"
    job_id = executor.submit_write(
        "write-key",
        entry_path,
        tmp_path / ".entry.ec.write.tmp",
        [memoryview(b"data")],
        4,
    )
    assert job_id is not None

    executor.shutdown()

    completions = executor.poll_completed()
    assert [completion.job_id for completion in completions] == [job_id]
    assert completions[0].success
    assert entry_path.read_bytes() == b"data"
    assert executor.submit_read(
        "read-key",
        entry_path,
        [memoryview(bytearray(4))],
        4,
    ) is None
    executor.shutdown()
