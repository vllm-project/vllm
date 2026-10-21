# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECFileSystemBackpressureConfig,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import (
    ECFileSystemTier,
    ECFileSystemTierManager,
    ECReadResult,
    ECWriteResult,
)

_NAMESPACE_DIGEST = "a" * 64
_CACHE_KEY = "b" * 64


class _FakeExecutor:
    def __init__(self) -> None:
        self.next_job_id = 0
        self.submissions: list[SimpleNamespace] = []
        self.completions: list[SimpleNamespace] = []
        self.shutdown_called = False

    def submit_read(
        self,
        cache_key: str,
        entry_path: Path,
        destinations: Sequence[memoryview],
        file_size_bytes: int,
    ) -> int:
        return self._submit("read", cache_key, entry_path, file_size_bytes)

    def submit_write(
        self,
        cache_key: str,
        entry_path: Path,
        temporary_path: Path,
        sources: Sequence[memoryview],
        file_size_bytes: int,
        *,
        fsync: bool = False,
    ) -> int:
        job_id = self._submit("write", cache_key, entry_path, file_size_bytes)
        self.submissions[-1].temporary_path = temporary_path
        self.submissions[-1].fsync = fsync
        return job_id

    def _submit(
        self,
        operation: str,
        cache_key: str,
        entry_path: Path,
        file_size_bytes: int,
    ) -> int:
        job_id = self.next_job_id
        self.next_job_id += 1
        self.submissions.append(
            SimpleNamespace(
                job_id=job_id,
                operation=operation,
                cache_key=cache_key,
                entry_path=entry_path,
                file_size_bytes=file_size_bytes,
            )
        )
        return job_id

    def finish(
        self,
        job_id: int,
        *,
        success: bool,
        error: str | None = None,
        queue_time: float = 0.0,
        transfer_time: float = 0.0,
    ) -> None:
        submission = self.submissions[job_id]
        self.completions.append(
            SimpleNamespace(
                job_id=job_id,
                operation=submission.operation,
                cache_key=submission.cache_key,
                success=success,
                file_size_bytes=submission.file_size_bytes,
                queue_time=queue_time,
                transfer_time=transfer_time,
                error=error,
            )
        )

    def poll_completed(self) -> list[SimpleNamespace]:
        completions, self.completions = self.completions, []
        return completions

    def drain(self) -> None:
        pass

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeFileSystemTier:
    def contains(self, cache_key: str) -> bool:
        return False

    def submit_read(
        self,
        cache_key: str,
        destinations: Sequence[memoryview],
    ) -> bool:
        return True

    def submit_write(
        self,
        cache_key: str,
        sources: Sequence[memoryview],
        file_size_bytes: int,
    ) -> bool:
        return True

    def poll_completed_reads(self) -> list[ECReadResult]:
        return []

    def poll_completed_writes(self) -> list[ECWriteResult]:
        return []

    def shutdown(self) -> None:
        pass


def test_filesystem_tier_protocol_is_structural() -> None:
    assert isinstance(_FakeFileSystemTier(), ECFileSystemTier)


def test_io_results_represent_success_and_failure() -> None:
    read_result = ECReadResult(
        cache_key="read", success=True, file_size_bytes=16
    )
    write_result = ECWriteResult(
        cache_key="write",
        success=False,
        error="disk full",
    )

    assert read_result.file_size_bytes == 16
    assert read_result.error is None
    assert not write_result.success
    assert write_result.error == "disk full"


def test_manager_publishes_only_completed_writes_and_routes_completions(
    tmp_path: Path,
) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        n_read_threads=1,
        n_write_threads=1,
        fsync=True,
        executor=executor,
    )

    assert manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
    assert not manager.contains(_CACHE_KEY)
    submission = executor.submissions[0]
    assert submission.entry_path.suffix == ".ec"
    assert submission.temporary_path.parent == submission.entry_path.parent
    assert submission.temporary_path.suffix == ".tmp"
    assert submission.fsync

    executor.finish(submission.job_id, success=True)
    assert manager.poll_completed_reads() == []
    assert manager.contains(_CACHE_KEY)
    assert manager.poll_completed_writes() == [
        ECWriteResult(cache_key=_CACHE_KEY, success=True, file_size_bytes=4)
    ]
    assert manager.poll_completed_writes() == []


def test_manager_read_failure_invalidates_indexed_entry(
    tmp_path: Path,
) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )
    assert manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
    executor.finish(0, success=True)
    manager.poll_completed_writes()

    assert manager.submit_read(_CACHE_KEY, [memoryview(bytearray(4))])
    executor.finish(1, success=False, error="missing")

    assert manager.poll_completed_writes() == []
    assert manager.poll_completed_reads() == [
        ECReadResult(
            cache_key=_CACHE_KEY,
            success=False,
            file_size_bytes=4,
            error="missing",
        )
    ]
    assert not manager.contains(_CACHE_KEY)


def test_stale_read_failure_does_not_remove_new_generation(tmp_path: Path) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )
    assert manager.submit_write(_CACHE_KEY, [memoryview(b"old!")], 4)
    executor.finish(0, success=True)
    manager.poll_completed_writes()

    assert manager.submit_read(_CACHE_KEY, [memoryview(bytearray(4))])
    assert manager.submit_write(_CACHE_KEY, [memoryview(b"new!")], 4)
    executor.finish(2, success=True)
    executor.finish(1, success=False, error="stale read")

    assert manager.poll_completed_reads() == [
        ECReadResult(
            cache_key=_CACHE_KEY,
            success=False,
            file_size_bytes=4,
            error="stale read",
        )
    ]
    assert manager.contains(_CACHE_KEY)
    assert manager.poll_completed_writes() == [
        ECWriteResult(cache_key=_CACHE_KEY, success=True, file_size_bytes=4)
    ]


def test_failed_rewrite_keeps_existing_index_entry(tmp_path: Path) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )
    assert manager.submit_write(_CACHE_KEY, [memoryview(b"old!")], 4)
    executor.finish(0, success=True)
    manager.poll_completed_writes()

    assert manager.submit_write(_CACHE_KEY, [memoryview(b"new!")], 4)
    executor.finish(1, success=False, error="disk full")

    assert manager.poll_completed_writes() == [
        ECWriteResult(
            cache_key=_CACHE_KEY,
            success=False,
            file_size_bytes=4,
            error="disk full",
        )
    ]
    assert manager.contains(_CACHE_KEY)


def test_manager_skips_new_writes_under_store_pressure(tmp_path: Path) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        n_read_threads=1,
        n_write_threads=1,
        backpressure=ECFileSystemBackpressureConfig(
            enabled=True,
            high_water_s=0.010,
            low_water_s=0.002,
        ),
        executor=executor,
    )
    file_size_bytes = 1 << 20
    for job_id in range(11):
        assert manager.submit_write(
            f"{job_id:064x}",
            [memoryview(b"x")],
            file_size_bytes,
        )
        executor.finish(job_id, success=True, transfer_time=0.04)
        assert len(manager.poll_completed_writes()) == 1

    assert not manager.submit_write(
        _CACHE_KEY,
        [memoryview(b"x")],
        file_size_bytes,
    )
    assert len(executor.submissions) == 11
    assert manager.submit_read(
        f"{0:064x}",
        [memoryview(bytearray(1))],
    )
    assert executor.submissions[-1].operation == "read"


def test_manager_shutdown_drains_completions_and_rejects_submission(
    tmp_path: Path,
) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )
    assert manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
    executor.finish(0, success=True)

    manager.shutdown()

    assert executor.shutdown_called
    assert manager.contains(_CACHE_KEY)
    assert manager.poll_completed_writes() == [
        ECWriteResult(cache_key=_CACHE_KEY, success=True, file_size_bytes=4)
    ]
    assert not manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
    assert not manager.submit_read(_CACHE_KEY, [memoryview(bytearray(4))])
    manager.shutdown()
