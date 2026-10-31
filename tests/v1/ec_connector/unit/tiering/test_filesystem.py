# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
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
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    get_ec_entry_path,
    get_ec_temporary_path,
)

_NAMESPACE_DIGEST = "a" * 64
_CACHE_KEY = "b" * 64
_CAPACITY_BYTES = 1 << 20


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

    def submit_delete(
        self,
        cache_key: str,
        entry_path: Path,
        file_size_bytes: int,
    ) -> int:
        return self._submit("delete", cache_key, entry_path, file_size_bytes)

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
        capacity_bytes=_CAPACITY_BYTES,
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
        capacity_bytes=_CAPACITY_BYTES,
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
        capacity_bytes=_CAPACITY_BYTES,
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
        capacity_bytes=_CAPACITY_BYTES,
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
        capacity_bytes=12 << 20,
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


def test_manager_evicts_least_recently_used_entry_in_background(
    tmp_path: Path,
) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=8,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )
    first_key = f"{1:064x}"
    second_key = f"{2:064x}"
    third_key = f"{3:064x}"

    for job_id, cache_key in enumerate((first_key, second_key)):
        assert manager.submit_write(cache_key, [memoryview(b"data")], 4)
        executor.finish(job_id, success=True)
        assert len(manager.poll_completed_writes()) == 1

    assert manager.used_bytes == 8
    assert manager.contains(first_key)
    assert manager.submit_write(third_key, [memoryview(b"data")], 4)
    executor.finish(2, success=True)
    assert len(manager.poll_completed_writes()) == 1

    assert manager.used_bytes == 8
    assert manager.contains(first_key)
    assert not manager.contains(second_key)
    assert manager.contains(third_key)
    deletion = executor.submissions[3]
    assert deletion.operation == "delete"
    assert deletion.cache_key == second_key
    assert not manager.submit_write(second_key, [memoryview(b"data")], 4)

    executor.finish(deletion.job_id, success=True)
    assert manager.poll_completed_reads() == []
    assert manager.submit_write(second_key, [memoryview(b"data")], 4)


def test_manager_does_not_evict_an_entry_with_an_active_read(
    tmp_path: Path,
) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=4,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )
    first_key = f"{1:064x}"
    second_key = f"{2:064x}"

    assert manager.submit_write(first_key, [memoryview(b"data")], 4)
    executor.finish(0, success=True)
    manager.poll_completed_writes()
    assert manager.submit_read(first_key, [memoryview(bytearray(4))])

    assert manager.submit_write(second_key, [memoryview(b"data")], 4)
    executor.finish(2, success=True)
    manager.poll_completed_writes()

    assert manager.contains(first_key)
    assert not manager.contains(second_key)
    assert executor.submissions[3].operation == "delete"
    assert executor.submissions[3].cache_key == second_key

    executor.finish(1, success=True)
    executor.finish(3, success=True)
    assert len(manager.poll_completed_reads()) == 1


def test_manager_rejects_entry_larger_than_capacity(tmp_path: Path) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=3,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )

    assert not manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
    assert executor.submissions == []
    assert manager.used_bytes == 0


def test_manager_recovers_published_entries_from_current_namespace(
    tmp_path: Path,
) -> None:
    cache_key = f"{1:064x}"
    entry_path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, cache_key)
    entry_path.parent.mkdir(parents=True)
    entry_path.write_bytes(b"data")

    malformed_path = entry_path.parent / "not-a-cache-key.ec"
    malformed_path.write_bytes(b"ignored")
    temporary_path = get_ec_temporary_path(entry_path, "unfinished")
    temporary_path.write_bytes(b"ignored")
    symlink_key = f"{2:064x}"
    symlink_path = get_ec_entry_path(
        tmp_path,
        _NAMESPACE_DIGEST,
        symlink_key,
    )
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    outside_path = tmp_path / "outside.ec"
    outside_path.write_bytes(b"outside")
    symlink_path.symlink_to(outside_path)

    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=_CAPACITY_BYTES,
        n_read_threads=1,
        n_write_threads=1,
        executor=_FakeExecutor(),
    )

    assert manager.contains(cache_key)
    assert not manager.contains(symlink_key)
    assert manager.used_bytes == 4


def test_manager_evicts_oldest_recovered_entry_over_capacity(
    tmp_path: Path,
) -> None:
    old_key = f"{1:064x}"
    new_key = f"{2:064x}"
    old_path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, old_key)
    new_path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, new_key)
    for entry_path in (old_path, new_path):
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_bytes(b"data")
    os.utime(old_path, ns=(1, 1))
    os.utime(new_path, ns=(2, 2))
    executor = _FakeExecutor()

    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=4,
        n_read_threads=1,
        n_write_threads=1,
        executor=executor,
    )

    assert not manager.contains(old_key)
    assert manager.contains(new_key)
    assert manager.used_bytes == 4
    assert len(executor.submissions) == 1
    assert executor.submissions[0].operation == "delete"
    assert executor.submissions[0].cache_key == old_key


def test_manager_shutdown_drains_completions_and_rejects_submission(
    tmp_path: Path,
) -> None:
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=_CAPACITY_BYTES,
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
