# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

import pytest

from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECFileSystemBackpressureConfig,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import (
    ECFileSystemTierManager,
    ECWriteResult,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    get_ec_entry_path,
    get_ec_namespace_path,
    get_ec_temporary_path,
)

_NAMESPACE_DIGEST = "a" * 64
_CACHE_KEY = "b" * 64
_CAPACITY_BYTES = 1 << 20
pytestmark = pytest.mark.cpu_test


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

    def stats(self):
        return SimpleNamespace(
            pending_read_jobs=0,
            pending_read_bytes=0,
            pending_write_jobs=0,
            pending_write_bytes=0,
            pending_delete_jobs=0,
            pending_delete_bytes=0,
            oldest_read_queue_age=0.0,
            oldest_write_queue_age=0.0,
            oldest_delete_queue_age=0.0,
        )


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
    assert submission.temporary_path.parent.parent == (
        get_ec_namespace_path(tmp_path, _NAMESPACE_DIGEST) / ".tmp"
    )
    assert submission.temporary_path.suffix == ".tmp"
    assert submission.fsync

    executor.finish(submission.job_id, success=True)
    assert manager.poll_completed_reads() == []
    assert manager.contains(_CACHE_KEY)
    assert manager.poll_completed_writes() == [
        ECWriteResult(cache_key=_CACHE_KEY, success=True, file_size_bytes=4)
    ]
    assert manager.poll_completed_writes() == []


def test_existing_or_pending_key_is_not_rewritten(tmp_path: Path) -> None:
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
    assert not manager.submit_write(_CACHE_KEY, [memoryview(b"new!")], 4)
    executor.finish(0, success=True)
    manager.poll_completed_writes()
    assert not manager.submit_write(_CACHE_KEY, [memoryview(b"new!")], 4)
    assert manager.submit_read(_CACHE_KEY, [memoryview(bytearray(4))])
    assert not manager.submit_write(_CACHE_KEY, [memoryview(b"new!")], 4)
    assert len(executor.submissions) == 2
    executor.finish(1, success=True)
    assert manager.poll_completed_reads()[0].success
    assert manager.contains(_CACHE_KEY)
    assert manager.poll_completed_writes() == []


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
    assert not manager.submit_write(third_key, [memoryview(b"data")], 4)

    assert manager.used_bytes == 8
    assert manager.contains(first_key)
    assert not manager.contains(second_key)
    assert not manager.contains(third_key)
    deletion = executor.submissions[2]
    assert deletion.operation == "delete"
    assert deletion.cache_key == second_key
    assert not manager.submit_write(second_key, [memoryview(b"data")], 4)

    executor.finish(deletion.job_id, success=True)
    assert manager.poll_completed_reads() == []
    assert manager.used_bytes == 4
    assert manager.submit_write(third_key, [memoryview(b"data")], 4)
    executor.finish(3, success=True)
    assert manager.poll_completed_writes()[0].success
    assert manager.contains(first_key) and manager.contains(third_key)


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

    assert not manager.submit_write(second_key, [memoryview(b"data")], 4)

    assert manager.contains(first_key)
    assert not manager.contains(second_key)
    assert len(executor.submissions) == 2

    executor.finish(1, success=True)
    assert len(manager.poll_completed_reads()) == 1
    assert not manager.submit_write(second_key, [memoryview(b"data")], 4)
    assert executor.submissions[2].operation == "delete"
    assert executor.submissions[2].cache_key == first_key
    executor.finish(2, success=True)
    manager.poll_completed_reads()
    assert manager.submit_write(second_key, [memoryview(b"data")], 4)


def test_pending_writes_reserve_capacity_and_failure_releases_it(tmp_path):
    executor = _FakeExecutor()
    manager = ECFileSystemTierManager(
        tmp_path, _NAMESPACE_DIGEST, 4, 1, 1, executor=executor
    )
    first, second = (f"{number:064x}" for number in range(2))
    assert manager.submit_write(first, [memoryview(b"data")], 4)
    assert manager.used_bytes == 4
    assert not manager.submit_write(second, [memoryview(b"data")], 4)
    assert len(executor.submissions) == 1
    executor.finish(0, success=False)
    assert not manager.poll_completed_writes()[0].success
    assert manager.used_bytes == 4
    assert executor.submissions[1].operation == "delete"
    executor.finish(1, success=True)
    manager.poll_completed_writes()
    assert manager.used_bytes == 0
    assert manager.submit_write(second, [memoryview(b"data")], 4)
    executor.finish(2, success=True)
    assert manager.poll_completed_writes()[0].success
    assert manager.used_bytes == 4


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


@pytest.mark.parametrize("failures", [1, 3])
def test_failed_eviction_retains_capacity_and_resumes_stores_after_retry(
    tmp_path, monkeypatch, failures
):
    from vllm.distributed.ec_transfer.ec_connector.tiering import executor

    remove_entry = executor.remove_entry

    def fail_unlink(path):
        nonlocal failures
        if failures:
            failures -= 1
            raise PermissionError("temporary unlink failure")
        remove_entry(path)

    monkeypatch.setattr(executor, "remove_entry", fail_unlink)
    manager = ECFileSystemTierManager(
        tmp_path, _NAMESPACE_DIGEST, 4, n_read_threads=1, n_write_threads=1
    )
    first, second, third = (f"{number:064x}" for number in range(3))
    try:
        assert manager.submit_write(first, [memoryview(b"data")], 4)
        manager.drain()
        assert manager.poll_completed_writes()[0].success
        for _ in range(failures):
            assert not manager.submit_write(second, [memoryview(b"data")], 4)
            manager.drain()
            assert manager.used_bytes == 4
            assert manager.contains(first)
            assert not manager.contains(second)
            assert sum(path.stat().st_size for path in tmp_path.rglob("*.ec")) == 4
        assert not manager.submit_write(second, [memoryview(b"data")], 4)
        # The retry is in flight; another store must still be rejected.
        assert not manager.submit_write(third, [memoryview(b"data")], 4)
        manager.drain()
        assert manager.used_bytes == 0
        assert not manager.contains(first)
        assert manager.submit_write(second, [memoryview(b"data")], 4)
        assert not manager.submit_write(third, [memoryview(b"data")], 4)
        manager.drain()
        assert manager.poll_completed_writes()[0].success
        assert sum(path.stat().st_size for path in tmp_path.rglob("*.ec")) == 4
    finally:
        manager.shutdown()


def test_manager_recovers_published_entries_from_current_namespace(
    tmp_path: Path,
) -> None:
    cache_key = f"{1:064x}"
    entry_path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, cache_key)
    entry_path.parent.mkdir(parents=True)
    entry_path.write_bytes(b"data")

    malformed_path = entry_path.parent / "not-a-cache-key.ec"
    malformed_path.write_bytes(b"ignored")
    temporary_path = get_ec_temporary_path(entry_path, "unfinished", "0" * 32)
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
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
    assert manager.used_bytes == 8
    assert len(executor.submissions) == 1
    assert executor.submissions[0].operation == "delete"
    assert executor.submissions[0].cache_key == old_key
    executor.finish(0, success=True)
    manager.poll_completed_reads()
    assert manager.used_bytes == 4


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


def test_async_lookup_discovers_late_file_and_releases_request(tmp_path: Path) -> None:
    from vllm.distributed.ec_transfer.ec_connector.tiering.io import store_entry

    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=_CAPACITY_BYTES,
        n_read_threads=1,
        n_write_threads=1,
    )

    path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, _CACHE_KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    store_entry(
        path, get_ec_temporary_path(path, "late", "0" * 32), [memoryview(b"data")], 4
    )
    try:
        assert not manager.contains(_CACHE_KEY)
        assert manager.lookup(_CACHE_KEY, 4, "request") is None
        assert _resolved(manager, _CACHE_KEY, "request")
        assert manager.contains(_CACHE_KEY)
        destination = bytearray(4)
        assert manager.submit_read(_CACHE_KEY, [memoryview(destination)])
        manager.drain()
        assert manager.poll_completed_reads()[0].success
        assert destination == b"data"
        manager.release_request("request")

        missing = "c" * 64
        assert manager.lookup(missing, 4, "request2") is None
        assert _resolved(manager, missing, "request2") is False
        missing_path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, missing)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        store_entry(
            missing_path,
            get_ec_temporary_path(missing_path, "new", "0" * 32),
            [memoryview(b"next")],
            4,
        )
        assert manager.lookup(missing, 4, "request2") is False
        manager.release_request("request2")
        assert manager.lookup(missing, 4, "request3") is None
        assert _resolved(manager, missing, "request3")
        manager.release_request("request3")
    finally:
        manager.shutdown()


def _resolved(manager, key, request_id):
    deadline = monotonic() + 5
    pause = Event()
    while (result := manager.lookup(key, 4, request_id)) is None:
        assert monotonic() < deadline
        pause.wait(0.001)
    return result


@pytest.mark.parametrize("indexed_at_startup", [True, False])
def test_external_deletion_invalidates_index_without_repeated_reads(
    tmp_path, indexed_at_startup
):
    from vllm.distributed.ec_transfer.ec_connector.tiering.io import store_entry

    path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, _CACHE_KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    if indexed_at_startup:
        store_entry(
            path,
            get_ec_temporary_path(path, "seed", "0" * 32),
            [memoryview(b"data")],
            4,
        )
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=_CAPACITY_BYTES,
        n_read_threads=1,
        n_write_threads=1,
    )
    try:
        if not indexed_at_startup:
            store_entry(
                path,
                get_ec_temporary_path(path, "late", "0" * 32),
                [memoryview(b"data")],
                4,
            )
            assert manager.lookup(_CACHE_KEY, 4, "request") is None
            assert _resolved(manager, _CACHE_KEY, "request")
        assert manager.lookup(_CACHE_KEY, 4, "request") is True
        path.unlink()
        destination = bytearray(b"keep")
        assert manager.submit_read(_CACHE_KEY, [memoryview(destination)])
        manager.drain()
        assert not manager.poll_completed_reads()[0].success
        assert destination == b"keep"
        assert not manager.contains(_CACHE_KEY)
        assert manager.used_bytes == 0
        assert manager.lookup(_CACHE_KEY, 4, "request") is False
        assert manager.lookup(_CACHE_KEY, 4, "request") is False
        manager.release_request("request")
    finally:
        manager.shutdown()


@pytest.mark.parametrize("stale_size", [False, True])
def test_index_hit_skips_probe_unless_recorded_size_is_stale(
    tmp_path, monkeypatch, stale_size
):
    from vllm.distributed.ec_transfer.ec_connector.tiering.io import store_entry

    path = get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, _CACHE_KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"old-data" if stale_size else b"data"
    store_entry(
        path,
        get_ec_temporary_path(path, "seed", "0" * 32),
        [memoryview(data)],
        len(data),
    )
    manager = ECFileSystemTierManager(
        root_dir=tmp_path,
        namespace_digest=_NAMESPACE_DIGEST,
        capacity_bytes=_CAPACITY_BYTES,
        n_read_threads=1,
        n_write_threads=1,
    )
    if stale_size:
        store_entry(
            path,
            get_ec_temporary_path(path, "fixed", "0" * 32),
            [memoryview(b"data")],
            4,
        )
    probes = []
    original = Path.lstat

    def record_probe(file_path, *args, **kwargs):
        if file_path == path:
            probes.append(file_path)
        return original(file_path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", record_probe)
    try:
        if stale_size:
            assert manager.lookup(_CACHE_KEY, 4, "first") is False
            assert manager.used_bytes == len(data)
            manager.drain()
            assert manager.used_bytes == 0
            assert not path.exists()
            store_entry(
                path,
                get_ec_temporary_path(path, "retry", "0" * 32),
                [memoryview(b"data")],
                4,
            )
            assert manager.lookup(_CACHE_KEY, 4, "first") is None
            assert _resolved(manager, _CACHE_KEY, "first")
        else:
            assert manager.lookup(_CACHE_KEY, 4, "first") is True
        assert manager.used_bytes == 4
        assert manager.lookup(_CACHE_KEY, 4, "second") is True
        manager.release_request("first")
        assert manager.lookup(_CACHE_KEY, 4, "second") is True
        manager.release_request("second")
    finally:
        manager.shutdown()
    assert probes == ([path] if stale_size else [])


@pytest.mark.parametrize("failure", ["read", "size_mismatch"])
def test_invalid_entry_holds_capacity_until_cleanup_succeeds(
    tmp_path, monkeypatch, failure
):
    from vllm.distributed.ec_transfer.ec_connector.tiering import executor

    release_delete = Event()
    delete_started = Event()
    deny_delete = True
    remove_entry = executor.remove_entry

    def fail_read(*args):
        raise OSError("temporary read failure")

    def delete(path):
        delete_started.set()
        assert release_delete.wait(timeout=10)
        if deny_delete:
            raise PermissionError("temporary unlink failure")
        remove_entry(path)

    monkeypatch.setattr(executor, "load_entry", fail_read)
    monkeypatch.setattr(executor, "remove_entry", delete)
    manager = ECFileSystemTierManager(tmp_path, _NAMESPACE_DIGEST, 4, 1, 1)
    other = "c" * 64
    try:
        assert manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
        manager.drain()
        assert manager.poll_completed_writes()[0].success
        assert manager.lookup(_CACHE_KEY, 4, "request")
        if failure == "read":
            assert manager.submit_read(_CACHE_KEY, [memoryview(bytearray(4))])
            manager._executor.drain()
            assert not manager.poll_completed_reads()[0].success
        else:
            assert manager.lookup(_CACHE_KEY, 2, "size-mismatch") is False
        assert manager.used_bytes == 4
        assert delete_started.wait(timeout=5)
        assert not manager.contains(_CACHE_KEY)
        assert manager.lookup(_CACHE_KEY, 4, "request") is False
        assert not manager.submit_write(other, [memoryview(b"next")], 4)
        release_delete.set()
        manager.drain()
        assert manager.used_bytes == 4
        assert len(list(tmp_path.rglob("*.ec"))) == 1
        # Failed cleanup must not turn the unreadable entry back into a hit.
        assert not manager.contains(_CACHE_KEY)
        assert manager.lookup(_CACHE_KEY, 4, "new-request") is False
        deny_delete = False
        assert not manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
        manager.drain()
        assert manager.used_bytes == 0
        assert not list(tmp_path.rglob("*.ec"))
        assert manager.submit_write(_CACHE_KEY, [memoryview(b"next")], 4)
        manager.drain()
        assert manager.poll_completed_writes()[0].success
        assert manager.contains(_CACHE_KEY)
        assert manager.lookup(_CACHE_KEY, 4, "request") is True
        assert sum(path.stat().st_size for path in tmp_path.rglob("*.ec")) == 4
    finally:
        release_delete.set()
        manager.release_request("request")
        manager.release_request("new-request")
        manager.release_request("size-mismatch")
        manager.shutdown()


@pytest.mark.parametrize("deny_recovery_cleanup", [False, True])
def test_recovery_skips_live_writer_and_cleans_crashed_writer(
    tmp_path, monkeypatch, deny_recovery_cleanup
):
    ready = tmp_path / "ready"
    code = """
import sys
from pathlib import Path
from threading import Event
from vllm.distributed.ec_transfer.ec_connector.tiering import io
from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import (
    ECFileSystemTierManager,
)
writev = io.os.writev
def blocked_write(fd, buffers):
    writev(fd, buffers)
    Path(sys.argv[2]).touch()
    Event().wait()
io.os.writev = blocked_write
manager = ECFileSystemTierManager(Path(sys.argv[1]), "a" * 64, 8, 1, 1)
assert manager.submit_write("b" * 64, [memoryview(b"data")], 4)
Event().wait()
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(ready)])
    live = recovered = None
    unlink = Path.unlink

    def unlink_or_fail(path, *args, **kwargs):
        if deny_recovery_cleanup and path.suffix == ".tmp":
            raise PermissionError("orphan cleanup denied")
        return unlink(path, *args, **kwargs)

    try:
        deadline = monotonic() + 30
        while not ready.exists():
            assert process.poll() is None
            assert monotonic() < deadline
            Event().wait(0.001)
        temporary = next(tmp_path.rglob("*.ec.*.tmp"))
        rglob = Path.rglob

        def reject_recursive_temporary_scan(path, pattern, *args, **kwargs):
            assert pattern != "*.tmp", (
                "cleanup must only scan the dead owner's directory"
            )
            return rglob(path, pattern, *args, **kwargs)

        monkeypatch.setattr(Path, "rglob", reject_recursive_temporary_scan)
        live = ECFileSystemTierManager(tmp_path, _NAMESPACE_DIGEST, 8, 1, 1)
        assert temporary.read_bytes() == b"data"
        assert live.submit_write("c" * 64, [memoryview(b"next")], 4)
        live.drain()
        assert live.poll_completed_writes()[0].success
        process.kill()
        process.wait(timeout=10)
        monkeypatch.setattr(Path, "unlink", unlink_or_fail)
        if deny_recovery_cleanup:
            with pytest.raises(PermissionError, match="orphan cleanup denied"):
                ECFileSystemTierManager(tmp_path, _NAMESPACE_DIGEST, 8, 1, 1)
            assert temporary.exists()
            assert (
                temporary.parents[2] / ".owners" / f"{temporary.parent.name}.lock"
            ).exists()
            deny_recovery_cleanup = False
        recovered = ECFileSystemTierManager(tmp_path, _NAMESPACE_DIGEST, 8, 1, 1)
        recovered.drain()
        assert not temporary.exists()
        assert recovered.contains("c" * 64)
    finally:
        deny_recovery_cleanup = False
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        if recovered is not None:
            recovered.shutdown()
        if live is not None:
            live.shutdown()
    assert not list(tmp_path.rglob("*.lock"))
    assert not list(
        (get_ec_namespace_path(tmp_path, _NAMESPACE_DIGEST) / ".tmp").iterdir()
    )


def test_failed_temporary_cleanup_retains_capacity_and_retries(tmp_path, monkeypatch):
    from vllm.distributed.ec_transfer.ec_connector.tiering import io

    unlink = Path.unlink
    replace = io.os.replace
    denied = True

    def fail_publish(*args):
        raise OSError("publish failed")

    def deny_cleanup(path, *args, **kwargs):
        if denied and path.suffix == ".tmp":
            raise PermissionError("temporary unlink denied")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(io.os, "replace", fail_publish)
    monkeypatch.setattr(Path, "unlink", deny_cleanup)
    manager = ECFileSystemTierManager(tmp_path, _NAMESPACE_DIGEST, 4, 1, 1)
    try:
        assert manager.submit_write(_CACHE_KEY, [memoryview(b"data")], 4)
        manager.drain()
        assert not manager.poll_completed_writes()[0].success
        assert len(list(tmp_path.rglob("*.ec.*.tmp"))) == 1
        assert manager.used_bytes == 4
        denied = False
        monkeypatch.setattr(io.os, "replace", replace)
        assert not manager.submit_write("c" * 64, [memoryview(b"next")], 4)
        manager.drain()
        assert not list(tmp_path.rglob("*.ec.*.tmp"))
        assert manager.used_bytes == 0
        assert manager.submit_write("c" * 64, [memoryview(b"next")], 4)
        manager.drain()
        assert manager.poll_completed_writes()[0].success
    finally:
        denied = False
        manager.shutdown()
    assert not list(tmp_path.rglob("*.lock"))


def test_recovery_many_dead_owners_uses_bounded_file_descriptors(tmp_path):
    namespace = get_ec_namespace_path(tmp_path, _NAMESPACE_DIGEST)
    (namespace / ".owners").mkdir(parents=True)
    for number in range(80):
        owner = f"{number:032x}"
        (namespace / ".owners" / f"{owner}.lock").touch()
        path = get_ec_temporary_path(
            get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, _CACHE_KEY), "write", owner
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(b"data")
    code = """
import resource
import sys
from pathlib import Path
from vllm.distributed.ec_transfer.ec_connector.tiering.temporary import (
    ECTemporaryFileOwner,
)
namespace = Path(sys.argv[1])
hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard_limit))
owner = ECTemporaryFileOwner(namespace)
owner.close({})
assert not list(namespace.glob('.tmp/*/*.tmp'))
assert not list((namespace / '.owners').iterdir())
"""
    subprocess.run([sys.executable, "-c", code, str(namespace)], check=True, timeout=30)


def test_recovery_metadata_error_preserves_owner_for_next_start(tmp_path, monkeypatch):
    from vllm.distributed.ec_transfer.ec_connector.tiering.temporary import (
        ECTemporaryFileOwner,
    )

    namespace = get_ec_namespace_path(tmp_path, _NAMESPACE_DIGEST)
    owner_id = "a" * 32
    lock = namespace / ".owners" / f"{owner_id}.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    path = get_ec_temporary_path(
        get_ec_entry_path(tmp_path, _NAMESPACE_DIGEST, _CACHE_KEY), "write", owner_id
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"data")
    lstat = Path.lstat

    def failed_stat(candidate, *args, **kwargs):
        if candidate == path:
            raise OSError("metadata I/O error")
        return lstat(candidate, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", failed_stat)
        with pytest.raises(OSError, match="metadata I/O error"):
            ECTemporaryFileOwner(namespace)
    assert path.read_bytes() == b"data"
    assert lock.exists()
    recovered = ECTemporaryFileOwner(namespace)
    recovered.close({})
    assert not path.exists()
    assert not lock.exists()
