# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import stat
import time
from typing import Any, cast

import pytest

from vllm.snapshot.manager import EngineSnapshotManager, SnapshotState
from vllm.snapshot.protocol import SnapshotControlClient, SnapshotControlError
from vllm.snapshot.providers import CriuCudaSnapshotProvider, FakeSnapshotProvider


def _capture_snapshot(client: SnapshotControlClient) -> dict[str, Any]:
    ticket = client.request("prepare_capture")["ticket"]
    marker = {
        "pid": ticket["root_pid"],
        "nonce": ticket["nonce"],
        "generation": ticket["generation"],
        "snapshot_id": ticket["snapshot_id"],
        "config_hash": ticket["config_hash"],
        "state": "detached",
    }
    with open(ticket["marker_path"], "w") as marker_file:
        json.dump(marker, marker_file)
    return client.request("capture")


def test_fake_provider_round_trip(tmp_path, sleeping_child):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
        config_hash="test-config",
    )
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    try:
        initial = client.request("status")
        assert initial["state"] == SnapshotState.READY
        captured = _capture_snapshot(client)
        assert captured["state"] == SnapshotState.HIBERNATED
        manifest = captured["manifest"]
        assert manifest["format_version"] == 2
        assert manifest["resource_policy"] == {
            "weights": "cuda_image",
            "kv": "cuda_image",
            "runtime": "cuda_image",
        }
        assert manifest["persistence"] == "durable"
        assert manifest["integrity"] == "optimistic"
        assert captured["manager_timings"]["total_seconds"] >= 0
        assert captured["manager_timings"]["artifact_inventory_seconds"] >= 0
        assert manifest["artifacts"]["files"]
        assert "sha256" not in manifest["artifacts"]["files"][0]
        snapshot_dir = tmp_path / "snapshots" / captured["snapshot_id"]
        assert (snapshot_dir / "manifest.sha256").is_file()
        assert os.waitpid(sleeping_child, os.WNOHANG | os.WUNTRACED)[1] != 0
        restored = client.request("restore")
        assert restored["state"] == SnapshotState.ATTACHING
        assert restored["manager_timings"]["total_seconds"] >= 0
        assert restored["manager_timings"]["artifact_validation_seconds"] >= 0
        restored = client.request(
            "confirm_attach",
            {
                field: restored["ticket"][field]
                for field in (
                    "nonce",
                    "generation",
                    "snapshot_id",
                    "config_hash",
                    "root_pid",
                )
            },
        )
        assert restored["state"] == SnapshotState.VERIFYING
        restored = client.request("complete_restore")
        assert restored["state"] == SnapshotState.READY
        assert restored["generation"] == 1
    finally:
        manager.close()


def test_page_cache_snapshot_skips_fsync(tmp_path, sleeping_child, monkeypatch):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
        config_hash="test-config",
        persistence="page_cache",
    )
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", fsync_calls.append)
    try:
        captured = _capture_snapshot(client)

        assert captured["persistence"] == "page_cache"
        assert captured["manifest"]["persistence"] == "page_cache"
        assert captured["manager_timings"]["fsync_seconds"] >= 0
        assert fsync_calls == []
        snapshot_dir = tmp_path / "snapshots" / captured["snapshot_id"]
        assert (snapshot_dir / "manifest.sha256").is_file()

        restored = client.request("restore")
        assert restored["state"] == SnapshotState.ATTACHING
        assert fsync_calls == []
    finally:
        manager.close()


def test_durable_snapshot_syncs_nested_directories(
    tmp_path, sleeping_child, monkeypatch
):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
        config_hash="test-config",
    )
    manager.start()
    nested = tmp_path / "tree" / "images" / "nested"
    nested.mkdir(parents=True)
    (nested / "image").write_bytes(b"image")
    synced_paths: list[Any] = []
    original_sync_directory = manager._sync_directory

    def record_sync(path):
        synced_paths.append(path)
        original_sync_directory(path)

    monkeypatch.setattr(manager, "_sync_directory", record_sync)
    try:
        manager._sync_tree(tmp_path / "tree")
    finally:
        manager.close()

    assert synced_paths == [nested, nested.parent, tmp_path / "tree"]


def test_rejects_invalid_transition(tmp_path, sleeping_child):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
    )
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    try:
        with pytest.raises(SnapshotControlError, match="requires HIBERNATED"):
            client.request("restore")
    finally:
        manager.close()


def test_prepare_capture_keeps_ready_state_when_root_is_dead(tmp_path, monkeypatch):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        123,
    )
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: False)

    with pytest.raises(RuntimeError, match="EngineCore root is not alive"):
        manager.prepare_capture()

    assert manager.snapshot_status()["state"] == SnapshotState.READY
    assert manager.snapshot_status()["operation_started_at"] is None


@pytest.mark.parametrize(
    "state", (SnapshotState.DRAINING, SnapshotState.ATTACHING, SnapshotState.VERIFYING)
)
def test_fail_capture_discards_engine_from_rollback_states(
    tmp_path, sleeping_child, state
):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
    )
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    try:
        client.request("prepare_capture")
        manager.status.state = state

        failed = client.request("fail_capture", {"error": "rollback failed"})

        assert failed["state"] == SnapshotState.FAILED
        assert failed["root_pid"] is None
        assert failed["last_error"] == "rollback failed"
        assert not manager._pid_alive(sleeping_child)
    finally:
        manager.close()


def test_fail_capture_waits_for_provider_process_exit(tmp_path, monkeypatch):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        123,
    )
    manager.status.state = SnapshotState.DRAINING
    alive = iter((True, True, False))
    events: list[Any] = []
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: next(alive))
    monkeypatch.setattr(
        manager.provider,
        "discard_restored",
        lambda pid: events.append(("discard", pid)),
    )
    monkeypatch.setattr(
        time, "sleep", lambda seconds: events.append(("sleep", seconds))
    )

    failed = manager.fail_capture("rollback failed")

    assert failed["state"] == SnapshotState.FAILED
    assert failed["root_pid"] is None
    assert events == [("discard", 123), ("sleep", 0.05)]


def test_fail_capture_records_provider_cleanup_failure(tmp_path, monkeypatch):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        123,
    )
    manager.status.state = SnapshotState.DRAINING
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: True)

    def fail_cleanup(pid):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(manager.provider, "discard_restored", fail_cleanup)

    failed = manager.fail_capture("rollback failed")

    assert failed["state"] == SnapshotState.FAILED
    assert failed["root_pid"] == 123
    assert failed["last_error"] == (
        "rollback failed; Engine cleanup failed: cleanup failed"
    )


def test_start_rejects_nonempty_snapshot_directory(tmp_path, sleeping_child):
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    (snapshot_root / "existing").write_text("preserve me")
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(snapshot_root),
        sleeping_child,
    )

    with pytest.raises(RuntimeError, match="not empty"):
        manager.start()

    assert (snapshot_root / "existing").read_text() == "preserve me"


def test_start_restricts_snapshot_directory_permissions(tmp_path, sleeping_child):
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir(mode=0o755)
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(snapshot_root),
        sleeping_child,
    )

    manager.start()
    try:
        assert stat.S_IMODE(snapshot_root.stat().st_mode) == 0o700
    finally:
        manager.close()


def test_manager_reuses_resolved_provider(tmp_path, sleeping_child):
    provider = FakeSnapshotProvider()

    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
        provider=provider,
    )

    assert manager.provider is provider


def test_resume_signal_reports_exited_process(tmp_path, sleeping_child, monkeypatch):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
    )

    def process_exited(pid, signum):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(manager, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(os, "kill", process_exited)

    assert not manager._signal_snapshot_resume(123)


def test_snapshot_tree_cleanup_reports_residual_state(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    committed = tmp_path / "committed"
    staging.mkdir()
    committed.mkdir()

    def fail_staging_cleanup(path):
        if path == staging:
            raise OSError("cleanup failed")
        path.rmdir()

    monkeypatch.setattr("vllm.snapshot.manager.shutil.rmtree", fail_staging_cleanup)

    error = EngineSnapshotManager._remove_snapshot_trees(staging, committed)

    assert error is not None
    assert str(staging) in str(error)
    assert staging.exists()
    assert not committed.exists()


def test_control_server_rejects_connection_at_capacity(tmp_path, sleeping_child):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
    )

    class Connection:
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()

    class Server:
        def accept(self):
            manager._stop.set()
            return connection, None

    while manager._request_slots.acquire(blocking=False):
        pass
    manager._server = cast(Any, Server())

    manager._serve()

    assert connection.closed


def test_optimistic_restore_allows_same_size_artifact_change(tmp_path, sleeping_child):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
        config_hash="test-config",
    )
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    try:
        captured = _capture_snapshot(client)
        snapshot_dir = tmp_path / "snapshots" / captured["snapshot_id"]
        artifact = snapshot_dir / "fake.img"
        artifact.write_bytes(b"x" * artifact.stat().st_size)

        restored = client.request("restore")

        assert restored["state"] == SnapshotState.ATTACHING
    finally:
        manager.close()


@pytest.mark.parametrize("change", ["add", "remove", "resize", "type"])
def test_optimistic_restore_rejects_artifact_inventory_change(
    tmp_path, sleeping_child, change
):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
        config_hash="test-config",
    )
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    try:
        captured = _capture_snapshot(client)
        snapshot_dir = tmp_path / "snapshots" / captured["snapshot_id"]
        artifact = snapshot_dir / "fake.img"
        if change == "add":
            (snapshot_dir / "added.img").write_bytes(b"added")
        elif change == "remove":
            artifact.unlink()
        elif change == "resize":
            artifact.write_bytes(artifact.read_bytes() + b"x")
        else:
            size = artifact.stat().st_size
            artifact.unlink()
            target = tmp_path / "same-size-target"
            target.write_bytes(b"x" * size)
            artifact.symlink_to(target)

        with pytest.raises(SnapshotControlError, match="artifact inventory mismatch"):
            client.request("restore")
    finally:
        manager.close()


def test_strict_restore_rejects_same_size_artifact_change(tmp_path, sleeping_child):
    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
        config_hash="test-config",
        integrity="strict",
    )
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    try:
        captured = _capture_snapshot(client)
        assert captured["integrity"] == "strict"
        assert captured["manifest"]["integrity"] == "strict"
        assert captured["manager_timings"]["artifact_sha256_seconds"] >= 0
        snapshot_dir = tmp_path / "snapshots" / captured["snapshot_id"]
        artifact = snapshot_dir / "fake.img"
        artifact.write_bytes(b"x" * artifact.stat().st_size)

        with pytest.raises(SnapshotControlError, match="artifact inventory mismatch"):
            client.request("restore")
    finally:
        manager.close()


def test_configured_artifact_identity_ignores_git_metadata(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    git_object = model / ".git" / "lfs" / "objects" / "duplicate"
    git_object.parent.mkdir(parents=True)
    git_object.write_bytes(b"weights")

    identity = EngineSnapshotManager._path_identity(model, "optimistic")

    assert [entry["path"] for entry in identity["files"]] == [
        "config.json",
        "model.safetensors",
    ]
    assert identity["total_bytes"] == 9
    assert all("sha256" not in entry for entry in identity["files"])


def test_optimistic_inventory_skips_file_digest(tmp_path, monkeypatch):
    (tmp_path / "artifact").write_bytes(b"content")

    def fail_digest(path):
        raise AssertionError(f"unexpected content digest for {path}")

    monkeypatch.setattr(
        EngineSnapshotManager, "_file_digest", staticmethod(fail_digest)
    )

    inventory = EngineSnapshotManager._artifact_inventory(tmp_path, "optimistic")

    assert inventory["total_bytes"] == 7
    assert "sha256" not in inventory


def test_nonretriable_restore_failure_enters_failed(tmp_path, sleeping_child):
    class NonretriableProvider(CriuCudaSnapshotProvider):
        def capture(self, root_pid, snapshot_dir):
            return FakeSnapshotProvider().capture(root_pid, snapshot_dir)

        def restore(self, snapshot_dir):
            raise RuntimeError("restore failed")

        def runtime_identity(self):
            return {"provider": "test"}

    manager = EngineSnapshotManager(
        str(tmp_path / "control.sock"),
        str(tmp_path / "snapshots"),
        sleeping_child,
    )
    manager.provider = NonretriableProvider("criu", "cuda-checkpoint")
    manager.compatibility = manager._runtime_compatibility()
    manager.start()
    client = SnapshotControlClient(str(tmp_path / "control.sock"))
    try:
        captured = _capture_snapshot(client)

        with pytest.raises(SnapshotControlError, match="restore failed"):
            client.request("restore")

        assert client.request("status")["state"] == SnapshotState.FAILED
        snapshot_dir = tmp_path / "snapshots" / captured["snapshot_id"]
        assert (snapshot_dir / "restore-attempt.json").is_file()
    finally:
        manager.close()
