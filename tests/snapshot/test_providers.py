# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.snapshot.providers import (
    CriuCudaSnapshotProvider,
    make_snapshot_provider,
)


def criu_provider() -> CriuCudaSnapshotProvider:
    return CriuCudaSnapshotProvider("criu", "cuda-checkpoint")


def test_criu_provider_resolves_executable_paths(tmp_path, monkeypatch):
    paths = {
        "criu": tmp_path / "bin" / "criu",
        "cuda-checkpoint": tmp_path / "bin" / "cuda-checkpoint",
    }
    monkeypatch.setattr(
        "vllm.snapshot.providers.shutil.which", lambda name: str(paths[name])
    )

    provider = make_snapshot_provider("criu_cuda")

    assert isinstance(provider, CriuCudaSnapshotProvider)
    assert provider.criu_path == str(paths["criu"].resolve())
    assert provider.cuda_checkpoint_path == str(paths["cuda-checkpoint"].resolve())


@pytest.mark.parametrize("missing", ("criu", "cuda-checkpoint"))
def test_criu_provider_requires_executables(monkeypatch, missing):
    monkeypatch.setattr(
        "vllm.snapshot.providers.shutil.which",
        lambda name: None if name == missing else f"/usr/bin/{name}",
    )

    with pytest.raises(FileNotFoundError, match=missing):
        make_snapshot_provider("criu_cuda")


def test_criu_provider_removes_successful_restore_work(tmp_path, monkeypatch):
    provider = criu_provider()
    snapshot_dir = tmp_path / "snapshot"
    (snapshot_dir / "images").mkdir(parents=True)

    def run(command, **kwargs):
        if "--pidfile" in command:
            pidfile = command[command.index("--pidfile") + 1]
            with open(pidfile, "w") as pidfile_handle:
                pidfile_handle.write(f"{os.getpid()}\n")

    monkeypatch.setattr(provider, "_run", run)
    monkeypatch.setattr(provider, "_cuda_pids", lambda *args, **kwargs: [123])
    monkeypatch.setattr(provider, "_run_cuda", lambda *args, **kwargs: None)

    result = provider.restore(snapshot_dir)

    assert result["root_pid"] == os.getpid()
    assert not list(snapshot_dir.glob("restore-work-*"))


def test_criu_provider_reports_restore_phase_timings(tmp_path, monkeypatch):
    provider = criu_provider()
    snapshot_dir = tmp_path / "snapshot"
    (snapshot_dir / "images").mkdir(parents=True)
    timestamps = iter(float(value) for value in range(6))

    def run(command, **kwargs):
        if "--pidfile" in command:
            pidfile = command[command.index("--pidfile") + 1]
            with open(pidfile, "w") as pidfile_handle:
                pidfile_handle.write(f"{os.getpid()}\n")

    monkeypatch.setattr(
        "vllm.snapshot.providers.time.monotonic", lambda: next(timestamps)
    )
    monkeypatch.setattr(provider, "_run", run)
    monkeypatch.setattr(provider, "_cuda_pids", lambda *args, **kwargs: [123])
    monkeypatch.setattr(provider, "_run_cuda", lambda *args, **kwargs: None)

    result = provider.restore(snapshot_dir)

    assert result["criu_restore_seconds"] == 1.0
    assert result["root_pid_read_seconds"] == 1.0
    assert result["cuda_pid_discovery_seconds"] == 1.0
    assert result["cuda_restore_action_seconds"] == 1.0
    assert result["cuda_unlock_seconds"] == 1.0
    assert result["cuda_restore_seconds"] == 4.0
    assert result["restore_seconds"] == 5.0


def test_criu_provider_removes_failed_restore_work(tmp_path, monkeypatch):
    provider = criu_provider()
    snapshot_dir = tmp_path / "snapshot"
    (snapshot_dir / "images").mkdir(parents=True)
    killed: list[int] = []

    def run(command, **kwargs):
        if "--pidfile" in command:
            pidfile = command[command.index("--pidfile") + 1]
            with open(pidfile, "w") as pidfile_handle:
                pidfile_handle.write(f"{os.getpid()}\n")

    def fail_cuda_restore(*args, **kwargs):
        raise RuntimeError("restore failed")

    monkeypatch.setattr(provider, "_run", run)
    monkeypatch.setattr(provider, "_cuda_pids", lambda *args, **kwargs: [123])
    monkeypatch.setattr(provider, "_run_cuda", fail_cuda_restore)
    monkeypatch.setattr(provider, "_kill_tree", killed.append)

    with pytest.raises(RuntimeError, match="restore failed"):
        provider.restore(snapshot_dir)

    assert killed == [os.getpid()]
    assert not list(snapshot_dir.glob("restore-work-*"))


@pytest.mark.parametrize("cuda_pids", ([], [123, 456]))
def test_criu_provider_requires_one_cuda_process(monkeypatch, cuda_pids):
    provider = criu_provider()
    monkeypatch.setattr(provider, "_cuda_pids", lambda *args, **kwargs: cuda_pids)

    with pytest.raises(RuntimeError, match=f"found {len(cuda_pids)}"):
        provider._cuda_pid(999)


def test_criu_capture_reports_cuda_rollback_failure(tmp_path, monkeypatch):
    provider = criu_provider()
    actions = []

    def run_cuda(action, pid, *extra):
        actions.append(action)
        if action == "checkpoint":
            raise RuntimeError("checkpoint failed")
        if action == "restore":
            raise RuntimeError("restore failed")

    monkeypatch.setattr(provider, "_cuda_pid", lambda root_pid: root_pid)
    monkeypatch.setattr(provider, "_run_cuda", run_cuda)

    with pytest.raises(RuntimeError, match="rollback cleanup failed: restore"):
        provider.capture(123, tmp_path / "snapshot")

    assert actions == ["lock", "checkpoint", "restore", "unlock"]


def test_criu_capture_skips_in_flight_connections(tmp_path, monkeypatch):
    provider = criu_provider()
    commands = []
    monkeypatch.setattr(provider, "_cuda_pid", lambda root_pid: root_pid)
    monkeypatch.setattr(provider, "_run_cuda", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        provider, "_run", lambda command, **kwargs: commands.append(command)
    )

    provider.capture(123, tmp_path / "snapshot")

    assert "--skip-in-flight" in commands[0]
