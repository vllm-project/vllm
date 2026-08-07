# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import builtins
import dataclasses
import importlib.metadata
import importlib.util
import json
import socket
import stat
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.snapshot.server import (
    ListenerConfig,
    Oracle,
    SnapshotBarrierError,
    SnapshotCanaryError,
    oracle_from_request_output,
    parse_control_args,
    read_release_marker,
    run_snapshot_child,
    wait_for_release_marker,
    write_ready_atomic,
)
from vllm_cli.snapshot.controller import (
    LocalSnapshotTools,
    ProcessInventory,
    SnapshotCreateError,
    SnapshotRestoreError,
    _TcpSocketRecord,
    _validate_tcp_connections,
    create_snapshot,
    restore_snapshot,
)
from vllm_cli.snapshot.manifest import (
    SnapshotCompatibilityError,
    SnapshotManifest,
    SocketIdentity,
    read_manifest,
    write_manifest_atomic,
)


@pytest.mark.asyncio
async def test_snapshot_child_waits_before_http_bind():
    events: list[str] = []
    engine = object()

    @asynccontextmanager
    async def engine_context():
        events.append("engine-ready")
        yield engine

    async def run_canary(actual_engine: object) -> Oracle:
        assert actual_engine is engine
        return Oracle(token_ids=(12095,), text=" Paris")

    def write_ready(oracle: Oracle) -> None:
        assert oracle.token_ids == (12095,)
        events.append("ready-written")

    async def wait_for_release() -> None:
        events.append("barrier-wait")
        events.append("barrier-release")

    async def bind_and_serve(actual_engine: object) -> None:
        assert actual_engine is engine
        events.append("http-bound")

    await run_snapshot_child(
        engine_context=engine_context(),
        run_canary=run_canary,
        prepare_snapshot=lambda: events.append("streams-detached"),
        write_ready=write_ready,
        wait_for_release=wait_for_release,
        bind_and_serve=bind_and_serve,
    )

    assert events == [
        "engine-ready",
        "streams-detached",
        "ready-written",
        "barrier-wait",
        "barrier-release",
        "http-bound",
    ]


@pytest.mark.asyncio
async def test_empty_canary_prevents_http_bind():
    bound = False

    @asynccontextmanager
    async def engine_context():
        yield object()

    async def run_canary(_engine: object) -> Oracle:
        return Oracle(token_ids=(), text="")

    async def bind_and_serve(_engine: object) -> None:
        nonlocal bound
        bound = True

    with pytest.raises(SnapshotCanaryError, match="no token"):
        await run_snapshot_child(
            engine_context=engine_context(),
            run_canary=run_canary,
            prepare_snapshot=lambda: None,
            write_ready=lambda _oracle: None,
            wait_for_release=_return_none,
            bind_and_serve=bind_and_serve,
        )

    assert not bound


@pytest.mark.asyncio
async def test_barrier_failure_prevents_http_bind():
    bound = False

    @asynccontextmanager
    async def engine_context():
        yield object()

    async def wait_for_release() -> None:
        raise RuntimeError("malformed release marker")

    async def bind_and_serve(_engine: object) -> None:
        nonlocal bound
        bound = True

    with pytest.raises(RuntimeError, match="malformed release marker"):
        await run_snapshot_child(
            engine_context=engine_context(),
            run_canary=lambda _engine: _return_oracle(),
            prepare_snapshot=lambda: None,
            write_ready=lambda _oracle: None,
            wait_for_release=wait_for_release,
            bind_and_serve=bind_and_serve,
        )

    assert not bound


async def _return_none() -> None:
    return None


async def _return_oracle() -> Oracle:
    return Oracle(token_ids=(12095,), text=" Paris")


def test_ready_marker_is_atomic_and_private(tmp_path: Path):
    ready_path = tmp_path / "ready.json"

    write_ready_atomic(ready_path, Oracle(token_ids=(12095,), text=" Paris"))

    assert json.loads(ready_path.read_text()) == {
        "token_ids": [12095],
        "text": " Paris",
    }
    assert stat.S_IMODE(ready_path.stat().st_mode) == 0o600
    assert not (tmp_path / "ready.json.tmp").exists()


def test_release_marker_supplies_listener(tmp_path: Path):
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps({"release": True, "host": "127.0.0.1", "port": 9000})
    )

    assert read_release_marker(release_path) == ListenerConfig(
        host="127.0.0.1", port=9000
    )


@pytest.mark.asyncio
async def test_release_wait_does_not_count_time_spent_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    release_path = tmp_path / "release.json"

    class FrozenLoop:
        def __init__(self) -> None:
            self.times = iter((0.0, 3600.0))

        def time(self) -> float:
            return next(self.times)

    async def release_after_poll(_delay: float) -> None:
        release_path.write_text(
            json.dumps({"release": True, "host": "127.0.0.1", "port": 9000})
        )

    monkeypatch.setattr("vllm.snapshot.server.asyncio.get_running_loop", FrozenLoop)
    monkeypatch.setattr("vllm.snapshot.server.asyncio.sleep", release_after_poll)

    listener = await wait_for_release_marker(
        release_path,
        timeout_s=1.0,
        poll_interval_s=0.05,
    )

    assert listener == ListenerConfig(host="127.0.0.1", port=9000)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"release": False, "host": "127.0.0.1", "port": 9000},
        {"release": True, "host": "127.0.0.1", "port": 0},
        {"release": True, "host": 4, "port": 9000},
    ],
)
def test_malformed_release_marker_is_rejected(
    tmp_path: Path, payload: dict[str, object]
):
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(payload))

    with pytest.raises(SnapshotBarrierError, match="release marker"):
        read_release_marker(release_path)


def test_request_output_becomes_exact_oracle():
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[12095], text=" Paris")]
    )

    assert oracle_from_request_output(output) == Oracle(
        token_ids=(12095,), text=" Paris"
    )


def test_child_control_flags_are_removed_from_vllm_args():
    control, remaining = parse_control_args(
        [
            "--ready-file",
            "ready.json",
            "--release-file",
            "release.json",
            "--release-timeout-s",
            "123",
            "--",
            "Qwen/Qwen3-0.6B",
            "--dtype",
            "float16",
        ]
    )

    assert control.ready_file == Path("ready.json")
    assert control.release_file == Path("release.json")
    assert control.release_timeout_s == 123
    assert remaining == ["Qwen/Qwen3-0.6B", "--dtype", "float16"]


class FakeSnapshotTools:
    def __init__(self):
        self.events: list[str] = []
        self.fail_at: str | None = None
        self.identity_change: dict[str, object] = {}
        self.launch_workdir: Path | None = None

    def preflight(self, _action: str, _artifact: Path) -> None:
        self.events.append("preflight")

    def launch_child(self, workdir: Path, _engine_argv: tuple[str, ...]):
        self.events.append("launch-child")
        self.launch_workdir = workdir
        assert not (workdir / "manifest.json").exists()
        (workdir / "child.log").touch()
        return 100

    def wait_ready(self, _workdir: Path, _root_pid: int) -> Oracle:
        self.events.append("wait-ready")
        return Oracle(token_ids=(12095,), text=" Paris")

    def inventory(self, root_pid: int) -> ProcessInventory:
        self.events.append("inventory")
        return ProcessInventory(
            root_pid=root_pid,
            process_tree=(root_pid, 101),
            cuda_holders=(101,),
            sockets=(),
        )

    def dump(self, workdir: Path, _inventory: ProcessInventory) -> None:
        self.events.append("criu-dump")
        images = workdir / "images"
        images.mkdir()
        (images / "pages.img").write_bytes(b"snapshot")

    def verify_dead(self, _inventory: ProcessInventory) -> None:
        self.events.append("verify-dead")

    def make_manifest(
        self,
        _args: argparse.Namespace,
        _engine_argv: tuple[str, ...],
        inventory: ProcessInventory,
        oracle: Oracle,
        _workdir: Path,
    ) -> SnapshotManifest:
        return _controller_manifest(
            process_tree=inventory.process_tree,
            cuda_holders=inventory.cuda_holders,
            oracle_token_ids=oracle.token_ids,
            oracle_text=oracle.text,
        )

    def publish(self, workdir: Path, target: Path, manifest: SnapshotManifest) -> None:
        self.events.append("publish")
        assert workdir == target
        write_manifest_atomic(workdir, manifest)

    def current_identity(self, manifest: SnapshotManifest) -> SnapshotManifest:
        return dataclasses.replace(manifest, **self.identity_change)

    def restore(self, _artifact: Path) -> int:
        self.events.append("criu-restore")
        return 100

    def release(self, _artifact: Path, _host: str | None, _port: int) -> None:
        self.events.append("release-barrier")
        if self.fail_at == "release-barrier":
            raise RuntimeError("release failed")

    def wait_listener(self, _root_pid: int, _host: str | None, _port: int) -> None:
        self.events.append("wait-listener")

    def request_oracle(
        self, _host: str | None, _port: int, _manifest: SnapshotManifest
    ) -> Oracle:
        self.events.append("request-oracle")
        return Oracle(token_ids=(12095,), text=" Paris")

    def cleanup(self, _root_pid: int, _manifest: SnapshotManifest) -> None:
        self.events.extend(["kill-process-group", "kill-cuda", "verify-clear"])


def _controller_manifest(**changes: object) -> SnapshotManifest:
    manifest = SnapshotManifest(
        schema_version=1,
        boundary="post-engine-init-pre-http-bind",
        complete=True,
        created_at="2026-08-06T00:00:00Z",
        artifact_bytes=8,
        source_revision="source-sha",
        binary_revision="binary-sha",
        python_version="3.12.3",
        torch_version="2.9.0",
        cuda_runtime="12.9",
        driver_version="575.57.08",
        criu_version="4.1",
        cuda_checkpoint_version="575.57.08",
        kernel_release="6.8.0",
        host_id="host-a",
        gpu_name="NVIDIA A10",
        gpu_uuid="GPU-abc",
        model="Qwen/Qwen3-0.6B",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        engine_args=(("tensor_parallel_size", 1),),
        environment=(("VLLM_USE_V1", "1"),),
        process_tree=(100, 101),
        cuda_holders=(101,),
        socket_inventory=(),
        oracle_token_ids=(12095,),
        oracle_text=" Paris",
    )
    return dataclasses.replace(manifest, **changes)


def test_create_publishes_only_after_complete_dump(tmp_path: Path):
    target = tmp_path / "snapshot"
    tools = FakeSnapshotTools()

    create_snapshot(
        argparse.Namespace(snapshot_dir=str(target), model_tag="Qwen/Qwen3-0.6B"),
        engine_argv=("Qwen/Qwen3-0.6B",),
        tools=tools,
    )

    assert read_manifest(target).complete
    assert tools.launch_workdir == target
    assert tools.events == [
        "preflight",
        "launch-child",
        "wait-ready",
        "inventory",
        "criu-dump",
        "verify-dead",
        "publish",
    ]


def test_restore_mismatch_stops_before_criu(tmp_path: Path):
    target = tmp_path / "snapshot"
    target.mkdir(mode=0o700)
    write_manifest_atomic(target, _controller_manifest())
    tools = FakeSnapshotTools()
    tools.identity_change = {"model_revision": "different"}

    with pytest.raises(SnapshotCompatibilityError, match="model_revision"):
        restore_snapshot(
            argparse.Namespace(snapshot_dir=str(target), host="127.0.0.1", port=9000),
            tools=tools,
        )

    assert "criu-restore" not in tools.events


def test_partial_restore_is_killed_and_cuda_is_clear(tmp_path: Path):
    target = tmp_path / "snapshot"
    target.mkdir(mode=0o700)
    write_manifest_atomic(target, _controller_manifest())
    tools = FakeSnapshotTools()
    tools.fail_at = "release-barrier"

    with pytest.raises(SnapshotRestoreError, match="release failed"):
        restore_snapshot(
            argparse.Namespace(snapshot_dir=str(target), host="127.0.0.1", port=9000),
            tools=tools,
        )

    assert tools.events[-3:] == [
        "kill-process-group",
        "kill-cuda",
        "verify-clear",
    ]


def test_failed_child_is_reaped_and_reported(tmp_path: Path):
    class FailedProcess:
        returncode = 17

        def poll(self) -> int:
            return self.returncode

    (tmp_path / "child.log").write_text("engine init failed")
    tools = LocalSnapshotTools()
    tools._children[100] = FailedProcess()  # type: ignore[assignment]

    with pytest.raises(SnapshotCreateError, match="engine init failed"):
        tools.wait_ready(tmp_path, 100)

    assert 100 not in tools._children


def test_tcp_inventory_accepts_connections_owned_by_the_snapshot_tree():
    records = (
        _TcpSocketRecord(
            family="AF_INET6",
            local_raw="0000000000000000FFFF00003C00000A:C262",
            remote_raw="0000000000000000FFFF00003C00000A:D6F6",
            inode=41,
        ),
        _TcpSocketRecord(
            family="AF_INET6",
            local_raw="0000000000000000FFFF00003C00000A:D6F6",
            remote_raw="0000000000000000FFFF00003C00000A:C262",
            inode=42,
        ),
    )

    assert _validate_tcp_connections(records, {41, 42}) == (
        SocketIdentity(
            family="AF_INET6",
            socket_type="SOCK_STREAM",
            local_address="[::ffff:10.0.0.60]:49762",
            remote_address="[::ffff:10.0.0.60]:55030",
            state="ESTABLISHED",
        ),
        SocketIdentity(
            family="AF_INET6",
            socket_type="SOCK_STREAM",
            local_address="[::ffff:10.0.0.60]:55030",
            remote_address="[::ffff:10.0.0.60]:49762",
            state="ESTABLISHED",
        ),
    )


def test_tcp_inventory_rejects_an_external_established_connection():
    records = (
        _TcpSocketRecord(
            family="AF_INET",
            local_raw="3C00000A:D6F6",
            remote_raw="01010101:01BB",
            inode=41,
        ),
    )

    with pytest.raises(SnapshotCreateError, match="external established TCP"):
        _validate_tcp_connections(records, {41})


def test_manifest_records_installed_binary_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tools = LocalSnapshotTools()
    original_read_text = Path.read_text
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "binary-build")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "host-a"
            if path == Path("/etc/machine-id")
            else original_read_text(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(tools, "_source_revision", lambda: "source-sha")
    monkeypatch.setattr(
        tools,
        "_gpu_identity",
        lambda: ("NVIDIA A10", "GPU-abc", "575.57.08"),
    )
    monkeypatch.setattr(tools, "_version", lambda _command: "4.1")
    monkeypatch.setattr(tools, "_sha256", lambda _path: "checkpoint-sha")
    monkeypatch.setattr(tools, "_environment_identity", lambda: ())

    manifest = tools.make_manifest(
        argparse.Namespace(
            model_tag="Qwen/Qwen3-0.6B",
            revision="model-sha",
            tokenizer_revision="tokenizer-sha",
        ),
        ("Qwen/Qwen3-0.6B",),
        ProcessInventory(
            root_pid=100,
            process_tree=(100, 101),
            cuda_holders=(101,),
            sockets=(),
        ),
        Oracle(token_ids=(12095,), text=" Paris"),
        tmp_path,
    )

    assert manifest.source_revision == "source-sha"
    assert manifest.binary_revision == "binary-build"


def test_torch_identity_does_not_import_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch_package = tmp_path / "torch"
    torch_package.mkdir()
    (torch_package / "version.py").write_text('cuda = "13.0"\n')
    tools = LocalSnapshotTools()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "2.9.0")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(torch_package)]),
    )
    real_import = builtins.__import__

    def reject_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"snapshot identity imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch_import)

    assert tools._torch_identity() == ("2.9.0", "13.0")


def test_source_revision_does_not_import_vllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package = tmp_path / "vllm"
    package.mkdir()
    package_init = package / "__init__.py"
    package_init.write_text("")
    tools = LocalSnapshotTools()
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(package_init)),
    )
    monkeypatch.setattr(
        tools,
        "_run",
        lambda command, **_kwargs: SimpleNamespace(
            stdout="source-sha\n" if "rev-parse" in command else ""
        ),
    )
    real_import = builtins.__import__

    def reject_vllm_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError(f"snapshot identity imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm_import)

    assert tools._source_revision() == "source-sha"


def test_source_revision_rejects_a_dirty_editable_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package = tmp_path / "vllm"
    package.mkdir()
    package_init = package / "__init__.py"
    package_init.write_text("")
    tools = LocalSnapshotTools()
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(package_init)),
    )

    def fake_run(command: list[str], **_kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(stdout="source-sha\n")
        return SimpleNamespace(stdout=" M vllm/config.py\n")

    monkeypatch.setattr(tools, "_run", fake_run)

    with pytest.raises(SnapshotCompatibilityError, match="dirty checkout"):
        tools._source_revision()


def test_restore_resets_child_log_to_captured_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "snapshot"
    artifact.mkdir()
    child_log = artifact / "child.log"
    child_log.write_bytes(b"startup log")
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    tools = LocalSnapshotTools()
    tools.shm_dir = shm_dir

    def fake_dump(action: str, _artifact: Path, arguments: list[str]) -> None:
        assert action == "dump"
        assert "--tcp-established" in arguments
        (shm_dir / "link_remap.270").write_bytes(b"semaphore state")

    monkeypatch.setattr(tools, "_criu", fake_dump)
    tools.dump(
        artifact,
        ProcessInventory(
            root_pid=100,
            process_tree=(100, 101),
            cuda_holders=(101,),
            sockets=(),
        ),
    )
    assert (artifact / "link-remaps/link_remap.270").read_bytes() == (
        b"semaphore state"
    )
    assert not (shm_dir / "link_remap.270").exists()
    child_log.write_bytes(b"startup log\nrestored output")

    def fake_restore(action: str, artifact: Path, arguments: list[str]) -> None:
        assert action == "restore"
        assert "--tcp-established" in arguments
        assert child_log.read_bytes() == b"startup log"
        assert (shm_dir / "link_remap.270").read_bytes() == b"semaphore state"
        (artifact / "restored.pid").write_text("100")

    monkeypatch.setattr(tools, "_criu", fake_restore)
    monkeypatch.setattr(
        tools,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="100"),
    )

    assert tools.restore(artifact) == 100


def test_release_refuses_an_occupied_listener(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    tools = LocalSnapshotTools()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        with pytest.raises(SnapshotRestoreError, match="already in use"):
            tools.release(artifact, "127.0.0.1", port)

    assert not (artifact / "release.json").exists()


def test_ipv6_wildcard_uses_loopback_for_correctness_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    tools = LocalSnapshotTools()
    requested_urls: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"token_ids":[12095],"text":" Paris"}]}'

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
        assert timeout == 120
        requested_urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert tools.request_oracle("::", 8000, _controller_manifest()) == Oracle(
        token_ids=(12095,), text=" Paris"
    )
    assert requested_urls == ["http://[::1]:8000/v1/completions"]
