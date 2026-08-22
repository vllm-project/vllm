# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm launch` CLI subcommand."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec, patch

import pytest

import vllm.snapshot.runtime as snapshot_runtime
import vllm.snapshot.server as snapshot_server
from vllm.entrypoints.cli import snapshot as snapshot_cli
from vllm.entrypoints.cli.launch import (
    LaunchSubcommand,
    RenderSubcommand,
    cmd_init,
)
from vllm.snapshot.controller import (
    LocalSnapshotTools,
    SnapshotCreateError,
    SnapshotRestoreError,
    create_snapshot,
    restore_snapshot,
)
from vllm.snapshot.manifest import (
    SnapshotCompatibilityError,
    SnapshotManifest,
    SnapshotRuntimeIdentity,
    SnapshotSecurityError,
    read_manifest,
    validate_artifact_root,
    write_manifest_atomic,
)
from vllm.snapshot.runtime import ProcessInventory, _TcpSocketRecord
from vllm.snapshot.server import SnapshotCanaryError
from vllm.snapshot.types import Oracle, oracles_match
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture
def launch_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    LaunchSubcommand().subparser_init(subparsers)
    return parser


def test_subcommand_name():
    assert LaunchSubcommand().name == "launch"


def test_cmd_init_returns_subcommand():
    result = cmd_init()
    assert len(result) == 1
    assert isinstance(result[0], LaunchSubcommand)


# -- Parsing: `vllm launch render` --


def test_parse_launch_render(launch_parser):
    args = launch_parser.parse_args(["launch", "render", "--model", "test-model"])
    assert args.launch_component == "render"


def test_parse_launch_requires_component(launch_parser):
    with pytest.raises(SystemExit):
        launch_parser.parse_args(["launch", "--model", "test-model"])


def test_parse_launch_invalid_component(launch_parser):
    with pytest.raises(SystemExit):
        launch_parser.parse_args(["launch", "unknown", "--model", "test-model"])


# -- Dispatch --


def test_cmd_launch_render_calls_run():
    args = argparse.Namespace(model_tag=None, model="test-model")
    with patch("vllm.entrypoints.cli.launch.uvloop.run") as mock_uvloop_run:
        RenderSubcommand.cmd(args)
        mock_uvloop_run.assert_called_once()


def test_cmd_launch_model_tag_overrides():
    args = argparse.Namespace(
        model_tag="tag-model",
        model="original-model",
        launch_command=lambda a: None,
    )
    LaunchSubcommand.cmd(args)
    assert args.model == "tag-model"


def test_cmd_launch_model_tag_none():
    args = argparse.Namespace(
        model_tag=None,
        model="original-model",
        launch_command=lambda a: None,
    )
    LaunchSubcommand.cmd(args)
    assert args.model == "original-model"


def test_cmd_dispatches():
    called = {}

    def fake_dispatch(args):
        called["args"] = args

    args = argparse.Namespace(launch_command=fake_dispatch)
    LaunchSubcommand.cmd(args)
    assert "args" in called


# -- Module registration --


def test_subparser_init_returns_parser():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    result = LaunchSubcommand().subparser_init(subparsers)
    assert isinstance(result, FlexibleArgumentParser)


def test_launch_registered_in_main():
    """Verify that launch module is importable as a CLI module."""
    import vllm.entrypoints.cli.launch as launch_module

    assert hasattr(launch_module, "cmd_init")
    subcmds = launch_module.cmd_init()
    assert any(s.name == "launch" for s in subcmds)


_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


_RUNTIME_MODULES = (
    "torch",
    "uvloop",
    "vllm.env_override",
    "vllm.entrypoints.cli.serve",
    "vllm.entrypoints.openai.cli_args",
    "vllm.snapshot.controller",
    "vllm.snapshot.server",
    "vllm.utils.argparse_utils",
    "vllm.v1.executor",
    "vllm.v1.worker.gpu_model_runner",
)


def _run_import_light_help(argv: list[str]) -> str:
    script = f"""
import os
import sys

os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
sys.argv = ["vllm", *{argv!r}]
from vllm.entrypoints.cli.main import main

try:
    main()
except SystemExit as exc:
    assert exc.code == 0

assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ
loaded = sorted(
    prefix
    for prefix in {_RUNTIME_MODULES!r}
    if any(
        name == prefix or name.startswith(f"{{prefix}}.") for name in sys.modules
    )
)
assert not loaded, loaded
"""
    return _run_python("-c", script).stdout


def _run_cli(argv: list[str]) -> None:
    from vllm.entrypoints.cli import main as cli_main

    with patch.object(sys, "argv", ["vllm", *argv]):
        cli_main.main()


def test_snapshot_environment_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    from vllm.entrypoints.cli import main as cli_main

    for name in ("cli_env_setup", "apply_runtime_environment"):
        monkeypatch.setattr(cli_main, name, pytest.fail)
    secret = "snapshot-secret"
    monkeypatch.setenv("VLLM_API_KEY", secret)
    monkeypatch.setenv("VLLM_USER_SETTING", secret)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", secret)
    environment = dict(LocalSnapshotTools()._environment_identity())
    assert (
        environment["VLLM_USER_SETTING"]
        == hashlib.sha256(b"VLLM_USER_SETTING\0snapshot-secret").hexdigest()
    )
    assert (
        environment["VLLM_USER_SETTING"] != environment["VLLM_WORKER_MULTIPROC_METHOD"]
    )
    assert secret not in str(environment)
    assert "VLLM_API_KEY" not in environment

    monkeypatch.setenv("VLLM_USER_SETTING", f"{secret}-changed")
    changed_environment = dict(LocalSnapshotTools()._environment_identity())
    assert changed_environment["VLLM_USER_SETTING"] != environment["VLLM_USER_SETTING"]

    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(artifact, _manifest(environment=tuple(environment.items())))
    before = os.environ.copy()

    _run_cli(["snapshot", "inspect", str(artifact)])

    assert os.environ == before
    output = capsys.readouterr().out
    assert secret not in output
    inspected = json.loads(output)
    inspected_environment = dict(inspected["environment"])
    assert inspected_environment == environment

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "current_identity", lambda _uuid: _runtime_identity())
    manifest = tools.make_manifest(
        argparse.Namespace(
            hf_token=secret,
            model_tag="Qwen/Qwen3-0.6B",
            revision=_MODEL_REVISION,
            served_model_name=None,
            tokenizer_revision=None,
        ),
        (
            "Qwen/Qwen3-0.6B",
            "--api-key",
            secret,
            f"{secret}-alternate",
            f"--api-key={secret}",
            "--hf-token",
            secret,
            f"--hf-token={secret}",
        ),
        ProcessInventory(100, (100, 101), (101,), "GPU-abc"),
        _oracle(),
        tmp_path,
    )

    assert manifest.engine_argv == (
        "Qwen/Qwen3-0.6B",
        "--api-key",
        "***",
        "***",
        "--api-key=***",
        "--hf-token",
        "***",
        "--hf-token=***",
    )


def parse_snapshot(*argv: str):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser", required=True)
    snapshot_cli.SnapshotSubcommand(
        create_requested=argv[:1] == ("create",)
    ).subparser_init(subparsers)
    return parser.parse_args(["snapshot", *argv])


def test_snapshot_help_stays_lazy():
    output = _run_import_light_help(["snapshot", "restore", "--help"])
    assert "snapshot" in output.lower()


def test_snapshot_create_cli_accepts_only_pinned_compact_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(snapshot_cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(snapshot_cli.platform, "machine", lambda: "x86_64")
    base = ("create", "Qwen/Qwen3-0.6B", "--snapshot-dir=/tmp/snapshot")
    compact = parse_snapshot(*base, "--revision", _MODEL_REVISION)

    snapshot_cli.validate_create_args(compact)
    assert compact.model_tag == "Qwen/Qwen3-0.6B"


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("prefill_context_parallel_size", 2, "prefill context parallel size 1"),
        ("model_tag", None, "immutable remote model"),
        ("logprobs_mode", "raw_logits", "log-probability mode"),
        ("speculative_config", {}, "does not support speculative decoding"),
    ],
)
def test_snapshot_create_rejects_invalid_config(
    attribute: str,
    value: object,
    message: str,
    tmp_path: Path,
):
    args = parse_snapshot(
        "create",
        "Qwen/Qwen3-0.6B",
        "--snapshot-dir=/tmp/snapshot",
        "--revision",
        _MODEL_REVISION,
    )
    if attribute == "model_tag":
        local_model = tmp_path / "model"
        local_model.mkdir()
        value = str(local_model)
    setattr(args, attribute, value)
    with pytest.raises(ValueError, match=message):
        snapshot_cli.validate_create_args(args)


def _oracle(
    token_ids: tuple[int, ...] = (12095,),
    text: str = " Paris",
    logprob: float = -0.125,
) -> Oracle:
    return Oracle(token_ids, text, logprob)


def _manifest(**changes: object) -> SnapshotManifest:
    manifest = SnapshotManifest(
        schema_version=1,
        boundary="post-engine-init-reloadable-state-released",
        created_at="2026-08-06T00:00:00Z",
        artifact_bytes=8,
        vllm_version="0.1.dev1+snapshot",
        python_version="3.12.3",
        torch_version="2.9.0",
        cuda_runtime="12.9",
        driver_version="575.57.08",
        criu_version="4.1",
        cuda_checkpoint_sha256="a" * 64,
        kernel_release="6.8.0",
        host_id="host-a",
        gpu_name="NVIDIA A10",
        gpu_uuid="GPU-abc",
        model="Qwen/Qwen3-0.6B",
        served_model_name="Qwen/Qwen3-0.6B",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        engine_argv=("Qwen/Qwen3-0.6B",),
        environment=(("VLLM_USE_V1", "1"),),
        process_tree=(100, 101),
        cuda_holders=(101,),
        oracle_token_ids=(12095,),
        oracle_text=" Paris",
        oracle_sampled_token_logprob=-0.125,
    )
    return SnapshotManifest.model_validate(
        {**manifest.model_dump(mode="python"), **changes}, strict=True
    )


def _runtime_identity(**changes: object) -> Any:
    fields = SnapshotRuntimeIdentity.model_fields.keys()
    payload = _manifest().model_dump(mode="python", include=fields)
    return SnapshotRuntimeIdentity.model_validate({**payload, **changes}, strict=True)


def _fake_snapshot_tools() -> Any:
    tools = create_autospec(LocalSnapshotTools, instance=True)
    tools.launch_child.return_value = 100
    tools.wait_ready.return_value = _oracle()
    tools.inventory.return_value = ProcessInventory(100, (100, 101), (101,), "GPU-abc")

    tools.current_identity.side_effect = lambda _gpu_uuid: _runtime_identity()
    tools.restore.return_value = 100
    tools.request_oracle.return_value = _oracle()
    return tools


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_snapshot_timeout_must_be_positive_and_finite(
    value: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VLLM_SNAPSHOT_TIMEOUT_S", value)
    with pytest.raises(ValueError, match="positive finite number"):
        LocalSnapshotTools()


def test_snapshot_create_abort_reaps_after_process_group_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tools = LocalSnapshotTools()
    process = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(1)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    tools._children[process.pid] = process
    (tmp_path / "child.log").write_text("root failed")
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )
    monkeypatch.setattr(os, "waitid", lambda *_args: object(), raising=False)

    try:
        with pytest.raises(SnapshotCreateError, match="root failed"):
            tools.wait_ready(tmp_path, process.pid)
        assert process.returncode is None
        tools.abort_create(process.pid)
    finally:
        if process.pid in tools._children:
            tools.abort_create(process.pid)

    assert killed == [(process.pid, signal.SIGKILL)]
    assert process.pid not in tools._children


def test_snapshot_create_abort_rejects_surviving_process_group(
    monkeypatch: pytest.MonkeyPatch,
):
    tools = LocalSnapshotTools()
    process = create_autospec(subprocess.Popen, instance=True)
    process.wait.side_effect = subprocess.TimeoutExpired("snapshot child", 10)
    tools._children[100] = process
    monkeypatch.setattr(os, "killpg", lambda _pid, _sig: None)

    with pytest.raises(SnapshotCreateError, match="survived SIGKILL"):
        tools.abort_create(100)


@pytest.mark.asyncio
async def test_snapshot_child_rejects_mismatch_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from vllm.entrypoints.launchers.api_server import entry as api_server

    canaries = iter((_oracle(), _oracle((42,), " Lyon")))

    @asynccontextmanager
    async def engine_context():
        yield object()

    async def next_canary(_engine: object) -> Oracle:
        return next(canaries)

    async def return_none(_engine: object) -> None:
        return None

    monkeypatch.setattr(
        api_server, "build_async_engine_client", lambda _args: engine_context()
    )
    monkeypatch.setattr(snapshot_server, "run_engine_canary", next_canary)
    monkeypatch.setattr(snapshot_server, "_release_reloadable_state", return_none)
    monkeypatch.setattr(snapshot_server, "_restore_reloadable_state", return_none)
    monkeypatch.setattr(snapshot_server, "detach_snapshot_streams", lambda: None)
    monkeypatch.setattr(
        snapshot_server,
        "write_ready_atomic",
        lambda *_args: pytest.fail("mismatched snapshot was published"),
    )
    control = snapshot_server.ControlArgs(
        ready_file=tmp_path / "ready.json",
        release_file=tmp_path / "release.json",
        release_timeout_s=1,
    )
    with pytest.raises(SnapshotCanaryError, match="rehearsal changed"):
        await snapshot_server.run_vllm_snapshot_child(control, argparse.Namespace())


def test_snapshot_create_rolls_back_failed_dump(tmp_path: Path):
    target = tmp_path / "snapshot"
    tools = _fake_snapshot_tools()
    tools.dump.side_effect = RuntimeError("dump failed")
    with pytest.raises(RuntimeError, match="dump failed"):
        create_snapshot(
            argparse.Namespace(snapshot_dir=str(target), model_tag="Qwen/Qwen3-0.6B"),
            tools=tools,
        )
    tools.abort_create.assert_called_once_with(100)
    assert not target.exists()


def _restore_fixture(tmp_path: Path) -> argparse.Namespace:
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(artifact, _manifest())
    return argparse.Namespace(snapshot_dir=str(artifact), host="127.0.0.1", port=9000)


def test_snapshot_restore_rejects_identity_mismatch(tmp_path: Path):
    args = _restore_fixture(tmp_path)

    identity_mismatch = _fake_snapshot_tools()
    identity_mismatch.current_identity.side_effect = lambda _gpu_uuid: (
        _runtime_identity(vllm_version="different")
    )
    with pytest.raises(SnapshotCompatibilityError, match="vllm_version"):
        restore_snapshot(args, tools=identity_mismatch)
    identity_mismatch.restore.assert_not_called()


def test_snapshot_restore_cleans_oracle_mismatch(tmp_path: Path):
    args = _restore_fixture(tmp_path)
    oracle_mismatch = _fake_snapshot_tools()
    oracle_mismatch.request_oracle.return_value = _oracle(logprob=-0.5)
    with pytest.raises(SnapshotRestoreError, match="oracle mismatch"):
        restore_snapshot(args, tools=oracle_mismatch)
    oracle_mismatch.cleanup.assert_called_once()
    oracle_mismatch.complete_restore.assert_not_called()
    assert oracles_match(_oracle(logprob=-100.0), _oracle(logprob=-100.0005))
    assert not oracles_match(_oracle(logprob=-100.0), _oracle(logprob=-100.002))


@pytest.mark.parametrize(
    ("invalid_update", "diagnostic"),
    [
        ({"unexpected": True}, "unexpected"),
        ({"schema_version": True}, "schema_version"),
        ({"boundary": "unknown"}, "boundary"),
        ({"process_tree": [100, 100]}, "root"),
        ({"cuda_holders": [999]}, "root"),
        ({"oracle_token_ids": [1, 2]}, "oracle_token_ids"),
        ({"oracle_sampled_token_logprob": 0}, "oracle_sampled_token_logprob"),
        (
            {"oracle_sampled_token_logprob": float("nan")},
            "oracle_sampled_token_logprob",
        ),
    ],
)
def test_snapshot_manifest_validation(
    tmp_path: Path,
    invalid_update: dict[str, object],
    diagnostic: str,
):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(artifact, _manifest())
    invalid = _manifest().model_dump(mode="json")
    invalid.update(invalid_update)
    (artifact / "manifest.json").write_text(json.dumps(invalid))
    with pytest.raises(
        SnapshotCompatibilityError,
        match=f"invalid snapshot manifest: {diagnostic}",
    ):
        read_manifest(artifact)


def _local_restore_artifact(tmp_path: Path, name: str = "snapshot") -> Path:
    artifact = tmp_path / name
    artifact.mkdir(mode=0o700)
    (artifact / "child.log").write_bytes(b"startup log")
    (artifact / "child.log.snapshot-size").write_text("11\n")
    return artifact


def _mark_snapshot_pids_free(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    probed_pids: list[int] = []

    def missing(pid: int, _signal: int) -> None:
        probed_pids.append(pid)
        raise ProcessLookupError

    monkeypatch.setattr(snapshot_runtime.os, "kill", missing)
    return probed_pids


@pytest.mark.parametrize(
    ("probe_error", "diagnostic"),
    [
        (None, "already occupied: 100"),
        (PermissionError(), "already occupied: 100"),
        (OSError("probe failed"), "availability probe failed: 100"),
    ],
)
def test_snapshot_restore_rejects_unavailable_pid_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_error: OSError | None,
    diagnostic: str,
):
    args = _restore_fixture(tmp_path)
    artifact = Path(args.snapshot_dir)
    (artifact / "child.log").write_bytes(b"startup log")
    (artifact / "release.json").write_bytes(b"release sentinel")
    before = {path.name: path.read_bytes() for path in artifact.iterdir()}
    tools = LocalSnapshotTools()

    def fail_probe(_pid: int, _signal: int) -> None:
        if probe_error is not None:
            raise probe_error

    monkeypatch.setattr(snapshot_runtime.os, "kill", fail_probe)
    monkeypatch.setattr(tools, "preflight", lambda *_args: None)
    monkeypatch.setattr(tools, "current_identity", lambda _uuid: _runtime_identity())
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))

    with pytest.raises(SnapshotRestoreError, match=diagnostic):
        restore_snapshot(args, tools=tools)
    assert {path.name: path.read_bytes() for path in artifact.iterdir()} == before


def test_snapshot_restore_never_signals_an_unpinned_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _mark_snapshot_pids_free(monkeypatch)
    artifact = _local_restore_artifact(tmp_path)
    tools = LocalSnapshotTools()
    signaled: list[int] = []
    monkeypatch.setattr(tools, "cleanup", signaled.append)
    monkeypatch.setattr(
        tools,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, (artifact / "restored.pid").read_text(), ""
        ),
    )

    def restore(_action: str, root: Path, _arguments: list[str]) -> None:
        (root / "restored.pid").write_text("333\n")

    monkeypatch.setattr(tools, "_criu", restore)
    with pytest.raises(SnapshotRestoreError, match="does not match"):
        tools.restore(artifact, _manifest())
    assert signaled == []
    assert json.loads((artifact / "release.json").read_text())["release"] is False


def test_snapshot_private_path_and_link_remap_security(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(SnapshotSecurityError, match="symlink"):
        validate_artifact_root(linked, creating=False)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    with pytest.raises(SnapshotSecurityError, match="world-writable"):
        validate_artifact_root(unsafe / "snapshot", creating=True)

    artifact = _local_restore_artifact(tmp_path)
    saved_remaps = artifact / "link-remaps"
    saved_remaps.mkdir()
    (saved_remaps / "link_remap.270").write_bytes(b"semaphore state")
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    target = shm_dir / "link_remap.270"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"semaphore state")
    target.symlink_to(replacement)
    tools = LocalSnapshotTools()
    tools.shm_dir = shm_dir
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))
    with pytest.raises(SnapshotRestoreError, match="conflicting CRIU link remap"):
        tools.restore(artifact, _manifest())


@pytest.mark.parametrize(
    ("blocked_state", "message"),
    [("tcp", "external established TCP"), ("io_uring", "kernel.io_uring_disabled=1")],
)
def test_snapshot_rejects_unsafe_process_state_before_criu(
    monkeypatch: pytest.MonkeyPatch, blocked_state: str, message: str
):
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100, 101))
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("101, GPU-abc",))
    monkeypatch.setattr(
        tools,
        "_descriptor_inventory",
        lambda _tree: (
            (101,) if blocked_state == "io_uring" else (),
            {41},
        ),
    )
    tcp_record = _TcpSocketRecord(
        family="AF_INET",
        local_raw="3C00000A:D6F6",
        remote_raw="01010101:01BB" if blocked_state == "tcp" else "00000000:0000",
        inode=41,
    )
    monkeypatch.setattr(tools, "_tcp_records", lambda: (tcp_record,))
    with pytest.raises(SnapshotCreateError, match=message):
        tools.inventory(100)


@pytest.mark.parametrize("argv", [["--help"], ["serve", "--help"]])
def test_help_is_import_light(argv):
    output = _run_import_light_help(argv)
    if argv[0] != "serve":
        return

    for argument in (
        "model_tag",
        "--headless",
        "--api-server-count",
        "-asc",
        "--config",
        "--grpc",
    ):
        assert argument in output
    for non_core_argument in ("--host", "--port", "--max-model-len"):
        assert non_core_argument not in output
    assert "Config Groups:" not in output
    assert "vllm serve --help=all" in output


def test_serve_help_all_uses_canonical_parser():
    output = _run_python("-m", "vllm.entrypoints.cli.main", "serve", "--help=all")
    for argument in ("model_tag", "--grpc", "--host", "--max-model-len"):
        assert argument in output.stdout


@pytest.mark.parametrize(
    ("argv", "module_name"),
    [
        pytest.param(
            ["serve", "--help=all"],
            "vllm.entrypoints.cli.serve",
            id="serve",
        ),
        pytest.param(
            ["bench", "--help"],
            "vllm.entrypoints.cli.benchmark.main",
            id="bench",
        ),
    ],
)
def test_runtime_cli_sets_environment_before_loading_selected_command(
    argv, module_name
):
    script = f"""
import importlib.abc
import os
import sys

os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)

class RuntimeImportOrderGuard(importlib.abc.MetaPathFinder):
    seen = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch":
            self.seen.add("torch")
            assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
        if fullname == {module_name!r}:
            self.seen.add("selected")
            assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
            assert "torch" in self.seen
            assert "vllm.env_override" in sys.modules
            assert "vllm.entrypoints.serve.utils.api_utils" not in sys.modules
        return None

guard = RuntimeImportOrderGuard()
sys.meta_path.insert(0, guard)
sys.argv = ["vllm", *{argv!r}]

from vllm.entrypoints.cli.main import main

try:
    main()
except SystemExit as exc:
    assert exc.code == 0

assert guard.seen == {{"torch", "selected"}}
"""
    _run_python("-c", script)


def test_serve_parser_uses_explicit_args_not_host_sys_argv():
    from vllm.entrypoints.cli.serve import ServeSubcommand

    with patch.object(sys, "argv", ["foreign-host", "serve", "--help"]):
        parser = FlexibleArgumentParser()
        subparsers = parser.add_subparsers(dest="subparser", required=True)
        selected_command = ServeSubcommand()
        selected_command.subparser_init(subparsers)
        args = parser.parse_args(["serve", "--host", "127.0.0.1"])

    assert selected_command.name == "serve"
    assert args.host == "127.0.0.1"


@pytest.mark.parametrize(
    "runtime_import",
    [
        pytest.param("import vllm.compilation.compiler_interface", id="compilation"),
        pytest.param("import vllm.v1.worker.gpu_model_runner", id="v1"),
        pytest.param("import vllm.v1.worker.gpu.model_runner", id="v2"),
        pytest.param("import vllm\nvllm.RequestOutput", id="lazy-export"),
    ],
)
def test_runtime_imports_preserve_environment_overrides(runtime_import):
    script = f"""
import os

{runtime_import}
from torch._inductor.lowering import FALLBACK_ALLOW_LIST

assert "vllm::runtime_override_probe" in FALLBACK_ALLOW_LIST
assert os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
"""
    _run_python("-c", script)
