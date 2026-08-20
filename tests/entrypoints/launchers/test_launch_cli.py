# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm launch` CLI subcommand."""

import argparse
import json
import os
import signal
import socket
import stat
import subprocess
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec, patch

import pytest

import vllm.snapshot.runtime as snapshot_runtime
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
from vllm.snapshot.server import (
    SnapshotCanaryError,
    run_snapshot_child,
)
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


def test_top_level_help_does_not_import_runtime_modules():
    script = """
import sys

sys.argv = ["vllm", "--help"]
from vllm.entrypoints.cli.main import main

exit_code = None
try:
    main()
except SystemExit as exc:
    exit_code = exc.code

assert exit_code == 0
blocked_prefixes = (
    "torch",
    "uvloop",
    "vllm.env_override",
    "vllm.entrypoints.openai.api_server",
    "vllm.v1.executor",
    "vllm.v1.metrics",
)
loaded = sorted(
    prefix
    for prefix in blocked_prefixes
    if any(
        name == prefix or name.startswith(f"{prefix}.") for name in sys.modules
    )
)
assert not loaded, loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


_MODULE_REPORT_PREFIX = "__VLLM_MODULES__="
_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _run_cli(argv: list[str]) -> None:
    from vllm.entrypoints.cli import main as cli_main

    with patch.object(sys, "argv", ["vllm", *argv]):
        cli_main.main()


def _run_cli_module_probe(
    argv: list[str], tracked_modules: set[str]
) -> tuple[str, set[str]]:
    script = f"""
import json
import sys

from vllm.entrypoints.cli.main import main

sys.argv = ["vllm", *{argv!r}]
try:
    main()
except SystemExit as exc:
    if exc.code:
        raise
finally:
    loaded = sorted(name for name in {tracked_modules!r} if name in sys.modules)
    print({_MODULE_REPORT_PREFIX!r} + json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = result.stdout.rsplit(_MODULE_REPORT_PREFIX, 1)[1].splitlines()[0]
    return result.stdout, set(json.loads(report))


def test_snapshot_create_preserves_vllm_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.entrypoints.cli import main as cli_main

    calls: list[str] = []
    monkeypatch.setattr(
        cli_main,
        "_setup_cli_environment",
        lambda: os.environ.__setitem__("VLLM_UNEXPECTED_SETUP", "1"),
    )
    monkeypatch.setattr(
        snapshot_cli, "run_create", lambda _args: calls.append("create")
    )
    monkeypatch.setenv("VLLM_USER_SETTING", "configured")
    before = {
        name: value for name, value in os.environ.items() if name.startswith("VLLM_")
    }

    _run_cli(
        [
            "snapshot",
            "create",
            "Qwen/Qwen3-0.6B",
            "--snapshot-dir=/tmp/vllm-snapshot",
        ]
    )
    after = {
        name: value for name, value in os.environ.items() if name.startswith("VLLM_")
    }
    assert calls == ["create"]
    assert after == before


def parse_snapshot(*argv: str):
    with patch.object(sys, "argv", ["vllm", "snapshot", *argv]):
        parser = FlexibleArgumentParser()
        subparsers = parser.add_subparsers(dest="subparser", required=True)
        snapshot_cli.SnapshotSubcommand().subparser_init(subparsers)
        return parser.parse_args(["snapshot", *argv])


def test_snapshot_help_stays_lazy():
    runtime_modules = {
        "torch",
        "uvloop",
        "vllm.entrypoints.openai.cli_args",
        "vllm.snapshot.controller",
        "vllm.snapshot.server",
        "vllm.v1.worker.gpu_model_runner",
    }
    stdout, loaded = _run_cli_module_probe(["snapshot", "--help"], runtime_modules)
    assert "snapshot" in stdout.lower()
    assert loaded == set()


def test_snapshot_create_cli_accepts_compact_and_full_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(snapshot_cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(snapshot_cli.platform, "machine", lambda: "x86_64")
    base = ("create", "Qwen/Qwen3-0.6B", "--snapshot-dir=/tmp/snapshot")
    compact = parse_snapshot(*base, "--revision", _MODEL_REVISION)
    local_model = tmp_path / "model"
    local_model.mkdir()
    full = parse_snapshot(
        "create",
        str(local_model),
        "--snapshot-dir=/tmp/snapshot",
        "--include-model-state",
    )

    snapshot_cli.validate_create_args(compact)
    snapshot_cli.validate_create_args(full)
    assert (compact.model_tag, compact.include_model_state) == (
        "Qwen/Qwen3-0.6B",
        False,
    )
    assert (full.model_tag, full.include_model_state) == (str(local_model), True)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("pcp2", "prefill context parallel size 1"),
        ("local-compact", "immutable remote model"),
        ("raw-logits", "log-probability mode"),
        ("speculative", r"speculative decoding.*--include-model-state"),
    ],
)
def test_snapshot_create_rejects_invalid_config(
    case: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = "Qwen/Qwen3-0.6B"
    argv = ["create", model, "--snapshot-dir=/tmp/snapshot"]
    if case == "local-compact":
        local_model = tmp_path / "model"
        local_model.mkdir()
        argv[1] = str(local_model)
    if case != "local-compact":
        argv += ["--revision", _MODEL_REVISION]
    extras = {
        "pcp2": ["--prefill-context-parallel-size", "2"],
        "raw-logits": ["--logprobs-mode", "raw_logits"],
        "speculative": [
            "--speculative-config",
            '{"method":"ngram","num_speculative_tokens":3}',
        ],
    }
    argv += extras.get(case, [])
    with pytest.raises(ValueError, match=message):
        snapshot_cli.validate_create_args(parse_snapshot(*argv))


def _oracle(
    token_ids: tuple[int, ...] = (12095,),
    text: str = " Paris",
    logprob: float = -0.125,
) -> Oracle:
    return Oracle(token_ids, text, logprob)


def _manifest(**changes: object) -> SnapshotManifest:
    manifest = SnapshotManifest(
        schema_version=1,
        boundary="post-engine-init-pre-http-bind",
        created_at="2026-08-06T00:00:00Z",
        artifact_bytes=8,
        source_revision="source-sha",
        binary_revision="binary-sha",
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


def _fake_snapshot_tools(*, fail_dump: bool = False) -> Any:
    tools = create_autospec(LocalSnapshotTools, instance=True)
    tools.launch_child.return_value = 100
    tools.wait_ready.return_value = _oracle()
    tools.inventory.return_value = ProcessInventory(100, (100, 101), (101,), "GPU-abc")

    def make_manifest(args, engine_argv, inventory, oracle, _workdir):
        return _manifest(
            boundary=(
                "post-engine-init-pre-http-bind"
                if args.include_model_state
                else "post-engine-init-reloadable-state-released"
            ),
            engine_argv=engine_argv,
            process_tree=inventory.process_tree,
            cuda_holders=inventory.cuda_holders,
            oracle_token_ids=oracle.token_ids,
            oracle_text=oracle.text,
            oracle_sampled_token_logprob=oracle.sampled_token_logprob,
        )

    tools.make_manifest.side_effect = make_manifest
    tools.publish.side_effect = write_manifest_atomic
    tools.current_identity.side_effect = lambda _gpu_uuid: _runtime_identity()
    tools.restore.return_value = 100
    tools.request_oracle.return_value = _oracle()
    if fail_dump:
        tools.dump.side_effect = RuntimeError("dump failed")
    return tools


def test_snapshot_create_abort_kills_surviving_child_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tools = LocalSnapshotTools()
    process = create_autospec(subprocess.Popen, instance=True)
    process.poll.return_value = 1
    process.wait.return_value = 1
    tools._children[100] = process
    (tmp_path / "child.log").write_text("root failed")
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )

    with pytest.raises(SnapshotCreateError, match="root failed"):
        tools.wait_ready(tmp_path, 100)
    tools.abort_create(100)

    assert killed == [(100, signal.SIGKILL)]
    assert 100 not in tools._children
    process.wait.assert_called_once_with(timeout=10)


async def _return_none() -> None:
    return None


@pytest.mark.asyncio
async def test_snapshot_child_rejects_compact_mismatch_before_publication():
    events: list[str] = []
    canaries = iter((_oracle(), _oracle((42,), " Lyon")))

    @asynccontextmanager
    async def engine_context():
        yield object()

    async def next_canary(_engine: object) -> Oracle:
        return next(canaries)

    with pytest.raises(SnapshotCanaryError, match="--include-model-state"):
        await run_snapshot_child(
            engine_context=engine_context(),
            run_canary=next_canary,
            prepare_snapshot=lambda: events.append("prepared"),
            write_ready=lambda _oracle: events.append("published"),
            wait_for_release=_return_none,
            bind_and_serve=lambda _engine: _return_none(),
            release_reloadable_state=lambda _engine: _return_none(),
            restore_reloadable_state=lambda _engine: _return_none(),
        )
    assert events == []


def test_snapshot_create_boundaries_permissions_and_rollback(
    tmp_path: Path,
):
    old_umask = os.umask(0)
    try:
        for include_model_state in (False, True):
            target = (
                tmp_path / "private-parent" / "full"
                if include_model_state
                else tmp_path / "compact"
            )
            tools = _fake_snapshot_tools()
            create_snapshot(
                argparse.Namespace(
                    snapshot_dir=str(target),
                    model_tag="Qwen/Qwen3-0.6B",
                    include_model_state=include_model_state,
                ),
                engine_argv=(
                    "Qwen/Qwen3-0.6B",
                    "--snapshot-dir",
                    str(target),
                    "--dtype",
                    "float16",
                    *(("--include-model-state",) if include_model_state else ()),
                ),
                tools=tools,
            )
            manifest = read_manifest(target)
            expected_argv: tuple[str, ...] = (
                "Qwen/Qwen3-0.6B",
                "--dtype",
                "float16",
            )
            if not include_model_state:
                expected_argv += ("--enable-sleep-mode",)
            assert (manifest.engine_argv, manifest.boundary) == (
                expected_argv,
                (
                    "post-engine-init-pre-http-bind"
                    if include_model_state
                    else "post-engine-init-reloadable-state-released"
                ),
            )
            assert stat.S_IMODE(target.stat().st_mode) == 0o700
            assert stat.S_IMODE((target / "manifest.json").stat().st_mode) == 0o600
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE((tmp_path / "private-parent").stat().st_mode) == 0o700

    def args(target: Path) -> argparse.Namespace:
        return argparse.Namespace(
            snapshot_dir=str(target),
            model_tag="Qwen/Qwen3-0.6B",
            include_model_state=False,
        )

    for phase in ("dump", "manifest"):
        target = tmp_path / f"failed-{phase}"
        tools = _fake_snapshot_tools(fail_dump=phase == "dump")
        if phase == "manifest":
            tools.make_manifest.side_effect = RuntimeError("manifest failed")
        with pytest.raises(RuntimeError, match=f"{phase} failed"):
            create_snapshot(args(target), tools=tools)
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
        _runtime_identity(binary_revision="different")
    )
    with pytest.raises(SnapshotCompatibilityError, match="binary_revision"):
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
        pytest.param({"unexpected": True}, "unexpected", id="extra-field"),
        pytest.param({"created_at": 1}, "created_at", id="strict-string"),
        pytest.param({"schema_version": True}, "schema_version", id="schema-bool"),
        pytest.param({"schema_version": 2}, "schema_version", id="schema-value"),
        pytest.param({"boundary": "unknown"}, "boundary", id="boundary"),
        pytest.param({"process_tree": []}, "process_tree", id="empty-process-tree"),
        pytest.param({"process_tree": [0]}, "process_tree.0", id="nonpositive-pid"),
        pytest.param({"process_tree": [100, 100]}, "root", id="duplicate-process-pid"),
        pytest.param({"cuda_holders": []}, "cuda_holders", id="empty-cuda-holders"),
        pytest.param(
            {"cuda_holders": [-1]}, "cuda_holders.0", id="nonpositive-cuda-holder"
        ),
        pytest.param({"cuda_holders": [999]}, "root", id="cuda-holder-outside-tree"),
        pytest.param({"cuda_holders": [101, 101]}, "root", id="duplicate-cuda-holder"),
        pytest.param(
            {"oracle_token_ids": []}, "oracle_token_ids", id="empty-oracle-token"
        ),
        pytest.param(
            {"oracle_token_ids": [1, 2]},
            "oracle_token_ids",
            id="multiple-oracle-tokens",
        ),
        pytest.param(
            {"oracle_token_ids": [-1]},
            "oracle_token_ids.0",
            id="negative-oracle-token",
        ),
        pytest.param(
            {"oracle_sampled_token_logprob": 0},
            "oracle_sampled_token_logprob",
            id="integer-logprob",
        ),
        pytest.param(
            {"oracle_sampled_token_logprob": float("nan")},
            "oracle_sampled_token_logprob",
            id="nan-logprob",
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
    args = argparse.Namespace(snapshot_dir=str(artifact), host="127.0.0.1", port=9000)

    invalid = _manifest().model_dump(mode="json")
    invalid.update(invalid_update)
    (artifact / "manifest.json").write_text(json.dumps(invalid))
    with pytest.raises(
        SnapshotCompatibilityError,
        match=f"invalid snapshot manifest: {diagnostic}",
    ):
        restore_snapshot(args, tools=_fake_snapshot_tools())


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
        pytest.param(None, "already occupied: 100", id="occupied"),
        pytest.param(
            PermissionError(), "already occupied: 100", id="permission-denied"
        ),
        pytest.param(
            OSError("probe failed"),
            "availability probe failed: 100",
            id="unexpected-error",
        ),
    ],
)
def test_snapshot_restore_rejects_unavailable_pid_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_error: OSError | None,
    diagnostic: str,
):
    artifact = _local_restore_artifact(tmp_path)
    release = artifact / "release.json"
    pidfile = artifact / "restored.pid"
    release.write_bytes(b"release sentinel")
    pidfile.write_bytes(b"pid sentinel")
    remaps = artifact / "link-remaps"
    remaps.mkdir()
    (remaps / "link_remap.270").write_bytes(b"semaphore state")
    write_manifest_atomic(artifact, _manifest(process_tree=(100,), cuda_holders=(100,)))
    before = {
        path.relative_to(artifact): path.read_bytes()
        for path in artifact.rglob("*")
        if path.is_file()
    }
    tools = LocalSnapshotTools()
    tools.shm_dir = tmp_path / "shm"
    tools.shm_dir.mkdir()
    criu_calls: list[str] = []

    def fail_probe(_pid: int, _signal: int) -> None:
        if probe_error is not None:
            raise probe_error

    monkeypatch.setattr(snapshot_runtime.os, "kill", fail_probe)
    monkeypatch.setattr(tools, "preflight", lambda *_args: None)
    monkeypatch.setattr(tools, "current_identity", lambda _uuid: _runtime_identity())
    monkeypatch.setattr(
        tools, "_criu", lambda action, *_args: criu_calls.append(action)
    )

    with pytest.raises(SnapshotRestoreError, match=diagnostic):
        restore_snapshot(
            argparse.Namespace(snapshot_dir=str(artifact), host="127.0.0.1", port=9000),
            tools=tools,
        )
    assert criu_calls == []
    assert {
        path.relative_to(artifact): path.read_bytes()
        for path in artifact.rglob("*")
        if path.is_file()
    } == before
    assert not (tools.shm_dir / "link_remap.270").exists()


def _fake_restored_tree(
    tools: LocalSnapshotTools,
    monkeypatch: pytest.MonkeyPatch,
    artifact: Path,
    states: dict[int, tuple[int, int, int, int]],
) -> None:
    command: tuple[str, ...] = (sys.executable, "-m", "vllm.snapshot.server")
    command += ("--release-file", str(artifact / "release.json"))
    monkeypatch.setattr(tools, "_process_state", states.__getitem__)
    monkeypatch.setattr(tools, "_process_states", lambda: dict(states))
    monkeypatch.setattr(tools, "_process_command", lambda _pid: command)


def _fake_pidfds(
    monkeypatch: pytest.MonkeyPatch, *pids: int
) -> tuple[list[tuple[int, signal.Signals]], dict[int, tuple[int, int]]]:
    pipes = {pid: os.pipe() for pid in pids}
    pid_by_fd = {read_fd: pid for pid, (read_fd, _write_fd) in pipes.items()}
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        snapshot_runtime.os,
        "pidfd_open",
        lambda pid: pipes[pid][0],
        raising=False,
    )

    def send(pidfd: int, signum: signal.Signals, _siginfo: None, _flags: int) -> None:
        pid = pid_by_fd[pidfd]
        signals.append((pid, signum))
        if signum == signal.SIGKILL:
            os.write(pipes[pid][1], b"x")

    monkeypatch.setattr(
        snapshot_runtime.signal, "pidfd_send_signal", send, raising=False
    )
    return signals, pipes


def _restored_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    states: dict[int, tuple[int, int, int, int]],
) -> tuple[
    Path,
    LocalSnapshotTools,
    list[tuple[int, signal.Signals]],
    dict[int, tuple[int, int]],
]:
    _mark_snapshot_pids_free(monkeypatch)
    artifact = _local_restore_artifact(tmp_path)
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_read_restored_pid", lambda _path: 100)
    monkeypatch.setattr(tools, "_criu", lambda *_args: None)
    _fake_restored_tree(tools, monkeypatch, artifact, states)
    signals, pipes = _fake_pidfds(monkeypatch, *states)
    return artifact, tools, signals, pipes


def _close_pipes(pipes: dict[int, tuple[int, int]]) -> None:
    for pipe in pipes.values():
        for descriptor in pipe:
            with suppress(OSError):
                os.close(descriptor)


@pytest.mark.parametrize(
    ("payload", "criu_error", "message"),
    [
        ("222\n", True, "criu failed"),
        ("333\n", False, "does not match"),
    ],
)
def test_snapshot_restore_never_signals_an_unpinned_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str | None,
    criu_error: bool,
    message: str,
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
        if payload is not None:
            (root / "restored.pid").write_text(payload)
        if criu_error:
            raise RuntimeError("criu failed")

    monkeypatch.setattr(tools, "_criu", restore)
    error = RuntimeError if criu_error else SnapshotRestoreError
    with pytest.raises(error, match=message):
        tools.restore(artifact, _manifest())
    assert signaled == []
    assert json.loads((artifact / "release.json").read_text())["release"] is False


def test_snapshot_restore_pins_and_cleans_the_exact_process_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    states = {100: (1, 100, 100, 10), 101: (100, 101, 100, 11)}
    artifact, tools, signals, pipes = _restored_tools(tmp_path, monkeypatch, states)
    monkeypatch.setattr(snapshot_runtime.time, "sleep", lambda _seconds: None)
    try:
        assert tools.restore(artifact, _manifest()) == 100
        with socket.socket() as available:
            available.bind(("127.0.0.1", 0))
            port = available.getsockname()[1]
        tools.release(artifact, "127.0.0.1", port)
        tools.cleanup(100)
        assert signals == [
            (100, signal.SIGTERM),
            (101, signal.SIGTERM),
            (100, signal.SIGKILL),
            (101, signal.SIGKILL),
        ]
    finally:
        _close_pipes(pipes)


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
    criu_calls: list[str] = []
    monkeypatch.setattr(
        tools, "_criu", lambda action, *_args: criu_calls.append(action)
    )
    with pytest.raises(SnapshotRestoreError, match="conflicting CRIU link remap"):
        tools.restore(artifact, _manifest())
    assert criu_calls == []


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
    monkeypatch.setattr(
        tools,
        "_tcp_records",
        lambda: (
            _TcpSocketRecord(
                family="AF_INET",
                local_raw="3C00000A:D6F6",
                remote_raw=(
                    "01010101:01BB" if blocked_state == "tcp" else "00000000:0000"
                ),
                inode=41,
            ),
        ),
    )
    with pytest.raises(SnapshotCreateError, match=message):
        tools.inventory(100)


def test_serve_help_is_compact_and_import_light():
    script = """
import os
import sys

os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
sys.argv = ["vllm", "serve", "--help"]
from vllm.entrypoints.cli.main import main

exit_code = None
try:
    main()
except SystemExit as exc:
    exit_code = exc.code

assert exit_code == 0
assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ
blocked_prefixes = (
    "torch",
    "uvloop",
    "vllm.env_override",
    "vllm.entrypoints.cli.serve",
    "vllm.entrypoints.launchers.api_server",
    "vllm.entrypoints.openai.api_server",
    "vllm.entrypoints.openai.cli_args",
    "vllm.entrypoints.serve.utils.api_utils",
    "vllm.v1.executor",
    "vllm.v1.metrics",
)
loaded = sorted(
    prefix
    for prefix in blocked_prefixes
    if any(
        name == prefix or name.startswith(f"{prefix}.") for name in sys.modules
    )
)
assert not loaded, loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for argument in (
        "model_tag",
        "--headless",
        "--api-server-count",
        "-asc",
        "--config",
        "--grpc",
    ):
        assert argument in result.stdout
    for non_core_argument in ("--host", "--port", "--max-model-len"):
        assert non_core_argument not in result.stdout
    assert "Config Groups:" not in result.stdout
    assert "vllm serve --help=all" in result.stdout


def test_serve_help_all_uses_canonical_parser():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "--help=all",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for argument in ("model_tag", "--grpc", "--host", "--max-model-len"):
        assert argument in result.stdout


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
    torch_seen = False
    selected_seen = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch":
            self.torch_seen = True
            assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
        if fullname == {module_name!r}:
            self.selected_seen = True
            assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
            assert self.torch_seen
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

assert guard.torch_seen
assert guard.selected_seen
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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


def test_public_lazy_exports_apply_runtime_environment_once():
    script = """
import sys

import vllm

assert "torch" not in sys.modules
vllm.RequestOutput
override_module = sys.modules["vllm.env_override"]
vllm.SamplingParams

from torch._inductor.lowering import FALLBACK_ALLOW_LIST

assert sys.modules["vllm.env_override"] is override_module
assert "vllm::runtime_override_probe" in FALLBACK_ALLOW_LIST
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "runtime_import",
    [
        pytest.param("import vllm.compilation.compiler_interface", id="compilation"),
        pytest.param("import vllm.config", id="config"),
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
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
