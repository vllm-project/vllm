# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm launch` CLI subcommand."""

import argparse
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import urllib.request
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec, patch

import pytest

import vllm.snapshot.controller as snapshot_controller
from vllm.entrypoints.cli import snapshot as snapshot_cli
from vllm.entrypoints.cli.launch import (
    LaunchSubcommand,
    RenderSubcommand,
    cmd_init,
)
from vllm.snapshot.controller import (
    LocalSnapshotTools,
    ProcessInventory,
    SnapshotCreateError,
    SnapshotRestoreError,
    _TcpSocketRecord,
    create_snapshot,
    restore_snapshot,
)
from vllm.snapshot.manifest import (
    SnapshotCompatibilityError,
    SnapshotManifest,
    SnapshotSecurityError,
    read_manifest,
    validate_artifact_root,
    write_manifest_atomic,
)
from vllm.snapshot.server import (
    SnapshotCanaryError,
    run_engine_canary,
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
_CLI_COMMAND_MODULES = {
    "vllm.entrypoints.cli.benchmark.main",
    "vllm.entrypoints.cli.collect_env",
    "vllm.entrypoints.cli.launch",
    "vllm.entrypoints.cli.openai",
    "vllm.entrypoints.cli.run_batch",
    "vllm.entrypoints.cli.serve",
    "vllm.entrypoints.cli.snapshot",
}


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


def test_snapshot_actions_preserve_the_complete_vllm_environment(
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
    monkeypatch.setattr(
        snapshot_cli, "run_restore_cli", lambda _argv: calls.append("restore")
    )
    monkeypatch.setenv("VLLM_USER_SETTING", "configured")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
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
    _run_cli(["snapshot", "restore", "/tmp/vllm-snapshot"])

    after = {
        name: value for name, value in os.environ.items() if name.startswith("VLLM_")
    }
    assert calls == ["create", "restore"]
    assert after == before


def parse_snapshot(*argv: str):
    with patch.object(sys, "argv", ["vllm", "snapshot", *argv]):
        parser = FlexibleArgumentParser()
        subparsers = parser.add_subparsers(dest="subparser", required=True)
        snapshot_cli.SnapshotSubcommand().subparser_init(subparsers)
        return parser.parse_args(["snapshot", *argv])


def test_snapshot_help_and_restore_dispatch_stay_lazy(
    monkeypatch: pytest.MonkeyPatch,
):
    stdout, loaded = _run_cli_module_probe(["--help"], _CLI_COMMAND_MODULES | {"torch"})
    assert (
        "{chat,complete,serve,launch,bench,collect-env,run-batch,snapshot}"
        in "".join(stdout.split())
    )
    assert loaded == set()
    runtime_modules = {
        "torch",
        "uvloop",
        "vllm.entrypoints.openai.cli_args",
        "vllm.snapshot.controller",
        "vllm.snapshot.server",
        "vllm.v1.worker.gpu_model_runner",
    }
    for argv in (["snapshot", "--help"], ["snapshot", "restore", "--help"]):
        stdout, loaded = _run_cli_module_probe(argv, runtime_modules)
        assert "snapshot" in stdout.lower()
        assert loaded == set()

    calls: list[list[str]] = []
    monkeypatch.setattr(snapshot_cli, "run_restore_cli", calls.append)
    _run_cli(["snapshot", "restore", "/tmp/qwen-snapshot"])
    assert calls == [["/tmp/qwen-snapshot"]]


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
        socket_inventory=(),
        oracle_token_ids=(12095,),
        oracle_text=" Paris",
        oracle_sampled_token_logprob=-0.125,
    )
    return SnapshotManifest.model_validate(
        {**manifest.model_dump(mode="python"), **changes}, strict=True
    )


def _fake_snapshot_tools(*, fail_dump: bool = False) -> Any:
    tools = create_autospec(LocalSnapshotTools, instance=True)
    tools.launch_child.return_value = 100
    tools.wait_ready.return_value = _oracle()
    tools.inventory.return_value = ProcessInventory(
        100, (100, 101), (101,), "GPU-abc", ()
    )

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
    tools.current_identity.side_effect = lambda manifest: manifest
    tools.restore.return_value = 100
    tools.request_oracle.return_value = _oracle()
    if fail_dump:
        tools.dump.side_effect = RuntimeError("dump failed")
    return tools


async def _record(events: list[str], event: str, value=None):
    events.append(event)
    return value


async def _return_none() -> None:
    return None


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.asyncio
async def test_snapshot_child_lifecycle_and_failure_nonpublication(compact: bool):
    events: list[str] = []

    @asynccontextmanager
    async def engine_context():
        events.append("engine-ready")
        yield object()

    await run_snapshot_child(
        engine_context=engine_context(),
        run_canary=lambda _engine: _record(events, "canary", _oracle()),
        prepare_snapshot=lambda: events.append("streams-detached"),
        write_ready=lambda _oracle: events.append("ready-written"),
        wait_for_release=lambda: _record(events, "released"),
        bind_and_serve=lambda _engine: _record(events, "http-bound"),
        release_reloadable_state=(
            (lambda _engine: _record(events, "release-state")) if compact else None
        ),
        restore_reloadable_state=(
            (lambda _engine: _record(events, "restore-state")) if compact else None
        ),
    )
    expected = ["engine-ready", "canary"]
    if compact:
        expected += ["release-state", "restore-state", "canary", "release-state"]
    expected += ["streams-detached", "ready-written", "released"]
    if compact:
        expected.append("restore-state")
    assert events == [*expected, "http-bound"]

    if not compact:
        return

    failure_events: list[str] = []
    canaries = iter((_oracle(), _oracle((42,), " Lyon")))

    @asynccontextmanager
    async def failing_context():
        yield object()

    async def next_canary(_engine: object) -> Oracle:
        return next(canaries)

    with pytest.raises(SnapshotCanaryError, match="--include-model-state"):
        await run_snapshot_child(
            engine_context=failing_context(),
            run_canary=next_canary,
            prepare_snapshot=lambda: failure_events.append("streams-detached"),
            write_ready=lambda _oracle: failure_events.append("ready-written"),
            wait_for_release=_return_none,
            bind_and_serve=lambda _engine: _return_none(),
            release_reloadable_state=lambda _engine: _return_none(),
            restore_reloadable_state=lambda _engine: _return_none(),
        )
    assert failure_events == []


@pytest.mark.asyncio
async def test_snapshot_canary_records_logprob_alias_and_fixed_tolerance(
    monkeypatch: pytest.MonkeyPatch,
):
    candidate = SimpleNamespace(
        token_ids=[12095],
        text=" Paris",
        logprobs=[{12095: SimpleNamespace(logprob=-0.125)}],
    )

    class FakeEngine:
        sampling_params = None

        async def generate(self, _prompt, sampling_params, **_kwargs):
            self.sampling_params = sampling_params
            yield SimpleNamespace(outputs=[candidate])

    engine = FakeEngine()
    assert await run_engine_canary(engine, "The capital of France is") == _oracle()
    assert engine.sampling_params is not None
    assert (
        engine.sampling_params.temperature,
        engine.sampling_params.min_tokens,
        engine.sampling_params.max_tokens,
        engine.sampling_params.seed,
        engine.sampling_params.logprobs,
    ) == (0, 1, 1, 0, 0)

    captured: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        assert timeout == 120
        captured.append(request)
        return io.BytesIO(
            b'{"choices":[{"token_ids":[12095],"text":" Paris",'
            b'"logprobs":{"token_logprobs":[-0.125]}}]}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    manifest = _manifest(model="source-model", served_model_name="snapshot-alias")
    assert LocalSnapshotTools().request_oracle("::", 8000, manifest) == _oracle()
    assert isinstance(captured[0].data, bytes)
    payload = json.loads(captured[0].data)
    assert (payload["model"], payload["logprobs"], payload["min_tokens"]) == (
        "snapshot-alias",
        0,
        1,
    )
    assert oracles_match(_oracle(logprob=-100.0), _oracle(logprob=-100.0005))
    assert not oracles_match(_oracle(logprob=-100.0), _oracle(logprob=-100.002))


def test_snapshot_create_outcomes_and_child_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            assert manifest.engine_argv == expected_argv
            assert manifest.boundary == (
                "post-engine-init-pre-http-bind"
                if include_model_state
                else "post-engine-init-reloadable-state-released"
            )
            tools.launch_child.assert_called_once_with(
                target, expected_argv, include_model_state=include_model_state
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

    failed = tmp_path / "failed"
    failing_tools = _fake_snapshot_tools(fail_dump=True)
    with pytest.raises(RuntimeError, match="dump failed"):
        create_snapshot(args(failed), tools=failing_tools)
    failing_tools.abort_create.assert_called_once()
    assert not failed.exists()


@pytest.mark.parametrize(
    ("invalid_update", "diagnostic"),
    [
        pytest.param({"unexpected": True}, "fields", id="extra-field"),
        pytest.param({"created_at": 1}, "created_at", id="strict-string"),
        pytest.param({"schema_version": True}, "schema_version", id="schema-bool"),
        pytest.param({"schema_version": 2}, "schema_version", id="schema-value"),
        pytest.param({"complete": 1}, "complete", id="complete-int"),
        pytest.param({"complete": False}, "complete", id="complete-false"),
        pytest.param({"boundary": "unknown"}, "boundary", id="boundary"),
        pytest.param({"process_tree": []}, "process_tree", id="empty-process-tree"),
        pytest.param({"process_tree": [0]}, "process_tree", id="nonpositive-pid"),
        pytest.param(
            {"process_tree": [100, 100]},
            "process_tree",
            id="duplicate-process-pid",
        ),
        pytest.param({"cuda_holders": []}, "cuda_holders", id="empty-cuda-holders"),
        pytest.param(
            {"cuda_holders": [-1]}, "cuda_holders", id="nonpositive-cuda-holder"
        ),
        pytest.param(
            {"cuda_holders": [999]},
            "cuda_holders",
            id="cuda-holder-outside-tree",
        ),
        pytest.param(
            {"cuda_holders": [101, 101]},
            "cuda_holders",
            id="duplicate-cuda-holder",
        ),
        pytest.param({"oracle_token_ids": []}, "oracle", id="empty-oracle-token"),
        pytest.param(
            {"oracle_token_ids": [1, 2]}, "oracle", id="multiple-oracle-tokens"
        ),
        pytest.param({"oracle_token_ids": [-1]}, "oracle", id="negative-oracle-token"),
        pytest.param(
            {"oracle_sampled_token_logprob": 0}, "oracle", id="integer-logprob"
        ),
        pytest.param(
            {"oracle_sampled_token_logprob": True}, "oracle", id="boolean-logprob"
        ),
        pytest.param(
            {"oracle_sampled_token_logprob": float("nan")},
            "oracle",
            id="nan-logprob",
        ),
        pytest.param(
            {"oracle_sampled_token_logprob": float("inf")},
            "oracle",
            id="infinite-logprob",
        ),
        pytest.param(
            {
                "socket_inventory": [
                    {
                        "family": "AF_INET",
                        "socket_type": "SOCK_STREAM",
                        "local_address": "127.0.0.1:8000",
                        "remote_address": None,
                        "state": "LISTEN",
                        "unexpected": True,
                    }
                ]
            },
            "socket_inventory",
            id="socket-extra-field",
        ),
    ],
)
def test_snapshot_manifest_validation_and_restore_lifecycle(
    tmp_path: Path,
    invalid_update: dict[str, object],
    diagnostic: str,
):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(artifact, _manifest())
    args = argparse.Namespace(snapshot_dir=str(artifact), host="127.0.0.1", port=9000)

    identity_mismatch = _fake_snapshot_tools()
    identity_mismatch.current_identity.side_effect = lambda _current: _manifest(
        model_revision="different"
    )
    with pytest.raises(SnapshotCompatibilityError, match="model_revision"):
        restore_snapshot(args, tools=identity_mismatch)
    identity_mismatch.restore.assert_not_called()

    successful = _fake_snapshot_tools()
    restore_snapshot(args, tools=successful)
    successful.complete_restore.assert_called_once_with(100)
    successful.cleanup.assert_not_called()

    oracle_mismatch = _fake_snapshot_tools()
    oracle_mismatch.request_oracle.return_value = _oracle(logprob=-0.5)
    with pytest.raises(SnapshotRestoreError, match="oracle mismatch"):
        restore_snapshot(args, tools=oracle_mismatch)
    oracle_mismatch.cleanup.assert_called_once()
    oracle_mismatch.complete_restore.assert_not_called()

    invalid = _manifest().model_dump(mode="json")
    invalid.update(invalid_update)
    (artifact / "manifest.json").write_text(json.dumps(invalid))
    with pytest.raises(SnapshotCompatibilityError, match=diagnostic):
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

    monkeypatch.setattr(snapshot_controller.os, "kill", missing)
    return probed_pids


def test_snapshot_restore_rejects_occupied_pid_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    blocker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        artifact = _local_restore_artifact(tmp_path)
        (artifact / "release.json").write_bytes(b"release sentinel")
        (artifact / "restored.pid").write_bytes(b"pid sentinel")
        with (artifact / "child.log").open("ab") as child_log:
            child_log.write(b" runtime tail")
        saved_remaps = artifact / "link-remaps"
        saved_remaps.mkdir()
        (saved_remaps / "link_remap.270").write_bytes(b"semaphore state")
        write_manifest_atomic(
            artifact,
            _manifest(
                process_tree=(blocker.pid,),
                cuda_holders=(blocker.pid,),
            ),
        )

        tools = LocalSnapshotTools()
        tools.shm_dir = tmp_path / "shm"
        tools.shm_dir.mkdir()
        monkeypatch.setattr(tools, "preflight", lambda *_args: None)
        monkeypatch.setattr(tools, "current_identity", lambda manifest: manifest)
        criu_calls: list[str] = []

        def unexpected_criu(action: str, *_args: object) -> None:
            criu_calls.append(action)
            raise RuntimeError("CRIU called before PID collision rejection")

        monkeypatch.setattr(tools, "_criu", unexpected_criu)

        def file_state() -> dict[str, bytes]:
            return {
                str(path.relative_to(tmp_path)): path.read_bytes()
                for path in tmp_path.rglob("*")
                if path.is_file()
            }

        before = file_state()
        args = argparse.Namespace(
            snapshot_dir=str(artifact), host="127.0.0.1", port=9000
        )
        with pytest.raises(SnapshotRestoreError) as exc_info:
            restore_snapshot(args, tools=tools)

        assert str(blocker.pid) in str(exc_info.value)
        assert criu_calls == []
        assert blocker.poll() is None
        assert file_state() == before
        assert not (tools.shm_dir / "link_remap.270").exists()
    finally:
        if blocker.poll() is None:
            blocker.terminate()
            try:
                blocker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                blocker.kill()
                blocker.wait(timeout=5)


@pytest.mark.parametrize(
    ("probe_error", "diagnostic"),
    [
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
def test_snapshot_restore_pid_probe_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_error: OSError,
    diagnostic: str,
):
    artifact = _local_restore_artifact(tmp_path)
    tools = LocalSnapshotTools()
    criu_calls: list[str] = []

    def fail_probe(_pid: int, _signal: int) -> None:
        raise probe_error

    monkeypatch.setattr(snapshot_controller.os, "kill", fail_probe)
    monkeypatch.setattr(
        tools, "_criu", lambda action, *_args: criu_calls.append(action)
    )

    with pytest.raises(SnapshotRestoreError, match=diagnostic):
        tools.restore(artifact, _manifest())
    assert criu_calls == []


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
        snapshot_controller.os,
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
        snapshot_controller.signal, "pidfd_send_signal", send, raising=False
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


def test_snapshot_criu_uses_file_locks_and_cuda_holder_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probed_pids = _mark_snapshot_pids_free(monkeypatch)
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    (artifact / "child.log").write_bytes(b"startup log")
    tools = LocalSnapshotTools()
    tools.shm_dir = tmp_path / "shm"
    tools.shm_dir.mkdir()
    calls: list[tuple[str, list[str]]] = []

    def criu(action: str, _artifact: Path, arguments: list[str]) -> None:
        calls.append((action, arguments))
        if action == "restore":
            raise RuntimeError("stop after command capture")

    monkeypatch.setattr(tools, "_criu", criu)
    tools.dump(
        artifact,
        ProcessInventory(100, (100, 101), (101,), "GPU-selected", ()),
    )
    with pytest.raises(RuntimeError, match="stop after command capture"):
        tools.restore(artifact, _manifest())

    assert probed_pids == [100, 101]
    assert [action for action, _arguments in calls] == ["dump", "restore"]
    assert all("--file-locks" in arguments for _action, arguments in calls)
    monkeypatch.setattr(
        tools,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            "7, GPU-other\n101, GPU-selected\n",
            "",
        ),
    )
    assert tools._gpu_uuid_for_pids((101,)) == "GPU-selected"


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
    monkeypatch.setattr(snapshot_controller.time, "sleep", lambda _seconds: None)
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
    monkeypatch.setattr(tools, "_cuda_pids", lambda: (101,))
    monkeypatch.setattr(
        tools,
        "_io_uring_pids",
        lambda _tree: (101,) if blocked_state == "io_uring" else (),
    )
    monkeypatch.setattr(tools, "_socket_inodes", lambda _tree: {41})
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
