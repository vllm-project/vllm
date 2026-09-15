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
    _child_engine_argv,
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
    validate_identity,
    write_manifest_atomic,
)
from vllm.snapshot.runtime import ProcessInventory, _decode_endpoint, _TcpSocketRecord
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


def _run_cli(argv: list[str]) -> None:
    from vllm.entrypoints.cli import main as cli_main

    with patch.object(sys, "argv", ["vllm", *argv]):
        cli_main.main()


def test_snapshot_environment_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    from vllm.entrypoints.serve.utils import api_utils

    monkeypatch.setattr(api_utils, "cli_env_setup", pytest.fail)
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

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD")
    implicit_worker_default = dict(LocalSnapshotTools()._environment_identity())
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    assert dict(LocalSnapshotTools()._environment_identity()) == implicit_worker_default

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


@pytest.mark.parametrize(
    "case",
    [
        (("--hf_token", "SECRET"), ("--hf_token", "***")),
        (("--hf_token=SECRET",), ("--hf_token=***",)),
        (("--api_key", "SECRET"), ("--api_key", "***")),
        (("--api_key=SECRET",), ("--api_key=***",)),
        (("--hf-tok", "SECRET"), ("--hf-tok", "***")),
        (("--hf-tok=SECRET",), ("--hf-tok=***",)),
        (("--hf_tok", "SECRET"), ("--hf_tok", "***")),
        (("--api-k", "SECRET", "SECOND"), ("--api-k", "***", "***")),
        (("--api-k=SECRET",), ("--api-k=***",)),
        (("--hf-overrides", "{}"), ("--hf-overrides", "{}")),
    ],
)
def test_snapshot_manifest_redacts_accepted_secret_options(case):
    engine_argv, expected = case
    tools = create_autospec(LocalSnapshotTools, instance=True)
    tools.current_identity.return_value = _runtime_identity()
    tools._artifact_bytes.return_value = 0
    manifest = LocalSnapshotTools.make_manifest(
        tools,
        snapshot_server.parse_vllm_args(["model", *engine_argv]),
        engine_argv,
        ProcessInventory(100, (100, 101), (101,), "GPU-abc"),
        _oracle(),
        Path(),
    )

    assert manifest.engine_argv == expected
    if "SECRET" in " ".join(engine_argv):
        assert "***" in " ".join(manifest.engine_argv)
        assert "SECRET" not in " ".join(manifest.engine_argv)


def parse_snapshot(*argv: str):
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser", required=True)
    snapshot_cli.SnapshotSubcommand(
        create_requested=argv[:1] == ("create",)
    ).subparser_init(subparsers)
    return parser.parse_args(["snapshot", *argv])


@pytest.mark.parametrize("option", ["--snapshot-dir", "--snapshot_dir", "--snapshot-d"])
@pytest.mark.parametrize("joined", [False, True])
def test_snapshot_create_forwards_only_engine_options(option, joined, monkeypatch):
    directory = [f"{option}=/tmp/snapshot"] if joined else [option, "/tmp/snapshot"]
    argv = ["create", "Qwen/Qwen3-0.6B", "--revision", _MODEL_REVISION, *directory]
    args = parse_snapshot(*argv)
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", *argv])

    child = snapshot_server.parse_vllm_args(list(_child_engine_argv(args, None)))

    assert child.model == "Qwen/Qwen3-0.6B"
    assert child.revision == _MODEL_REVISION
    assert child.enable_sleep_mode


@pytest.mark.parametrize("model_in_cli", [False, True])
def test_snapshot_create_config_preserves_cli_precedence(
    tmp_path, monkeypatch, model_in_cli
):
    config = tmp_path / "engine.yaml"
    config.write_text("model: config-model\nmax-model-len: 512\ndtype: float16\n")
    model = ["Qwen/Qwen3-0.6B"] if model_in_cli else []
    argv = [
        "create",
        *model,
        "--snapshot-dir",
        str(tmp_path / "snapshot"),
        "--config",
        str(config),
        "--revision",
        _MODEL_REVISION,
        "--max-model-len",
        "1024",
    ]
    args = parse_snapshot(*argv)
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", *argv])
    child = snapshot_server.parse_vllm_args(list(_child_engine_argv(args, None)))

    assert (
        (args.model_tag or args.model)
        == child.model
        == ("Qwen/Qwen3-0.6B" if model_in_cli else "config-model")
    )
    assert args.max_model_len == child.max_model_len == 1024
    assert args.dtype == child.dtype == "float16"


@pytest.mark.parametrize(
    ("probe", "status", "output"),
    [
        (None, 0, ""),
        ("nvidia-smi", 1, ""),
        ("find", 1, ""),
        ("ss", 1, ""),
        ("nvidia-smi", 0, "12345"),
        ("find", 0, "link_remap.fixture"),
        ("ss", 0, "LISTEN"),
    ],
)
def test_snapshot_e2e_cleanup_requires_successful_empty_probes(probe, status, output):
    script = (
        Path(__file__).resolve().parents[3]
        / ".buildkite/scripts/initialized-snapshot-e2e.sh"
    )
    if not script.exists():
        pytest.skip("snapshot E2E script is not packaged in this test image")
    source = script.read_text()
    # Exercise the real probes without entering Docker/CRIU setup.
    functions = source[source.index("link_remaps() {") : source.index("wait_clean() {")]
    shell = (
        """set -Eeuo pipefail
GPU_UUID=fixture PORT_ONE=18001 PORT_TWO=18002 LINK_REMAP_BASELINE=''
SNAPSHOT_PIDS=(999999999)
nvidia-smi() { :; }
find() { :; }
ss() { :; }
"""
        + functions
    )
    if probe:
        shell += f'\n{probe}() {{ printf %s "{output}"; return {status}; }}\n'
    shell += "\nif state_clean; then exit 0; else exit 1; fi\n"

    result = subprocess.run(["bash", "-c", shell], capture_output=True, timeout=10)

    assert result.returncode == (0 if probe is None else 1), result.stderr


def test_snapshot_create_cli_accepts_only_pinned_compact_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(snapshot_cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(snapshot_cli.platform, "machine", lambda: "x86_64")
    base = ("create", "Qwen/Qwen3-0.6B", "--snapshot-dir=/tmp/snapshot")
    compact = parse_snapshot(*base, "--revision", _MODEL_REVISION)

    snapshot_cli.validate_create_args(compact)
    assert compact.model_tag == "Qwen/Qwen3-0.6B"

    dispatched: list[argparse.Namespace] = []
    monkeypatch.setattr(snapshot_cli, "run_create", dispatched.append)
    _run_cli(["snapshot", *base, "--revision", _MODEL_REVISION])
    assert dispatched[0].revision == _MODEL_REVISION


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


def _supply_waitid(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wait_ready` binds Linux-only `os.waitid` before it reads the marker."""
    monkeypatch.setattr(os, "waitid", lambda *_args: None, raising=False)


def test_snapshot_wait_ready_reads_back_the_oracle_the_child_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _supply_waitid(monkeypatch)
    snapshot_server.write_ready_atomic(tmp_path / "ready.json", _oracle())

    assert LocalSnapshotTools().wait_ready(tmp_path, os.getpid()) == _oracle()


@pytest.mark.parametrize(
    "payload",
    [
        {"token_ids": [12095], "sampled_token_logprob": -0.125},
        {"token_ids": ["12095"], "text": " Paris", "sampled_token_logprob": -0.125},
    ],
)
def test_snapshot_wait_ready_rejects_a_malformed_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
):
    _supply_waitid(monkeypatch)
    (tmp_path / "ready.json").write_text(json.dumps(payload))

    with pytest.raises(SnapshotCreateError, match="ready marker invalid"):
        LocalSnapshotTools().wait_ready(tmp_path, os.getpid())


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


def test_snapshot_release_marker_ignores_unknown_keys(tmp_path: Path):
    marker = tmp_path / "release.json"
    marker.write_text(json.dumps({"release": True, "port": 8000, "unknown": 1}))

    listener = snapshot_server.read_release_marker(marker)

    assert listener == snapshot_server.ListenerConfig(host=None, port=8000)


@pytest.mark.parametrize(
    "payload",
    [
        ["release"],
        {"port": 8000},
        {"release": "true", "port": 8000},
        {"release": 1, "port": 8000},
        {"release": True, "port": 0},
        {"release": True, "port": 70000},
        {"release": True, "port": True},
        {"release": True, "port": "8000"},
        {"release": True, "port": 8000, "host": 5},
    ],
)
def test_snapshot_release_marker_rejects_an_invalid_payload(
    tmp_path: Path, payload: object
):
    marker = tmp_path / "release.json"
    marker.write_text(json.dumps(payload))

    with pytest.raises(snapshot_server.SnapshotBarrierError, match="release marker"):
        snapshot_server.read_release_marker(marker)


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


def test_snapshot_identity_mismatch_names_every_differing_field():
    with pytest.raises(SnapshotCompatibilityError) as mismatch:
        validate_identity(
            _manifest(),
            _runtime_identity(vllm_version="different", gpu_uuid="GPU-other"),
        )

    assert str(mismatch.value) == "snapshot mismatch: vllm_version, gpu_uuid"


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


@pytest.mark.parametrize("mode", ["pinned", "unpinned", "cleanup-fails"])
def test_snapshot_restore_terminates_only_a_pinned_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    _mark_snapshot_pids_free(monkeypatch)
    artifact = _local_restore_artifact(tmp_path)
    manifest = _manifest()
    root_pid = manifest.process_tree[0]
    tools = LocalSnapshotTools()
    terminated: list[int] = []

    def terminate_and_wait(pid: int) -> None:
        terminated.append(pid)
        if mode == "cleanup-fails":
            raise SnapshotRestoreError("restored process cleanup is incomplete")
        tools._restored_processes.pop(pid)

    def pin(
        _artifact: Path, process_tree: tuple[int, ...], _holders: tuple[int, ...]
    ) -> None:
        if mode == "unpinned":
            raise SnapshotRestoreError(
                "restored session does not match the captured process tree"
            )
        tools._restored_processes[process_tree[0]] = ()

    monkeypatch.setattr(tools, "cleanup", terminate_and_wait)
    monkeypatch.setattr(tools, "_pin_restored_tree", pin)
    monkeypatch.setattr(tools, "_criu", lambda *_args: None)
    monkeypatch.setattr(
        tools,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "333\n", ""),
    )

    with pytest.raises(SnapshotRestoreError, match="does not match") as failure:
        tools.restore(artifact, manifest)

    # A pinned tree is ours: it must be terminated and waited for, not left to
    # honour the abort marker. An unpinned PID is never signalled.
    assert terminated == ([] if mode == "unpinned" else [root_pid])
    if mode == "cleanup-fails":
        # The failure to terminate is reported beside the primary error, and a
        # tree that may still be running stays pinned.
        assert "restore cleanup failed" in str(failure.value)
        assert tools._restored_processes == {root_pid: ()}
    else:
        assert tools._restored_processes == {}
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_state: str, message: str
):
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100, 101))
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("101, GPU-abc",))
    monkeypatch.setattr(
        tools,
        "_descriptor_inventory",
        lambda _tree: (
            (101,) if blocked_state == "io_uring" else (),
            {41: 101},
        ),
    )
    tcp_record = _TcpSocketRecord(
        family="AF_INET",
        local_raw="3C00000A:D6F6",
        remote_raw="01010101:01BB" if blocked_state == "tcp" else "00000000:0000",
        inode=41,
    )
    monkeypatch.setattr(tools, "_tcp_records", lambda: (tcp_record,))
    with pytest.raises(SnapshotCreateError, match=message) as excinfo:
        tools.inventory(100, tmp_path)
    if blocked_state == "tcp":
        assert "pid 101" in str(excinfo.value)
        assert "1.1.1.1:443" in str(excinfo.value)


def test_decode_endpoint_families():
    assert _decode_endpoint("AF_INET", "3C00000A:D6F6") == "10.0.0.60:55030"
    assert (
        _decode_endpoint("AF_INET6", "00000000000000000000000001000000:01BB")
        == "[::1]:443"
    )


def test_tcp_records_skips_missing_tables(tmp_path, monkeypatch):
    tcp, tcp6 = tmp_path / "tcp", tmp_path / "tcp6"
    contents = "header\n0: 0100007F:1F90 00000000:0000 01 0 0 0 0 0 41\n"
    tcp.write_text(contents)
    tables = (("AF_INET", tcp), ("AF_INET6", tcp6))
    monkeypatch.setattr(snapshot_runtime, "_TCP_TABLES", tables)

    tools = LocalSnapshotTools()
    assert tuple(record.family for record in tools._tcp_records()) == ("AF_INET",)

    tcp6.write_text(contents)
    families = tuple(record.family for record in tools._tcp_records())
    assert families == ("AF_INET", "AF_INET6")


def test_snapshot_manifest_records_external_cache_files(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    recorded = (
        (str(tmp_path / "kernel.so"), "b" * 64),
        (str(tmp_path / "launcher.lock"), "locked"),
    )
    write_manifest_atomic(artifact, _manifest(external_cache_files=recorded))

    assert read_manifest(artifact).external_cache_files == recorded

    # A manifest written before this field existed still loads.
    legacy = _manifest().model_dump(mode="json")
    del legacy["external_cache_files"]
    (artifact / "manifest.json").write_text(json.dumps(legacy))

    assert read_manifest(artifact).external_cache_files == ()


def _pin_generated_cache_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the resolver off whatever cache roots the host running this has."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton_cache_default"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor_cache"))


def _unreadable(_path: Path) -> str:
    raise OSError("cache file is unreadable")


def test_snapshot_inventory_records_open_generated_cache_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "triton_cache"
    cache_root.mkdir()
    kernel = cache_root / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    lock = cache_root / "launcher.lock"
    lock.write_bytes(b"held by pid 100")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    (artifact / "child.log.lock").write_bytes(b"")
    _pin_generated_cache_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    targets = {
        100: (str(kernel), str(weights), "socket:[41]"),
        101: (str(lock), str(cache_root), str(artifact / "child.log.lock")),
        102: (f"{cache_root / 'evicted.so'} (deleted)",),
    }
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100, 101, 102))
    monkeypatch.setattr(tools, "_descriptor_targets", lambda pid: targets[pid])
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("101, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())
    monkeypatch.setattr(tools, "current_identity", lambda _uuid: _runtime_identity())

    manifest = tools.make_manifest(
        argparse.Namespace(
            model_tag="Qwen/Qwen3-0.6B",
            revision=_MODEL_REVISION,
            served_model_name=None,
            tokenizer_revision=None,
        ),
        ("Qwen/Qwen3-0.6B",),
        tools.inventory(100, artifact),
        _oracle(),
        artifact,
    )

    # Weights, sockets, the cache directory itself, files inside the artifact,
    # and deleted targets that CRIU remaps are all left out.
    assert manifest.external_cache_files == (
        (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
        (str(lock), "locked"),
    )


def test_snapshot_inventory_marks_a_cache_file_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "triton_cache"
    cache_root.mkdir()
    kernel = cache_root / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    lock = cache_root / "launcher.lock"
    lock.write_bytes(b"held by pid 100")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    _pin_generated_cache_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(
        tools, "_descriptor_targets", lambda _pid: (str(kernel), str(lock))
    )
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())
    monkeypatch.setattr(tools, "_sha256", _unreadable)

    # A file that cannot be digested keeps its own marker, so the manifest says
    # which entries restore can check for existence only.
    assert tools.inventory(100, artifact).external_cache_files == (
        (str(kernel), "unreadable"),
        (str(lock), "locked"),
    )


def test_snapshot_inventory_matches_resolved_cache_roots_not_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Descriptor targets come back kernel-canonical, so the fixture must be too.
    base = Path(os.path.realpath(tmp_path))
    real_cache = base / "mnt_cache"
    real_cache.mkdir()
    kernel = real_cache / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    linked_cache = base / "linked_cache"
    linked_cache.symlink_to(real_cache, target_is_directory=True)
    sibling = base / "vllm_cache-backup"
    sibling.mkdir()
    stale = sibling / "stale.so"
    stale.write_bytes(b"stale kernel")
    artifact = base / "snapshot"
    artifact.mkdir(mode=0o700)

    _pin_generated_cache_roots(monkeypatch, base)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(linked_cache))

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(
        tools, "_descriptor_targets", lambda _pid: (str(kernel), str(stale))
    )
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())

    # A cache root reached through a symlink still covers the file the kernel
    # reports by its real path, and a directory that merely shares the root's
    # name prefix is not part of the cache at all.
    assert tools.inventory(100, artifact).external_cache_files == (
        (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
    )


def test_snapshot_restore_rejects_rotated_cache_file_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    lock = tmp_path / "launcher.lock"
    lock.write_bytes(b"held by pid 100")
    manifest = _manifest(
        external_cache_files=(
            (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
            (str(lock), "locked"),
        )
    )
    artifact = _local_restore_artifact(tmp_path)
    before = {path.name: path.read_bytes() for path in artifact.iterdir()}
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))

    kernel.write_bytes(b"recompiled kernel")
    with pytest.raises(SnapshotRestoreError, match="changed since capture") as changed:
        tools.restore(artifact, manifest)
    assert str(kernel) in str(changed.value)

    # Restoring the digest gets past the first entry, so the lock file is only
    # checked for existence: the second failure proves the first one passed.
    kernel.write_bytes(b"compiled kernel")
    lock.unlink()
    with pytest.raises(SnapshotRestoreError, match="vanished since capture") as gone:
        tools.restore(artifact, manifest)
    assert str(lock) in str(gone.value)

    assert {path.name: path.read_bytes() for path in artifact.iterdir()} == before


def test_snapshot_restore_rejects_a_cache_file_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    manifest = _manifest(
        external_cache_files=(
            (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
        )
    )
    artifact = _local_restore_artifact(tmp_path)
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))
    monkeypatch.setattr(tools, "_sha256", _unreadable)

    # A recorded digest that cannot be read again is a failed check, not a file
    # to wave through on existence.
    with pytest.raises(
        SnapshotRestoreError, match="unreadable since capture"
    ) as blocked:
        tools.restore(artifact, manifest)
    assert str(kernel) in str(blocked.value)


def test_snapshot_runtime_installer_logs_the_failing_exit_code():
    installer = Path(__file__).parents[3] / "tools" / "install_snapshot_runtime.sh"
    if not installer.is_file():
        pytest.skip("installer script is not shipped in this test image")
    timing = [
        line
        for line in installer.read_text().splitlines()
        if line.startswith(("_snapshot_install_started=", "trap "))
    ]
    assert len(timing) == 2

    result = subprocess.run(
        ["bash", "-c", "\n".join([*timing, "exit 7"])],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert "(exit 7)" in result.stdout
