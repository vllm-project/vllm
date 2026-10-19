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
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec, patch

import pytest

import vllm.snapshot.attachments as snapshot_attachments
import vllm.snapshot.runtime as snapshot_runtime
import vllm.snapshot.server as snapshot_server
from vllm.entrypoints.cli import snapshot as snapshot_cli
from vllm.entrypoints.cli.launch import (
    LaunchSubcommand,
    RenderSubcommand,
    cmd_init,
)
from vllm.snapshot.attachments import (
    GeneratedFile,
    SnapshotAttachmentError,
    capture_attachments,
    collect_generated_files,
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
    SnapshotAttachment,
    SnapshotCompatibilityError,
    SnapshotManifest,
    SnapshotRuntimeIdentity,
    SnapshotSecurityError,
    read_manifest,
    validate_artifact_root,
    validate_identity,
    write_manifest_atomic,
)
from vllm.snapshot.runtime import (
    ProcessInventory,
    _decode_endpoint,
    _TcpSocketRecord,
)
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
    ("configured", "expected"),
    [(None, "1"), ("0", "0"), ("1", "1")],
)
def test_snapshot_child_defaults_nccl_ib_without_mutating_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    expected: str,
):
    if configured is None:
        monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)
    else:
        monkeypatch.setenv("NCCL_IB_DISABLE", configured)
    parent_environment = os.environ.copy()
    process = create_autospec(subprocess.Popen, instance=True)
    process.pid = 100
    popen = create_autospec(subprocess.Popen, return_value=process)
    monkeypatch.setattr(snapshot_runtime.subprocess, "Popen", popen)

    LocalSnapshotTools().launch_child(tmp_path, ("model",))

    assert popen.call_args.kwargs["env"]["NCCL_IB_DISABLE"] == expected
    assert os.environ == parent_environment


def test_snapshot_environment_identity_applies_effective_nccl_ib_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)
    implicit = LocalSnapshotTools()._environment_identity()
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    explicit_disabled = LocalSnapshotTools()._environment_identity()
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    explicit_enabled = LocalSnapshotTools()._environment_identity()

    assert implicit == explicit_disabled
    assert explicit_enabled != implicit


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


def test_snapshot_create_strips_the_portable_option(monkeypatch):
    argv = [
        "create",
        "Qwen/Qwen3-0.6B",
        "--revision",
        _MODEL_REVISION,
        "--portable",
        "--dtype",
        "float16",
    ]
    args = parse_snapshot(*argv)
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", *argv])

    child_argv = list(_child_engine_argv(args, None))

    # The flag carries no value, so the option after it survives.
    assert "--portable" not in child_argv
    assert "--dtype" in child_argv


def test_snapshot_create_strips_the_work_dir_option(monkeypatch):
    argv = [
        "create",
        "Qwen/Qwen3-0.6B",
        "--revision",
        _MODEL_REVISION,
        "--snapshot-dir",
        "/tmp/snapshot",
        "--portable",
        "--work-dir",
        "/tmp/work",
    ]
    args = parse_snapshot(*argv)
    monkeypatch.setattr(sys, "argv", ["vllm", "snapshot", *argv])

    child_argv = list(_child_engine_argv(args, None))

    assert child_argv[0] == "Qwen/Qwen3-0.6B"
    assert "--portable" not in child_argv
    assert "--work-dir" not in child_argv
    assert "/tmp/work" not in child_argv


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


@pytest.mark.parametrize("container_attempted", [False, True])
def test_snapshot_e2e_cleanup_checks_state_only_after_container_attempt(
    container_attempted: bool,
):
    script = (
        Path(__file__).resolve().parents[3]
        / ".buildkite/scripts/initialized-snapshot-e2e.sh"
    )
    if not script.exists():
        pytest.skip("snapshot E2E script is not packaged in this test image")
    source = script.read_text()
    cleanup = source[source.index("cleanup() {") : source.index("on_exit() {")]
    shell = (
        "set -Eeuo pipefail\n"
        "CONTAINER_NAME='' PORT_TWO=18002\n"
        f"CHECK_CLEANUP_STATE={int(container_attempted)}\n"
        "wait_clean() { echo checked-state; return 1; }\n"
        + cleanup
        + "\nif cleanup; then exit 0; else exit 1; fi\n"
    )

    result = subprocess.run(["bash", "-c", shell], capture_output=True, timeout=10)

    assert result.returncode == int(container_attempted), result.stderr
    assert (b"checked-state" in result.stdout) == container_attempted


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
        ("spec_method", "ngram", "does not support speculative decoding"),
        ("spec_model", "draft-model", "does not support speculative decoding"),
        ("spec_tokens", 3, "does not support speculative decoding"),
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
    if changes.get("portable"):
        changes.setdefault("work_dir", "/tmp/vllm-snapshot-work-fixture")
        changes.setdefault("cleanup_work_dir", False)
    else:
        changes.setdefault("work_dir", "/tmp/vllm-snapshot-artifact-fixture")
        changes.setdefault("cleanup_work_dir", False)
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
    tools.create_work_dir.return_value = (Path("/tmp/vllm-snapshot-work"), False)
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


@pytest.mark.parametrize(
    "failure_kind", ["called-process", "timeout", "missing", "unreadable"]
)
def test_snapshot_create_preserves_bounded_criu_diagnostics_after_cleanup(
    tmp_path: Path, failure_kind: str, caplog: pytest.LogCaptureFixture
):
    target = tmp_path / "snapshot"
    tools = _fake_snapshot_tools()
    tools.create_work_dir.return_value = (target, False)
    criu_tools = LocalSnapshotTools()
    criu_tools.plugin_dir = tmp_path
    original_run = criu_tools._run
    criu_tools._privileged = lambda: []  # type: ignore[method-assign]
    captured = "captured stderr " + "x" * 9000
    if failure_kind == "timeout":
        primary: subprocess.SubprocessError = subprocess.TimeoutExpired(
            ["criu", "dump"], 1, output=b"captured stdout\xff", stderr=captured.encode()
        )
    else:
        primary = subprocess.CalledProcessError(
            1, ["criu", "dump"], output="captured stdout", stderr=captured
        )

    def run(command: list[str], **_kwargs: object):
        if "tail" not in command:
            raise primary
        if failure_kind == "unreadable":
            raise PermissionError("log is root-owned")
        if failure_kind == "timeout":
            return subprocess.CompletedProcess(command, 0, "timeout log tail", "")
        return original_run(command, **_kwargs)

    criu_tools._run = run  # type: ignore[method-assign]

    def fail_dump(
        artifact: Path,
        work_dir: Path,
        _inventory: ProcessInventory,
        *,
        portable: bool = False,
    ) -> None:
        images = artifact / "images"
        images.mkdir()
        if failure_kind != "missing":
            (work_dir / "dump.log").write_text(
                "omitted dump prefix" + "d" * 9000 + "distinctive dump diagnostic"
            )
        criu_tools._criu("dump", artifact, [], work_dir)

    tools.dump.side_effect = fail_dump
    with pytest.raises(type(primary)) as excinfo:
        create_snapshot(
            argparse.Namespace(snapshot_dir=str(target), model_tag="Qwen/Qwen3-0.6B"),
            tools=tools,
        )

    assert excinfo.value is primary
    assert not target.exists()
    diagnostics = "\n".join(caplog.messages)
    assert "captured stdout" in diagnostics
    assert "captured stderr" not in diagnostics
    assert "x" * 100 in diagnostics
    assert len(diagnostics) < 17000
    if failure_kind in {"missing", "unreadable"}:
        assert "dump.log unavailable" in diagnostics
        expected_error = (
            "CalledProcessError" if failure_kind == "missing" else "PermissionError"
        )
        assert expected_error in diagnostics
    elif failure_kind == "called-process":
        assert "distinctive dump diagnostic" in diagnostics
        assert "omitted dump prefix" not in diagnostics
    else:
        assert "timeout log tail" in diagnostics
    if failure_kind == "timeout":
        assert "captured stdout�" in diagnostics


def test_snapshot_criu_failure_selects_restore_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    tools = LocalSnapshotTools()
    tools.plugin_dir = tmp_path
    tools._privileged = lambda: []  # type: ignore[method-assign]
    artifact = tmp_path / "snapshot"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)
    (work_dir / "restore.log").write_text(
        "omitted restore prefix" + "r" * 9000 + "distinctive restore diagnostic"
    )
    primary = subprocess.CalledProcessError(1, ["criu", "restore"])
    original_run = tools._run
    tailed: list[str] = []

    def run(command: list[str], **_kwargs: object):
        if "tail" not in command:
            raise primary
        tailed.append(command[-1])
        return original_run(command, **_kwargs)

    tools._run = run  # type: ignore[method-assign]

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        tools._criu("restore", artifact, [], work_dir)

    assert excinfo.value is primary
    assert tailed == [str(work_dir / "restore.log")]
    diagnostics = "\n".join(caplog.messages)
    assert "distinctive restore diagnostic" in diagnostics
    assert "omitted restore prefix" not in diagnostics


def test_snapshot_create_removes_a_temporary_work_dir_when_a_later_step_fails(
    tmp_path: Path,
):
    target = tmp_path / "snapshot"
    work_dir = tmp_path / "work"
    tools = _fake_snapshot_tools()
    tools.create_work_dir.return_value = (work_dir, True)
    tools.dump.side_effect = RuntimeError("dump failed")

    with pytest.raises(RuntimeError, match="dump failed"):
        create_snapshot(
            argparse.Namespace(
                snapshot_dir=str(target), model_tag="Qwen/Qwen3-0.6B", portable=True
            ),
            tools=tools,
        )

    # Neither the partial artifact nor the temporary work directory it drew
    # survives the failure.
    assert not target.exists()
    assert not work_dir.exists()


def test_snapshot_create_keeps_a_named_work_dir_when_a_later_step_fails(
    tmp_path: Path,
):
    target = tmp_path / "snapshot"
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    tools = _fake_snapshot_tools()
    tools.create_work_dir.return_value = (work_dir, False)
    tools.dump.side_effect = RuntimeError("dump failed")

    with pytest.raises(RuntimeError, match="dump failed"):
        create_snapshot(
            argparse.Namespace(
                snapshot_dir=str(target), model_tag="Qwen/Qwen3-0.6B", portable=True
            ),
            tools=tools,
        )

    assert not target.exists()
    assert work_dir.exists()


def test_snapshot_create_cleans_up_when_the_work_dir_is_refused(tmp_path: Path):
    target = tmp_path / "snapshot"
    tools = _fake_snapshot_tools()
    tools.create_work_dir.side_effect = SnapshotCreateError("work dir refused")

    with pytest.raises(SnapshotCreateError, match="work dir refused"):
        create_snapshot(
            argparse.Namespace(snapshot_dir=str(target), model_tag="Qwen/Qwen3-0.6B"),
            tools=tools,
        )

    # The refusal happened after the artifact directory was created, so the
    # finally block still removes it and a retry is not blocked.
    assert not target.exists()
    tools.launch_child.assert_not_called()


def _restore_fixture(tmp_path: Path) -> argparse.Namespace:
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(artifact, _manifest(work_dir=str(artifact)))
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


@pytest.mark.parametrize(
    ("manifest_portable", "requested_portable"),
    [
        (True, None),
        (True, False),
        (False, True),
    ],
)
def test_snapshot_restore_requires_portable_to_match(
    tmp_path: Path,
    manifest_portable: bool,
    requested_portable: bool | None,
):
    args = _restore_fixture(tmp_path)
    write_manifest_atomic(
        tmp_path / "snapshot", _manifest(portable=manifest_portable)
    )
    if requested_portable is not None:
        setattr(args, "portable", requested_portable)

    mismatch = _fake_snapshot_tools()
    with pytest.raises(SnapshotRestoreError, match="requires --portable to match"):
        restore_snapshot(args, tools=mismatch)
    mismatch.restore.assert_not_called()


def test_snapshot_restore_accepts_matching_portable(tmp_path: Path):
    args = _restore_fixture(tmp_path)
    write_manifest_atomic(tmp_path / "snapshot", _manifest(portable=True))
    args.portable = True

    tools = _fake_snapshot_tools()
    restore_snapshot(args, tools=tools)
    tools.restore.assert_called_once()


def test_snapshot_manifest_records_whether_the_work_dir_is_cleaned_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    external = tmp_path / "work"
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "current_identity", lambda _uuid: _runtime_identity())
    monkeypatch.setattr(tools, "_image_pstree", lambda *_args: Pstree((100,), (100,)))
    base = argparse.Namespace(
        model_tag="Qwen/Qwen3-0.6B",
        revision=_MODEL_REVISION,
        served_model_name=None,
        tokenizer_revision=None,
        portable=True,
    )

    manifest = tools.make_manifest(
        base,
        ("Qwen/Qwen3-0.6B",),
        ProcessInventory(100, (100,), (100,), "GPU-abc"),
        _oracle(),
        artifact,
        work_dir=external,
        cleanup_work_dir=False,
    )
    assert manifest.work_dir == str(external)
    assert manifest.cleanup_work_dir is False

    base.portable = False
    manifest = tools.make_manifest(
        base,
        ("Qwen/Qwen3-0.6B",),
        ProcessInventory(100, (100,), (100,), "GPU-abc"),
        _oracle(),
        artifact,
    )
    assert manifest.work_dir == str(artifact)
    assert manifest.cleanup_work_dir is False


def test_snapshot_restore_releases_through_the_manifest_work_dir(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(
        artifact, _manifest(portable=True, work_dir="/tmp/snap-work")
    )
    args = argparse.Namespace(
        snapshot_dir=str(artifact), host="127.0.0.1", port=9000, portable=True
    )

    tools = _fake_snapshot_tools()
    restore_snapshot(args, tools=tools)

    # The barrier marker belongs to the restoring machine's copy of the work
    # directory, never to the shared artifact.
    tools.release.assert_called_once_with(Path("/tmp/snap-work"), "127.0.0.1", 9000)


def test_snapshot_create_work_dir_requires_portable():
    args = parse_snapshot(
        "create",
        "Qwen/Qwen3-0.6B",
        "--revision",
        _MODEL_REVISION,
        "--snapshot-dir",
        "/tmp/snapshot",
        "--work-dir",
        "/tmp/work",
    )

    with pytest.raises(ValueError, match="requires --portable"):
        snapshot_cli.validate_create_args(args)


def test_snapshot_create_work_dir_in_place_layout(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)

    tools = LocalSnapshotTools()
    work_dir, cleanup = tools.create_work_dir(None, artifact, portable=False)
    assert (work_dir, cleanup) == (artifact, False)


@pytest.mark.parametrize("requested", [None, "explicit"])
def test_snapshot_create_work_dir_stays_outside_the_artifact(
    tmp_path: Path, requested: str | None
):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    tools = LocalSnapshotTools()
    request = None if requested is None else str(tmp_path / requested)

    work_dir, cleanup = tools.create_work_dir(request, artifact, portable=True)

    assert not work_dir.is_relative_to(artifact)
    assert work_dir.stat().st_mode & 0o777 == 0o700
    assert cleanup is (request is None)
    if request is None:
        assert work_dir.is_relative_to(Path(tempfile.gettempdir()))
    with pytest.raises(SnapshotCreateError, match="must stay outside"):
        tools.create_work_dir(str(artifact), artifact, portable=True)
    with pytest.raises(SnapshotCreateError, match="must stay outside"):
        tools.create_work_dir(str(artifact / "work"), artifact, portable=True)
    with pytest.raises(SnapshotCreateError, match="already exists"):
        tools.create_work_dir(str(work_dir), artifact, portable=True)
    shutil.rmtree(work_dir)


def test_snapshot_portable_create_records_and_discards_an_automatic_work_dir(
    tmp_path: Path,
):
    target = tmp_path / "snapshot"
    work_dir = tmp_path / "work"
    tools = _fake_snapshot_tools()

    def make_work_dir(_requested, _artifact, *, portable):
        work_dir.mkdir(mode=0o700)
        return work_dir, True

    tools.create_work_dir.side_effect = make_work_dir
    create_snapshot(
        argparse.Namespace(
            snapshot_dir=str(target),
            model_tag="Qwen/Qwen3-0.6B",
            portable=True,
        ),
        tools=tools,
    )

    tools.create_work_dir.assert_called_once_with(None, target, portable=True)
    assert tools.launch_child.call_args[0][0] == work_dir
    assert tools.make_manifest.call_args[1]["work_dir"] == work_dir
    # The manifest is the work directory's only record, so create leaves no
    # automatic copy behind on the machine that built the artifact.
    assert not work_dir.exists()


def test_snapshot_restore_discards_a_temporary_work_dir_when_done(
    tmp_path: Path,
):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(
        artifact,
        _manifest(
            portable=True, work_dir=str(tmp_path / "work"), cleanup_work_dir=True
        ),
    )
    args = argparse.Namespace(
        snapshot_dir=str(artifact), host="127.0.0.1", port=9000, portable=True
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)

    tools = _fake_snapshot_tools()
    restore_snapshot(args, tools=tools)

    # The tree it leaves behind holds open files there; the directory itself
    # is gone, like the one a create with no --work-dir leaves behind.
    assert not work_dir.exists()


def test_snapshot_restore_keeps_a_work_dir_the_operator_named(
    tmp_path: Path,
):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    write_manifest_atomic(
        artifact,
        _manifest(
            portable=True, work_dir=str(tmp_path / "work"), cleanup_work_dir=False
        ),
    )
    args = argparse.Namespace(
        snapshot_dir=str(artifact), host="127.0.0.1", port=9000, portable=True
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)

    tools = _fake_snapshot_tools()
    restore_snapshot(args, tools=tools)

    assert work_dir.is_dir()


def test_snapshot_restore_reuses_a_work_dir_after_a_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    (work_dir / "release.json").write_text('{"release": true}')
    (work_dir / "restored.pid").write_text("100\n")
    (work_dir / "child.log").write_bytes(b"x" * 5)
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    (artifact / "child.log.snapshot-size").write_text("5\n")
    manifest = _manifest(portable=True, work_dir=str(work_dir))
    tools = LocalSnapshotTools()
    criu_arguments: list[list[str]] = []

    def fake_criu(
        _action: str, _artifact: Path, arguments: list[str], _work_dir: Path
    ) -> None:
        criu_arguments.append(list(arguments))
        pidfile = Path(arguments[arguments.index("--pidfile") + 1])
        pidfile.write_text("100\n")

    def fake_run(command: list[str], **_kwargs: object):
        return subprocess.CompletedProcess(command, 0, "100\n", "")

    monkeypatch.setattr(tools, "_criu", fake_criu)
    monkeypatch.setattr(tools, "_run", fake_run)
    monkeypatch.setattr(tools, "_pin_restored_tree", lambda *_args: None)

    assert tools.restore(artifact, manifest) == 100

    # The earlier run's transaction files are replaced, so their stale
    # contents cannot reach this run's barrier or PID check.
    assert criu_arguments[0][criu_arguments[0].index("--pidfile") + 1] == str(
        work_dir / "restored.pid"
    )
    assert (work_dir / "restored.pid").read_text() == "100\n"


def test_snapshot_restore_populates_a_fresh_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    (work_dir / "child.log").write_bytes(b"x" * 5)
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    (artifact / "child.log.snapshot-size").write_text("5\n")
    manifest = _manifest(portable=True, work_dir=str(work_dir))
    tools = LocalSnapshotTools()
    criu_arguments: list[list[str]] = []

    def fake_criu(
        _action: str, _artifact: Path, arguments: list[str], _work_dir: Path
    ) -> None:
        criu_arguments.append(list(arguments))
        pidfile = Path(arguments[arguments.index("--pidfile") + 1])
        pidfile.write_text("100\n")

    def fake_run(command: list[str], **_kwargs: object):
        assert command[-1].endswith("restored.pid")
        return subprocess.CompletedProcess(command, 0, "100\n", "")

    monkeypatch.setattr(tools, "_criu", fake_criu)
    monkeypatch.setattr(tools, "_run", fake_run)
    monkeypatch.setattr(tools, "_pin_restored_tree", lambda *_args: None)

    assert tools.restore(artifact, manifest) == 100

    # An empty work directory is taken over as found, the control files land
    # in it, and CRIU keeps its scratch state out of the artifact.
    arguments = criu_arguments[0]
    assert arguments[arguments.index("--pidfile") + 1] == str(
        work_dir / "restored.pid"
    )
    assert arguments[arguments.index("--work-dir") + 1] == str(work_dir)
    assert "restored.pid" in [path.name for path in work_dir.iterdir()]
    assert (work_dir / "child.log").stat().st_size == 5
    assert [path.name for path in artifact.iterdir()] == ["child.log.snapshot-size"]


def test_snapshot_restore_refuses_a_work_dir_it_does_not_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    # mkdir filters the mode through umask, so loosen it explicitly.
    work_dir.chmod(0o777)
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    manifest = _manifest(portable=True, work_dir=str(work_dir))

    with pytest.raises(SnapshotRestoreError, match="not a private directory"):
        LocalSnapshotTools().restore(artifact, manifest)


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
        ({"portable": "yes"}, "portable"),
        ({"cleanup_work_dir": True}, "root"),
        (
            {
                "portable": True,
                "work_dir": "relative/path",
                "cleanup_work_dir": False,
            },
            "root",
        ),
        ({"work_dir": 5}, "work_dir"),
        ({"cleanup_work_dir": "yes"}, "cleanup_work_dir"),
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
    manifest = _manifest(work_dir=str(artifact))
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
        tools.restore(artifact, _manifest(work_dir=str(artifact)))


@pytest.mark.parametrize("inventory_error", [None, PermissionError("denied")])
def test_snapshot_dump_rejects_unavailable_link_remap_inventory_before_criu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory_error: PermissionError | None,
):
    tools = LocalSnapshotTools()
    workdir = tmp_path / "snapshot"
    workdir.mkdir()
    tools.shm_dir = tmp_path / "missing-shm"
    if inventory_error is not None:
        tools.shm_dir.mkdir()

        def fail_inventory(_path: Path):
            raise inventory_error

        monkeypatch.setattr(Path, "iterdir", fail_inventory)
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))

    with pytest.raises(SnapshotCreateError, match="inventory CRIU link remaps"):
        tools.dump(workdir, workdir, ProcessInventory(100, (100,), (100,), "GPU"))


def test_snapshot_restore_rejects_saved_remap_without_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    artifact = _local_restore_artifact(tmp_path)
    saved = artifact / "link-remaps"
    saved.mkdir()
    (saved / "270").write_bytes(b"invalid remap name")
    tools = LocalSnapshotTools()
    tools.shm_dir = tmp_path / "shm"
    tools.shm_dir.mkdir()
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))

    with pytest.raises(SnapshotRestoreError, match="invalid saved link remap"):
        tools.restore(artifact, _manifest())

    assert not tuple(tools.shm_dir.iterdir())


def test_snapshot_link_remap_inventory_matches_only_numeric_suffixes(tmp_path: Path):
    for name in (
        "link_remap.0",
        "link_remap.270",
        "link_remap.",
        "link_remap.12a",
        "link-remap.12",
        "prefix.link_remap.12",
        "270",
    ):
        (tmp_path / name).touch()
    tools = LocalSnapshotTools()
    tools.shm_dir = tmp_path

    assert tools._link_remap_names() == {"link_remap.0", "link_remap.270"}


def test_snapshot_dump_preserves_primary_failure_when_cleanup_inventory_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tools = LocalSnapshotTools()
    workdir = tmp_path / "snapshot"
    workdir.mkdir()
    inventory_calls = 0

    def inventory_names() -> set[str]:
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls > 1:
            raise SnapshotCreateError(
                "could not inventory CRIU link remaps: PermissionError: denied"
            )
        return set()

    monkeypatch.setattr(tools, "_link_remap_names", inventory_names)

    def fail_dump(*_args):
        raise RuntimeError("dump failed")

    monkeypatch.setattr(tools, "_criu", fail_dump)

    with pytest.raises(SnapshotCreateError) as failure:
        tools.dump(workdir, workdir, ProcessInventory(100, (100,), (100,), "GPU"))

    assert "RuntimeError: dump failed" in str(failure.value)
    assert "cleanup inventory failed" in str(failure.value)


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
        tools.inventory(100, tmp_path, tmp_path)
    if blocked_state == "tcp":
        assert "pid 101" in str(excinfo.value)
        assert "1.1.1.1:443" in str(excinfo.value)


def test_snapshot_rejects_open_infiniband_device_before_criu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100, 101))
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("101, GPU-abc",))
    monkeypatch.setattr(
        tools,
        "_descriptor_targets",
        lambda pid: ("/dev/infiniband/uverbs7",) if pid == 101 else (),
    )
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))

    with pytest.raises(SnapshotCreateError) as failure:
        tools.inventory(100, tmp_path)

    message = str(failure.value)
    assert "pid 101" in message
    assert "/dev/infiniband/uverbs7" in message
    assert "NCCL_IB_DISABLE=1" in message


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
        (str(tmp_path / "launcher.lock"), "a" * 64),
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
    monkeypatch.setenv("FLASHINFER_CACHE_DIR", str(tmp_path / "flashinfer_cache"))
    monkeypatch.setattr(
        snapshot_attachments, "_MACHINE_STATE_ROOTS", (str(tmp_path / "libuuid"),)
    )


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
    monkeypatch.setattr(
        snapshot_attachments, "fd_targets", lambda pid: targets[pid]
    )
    monkeypatch.setattr(snapshot_attachments, "_mapping_targets", lambda _pid: ())
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("101, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())
    monkeypatch.setattr(tools, "current_identity", lambda _uuid: _runtime_identity())
    inventory = tools.inventory(100, artifact, artifact)

    manifest = tools.make_manifest(
        argparse.Namespace(
            model_tag="Qwen/Qwen3-0.6B",
            revision=_MODEL_REVISION,
            served_model_name=None,
            tokenizer_revision=None,
        ),
        ("Qwen/Qwen3-0.6B",),
        inventory,
        _oracle(),
        artifact,
    )

    # Weights, sockets, the cache root directory itself, files inside the
    # artifact, and deleted targets that CRIU remaps are all left out. A lock
    # file is recorded by existence only, since its content is volatile.
    assert manifest.external_cache_files == (
        (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
        (str(lock), "locked"),
    )
    # An in-place checkpoint records no generated files: attachments are a
    # portable create's inventory.
    assert inventory.generated_files == ()


def test_snapshot_carries_libuuid_clock_file_with_cache_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    libuuid_root = tmp_path / "libuuid"
    libuuid_root.mkdir()
    clock = libuuid_root / "clock.txt"
    clock.write_bytes(b"clock: 3610 tv: 0000001789454500 00707119 adj: 0\n")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    _pin_generated_cache_roots(monkeypatch, tmp_path)

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(
        snapshot_attachments, "fd_targets", lambda _pid: (str(clock),)
    )
    monkeypatch.setattr(snapshot_attachments, "_mapping_targets", lambda _pid: ())
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())

    inventory = tools.inventory(100, artifact, tmp_path / "work", portable=True)

    # libuuid keeps the clock file open for the process lifetime, so CRIU
    # reopens its path on restore; the watermark is machine state, so it
    # travels as warm state instead of a fingerprinted dependency.
    assert inventory.generated_files == (
        GeneratedFile(str(clock)),
        GeneratedFile(str(tmp_path / "work" / "child.log")),
    )


def test_snapshot_attachment_target_check_allows_libuuid_state_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from vllm.snapshot.attachments import _check_target

    # The real root is uuidd-owned and group-writable by design, so the
    # private-path contract cannot hold for it; a sibling directory with the
    # same permissions outside the root stays refused. The mode is set
    # explicitly: mkdir() filters it through the process umask, and the
    # directory has to stay group-writable whatever umask the host runs with.
    state_root = tmp_path / "libuuid"
    state_root.mkdir()
    state_root.chmod(0o775)
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    sibling.chmod(0o775)
    monkeypatch.setattr(
        snapshot_attachments, "_MACHINE_STATE_ROOTS", (str(state_root),)
    )

    _check_target(state_root)
    _check_target(state_root / "clock.txt")

    with pytest.raises(SnapshotSecurityError, match="group- or world-writable"):
        _check_target(sibling / "clock.txt")


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
        snapshot_attachments,
        "fd_targets",
        lambda _pid: (str(kernel), str(lock)),
    )
    monkeypatch.setattr(snapshot_attachments, "_mapping_targets", lambda _pid: ())
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())
    monkeypatch.setattr(snapshot_attachments, "sha256_file", _unreadable)

    # A file that cannot be digested keeps its own marker, so the manifest says
    # which entries restore can check for existence only.
    assert tools.inventory(100, artifact, artifact).external_cache_files == (
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
        snapshot_attachments,
        "fd_targets",
        lambda _pid: (str(kernel), str(stale)),
    )
    monkeypatch.setattr(snapshot_attachments, "_mapping_targets", lambda _pid: ())
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())

    # A cache root reached through a symlink still covers the file the kernel
    # reports by its real path, and a directory that merely shares the root's
    # name prefix is not part of the cache at all.
    assert tools.inventory(100, artifact, artifact).external_cache_files == (
        (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
    )


def test_snapshot_inventory_records_a_generated_file_the_tree_only_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "triton_cache"
    cache_root.mkdir()
    kernel = cache_root / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    _pin_generated_cache_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(snapshot_attachments, "fd_targets", lambda _pid: ())
    monkeypatch.setattr(
        snapshot_attachments, "_mapping_targets", lambda _pid: (str(kernel),)
    )
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())

    # A kernel reached only through its pages holds no descriptor. CRIU still
    # reopens its path, so without this the artifact stayed pointed at a file
    # the host could replace with no early error.
    inventory = tools.inventory(100, artifact, tmp_path / "work", portable=True)
    assert inventory.generated_files == (
        GeneratedFile(
            str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()
        ),
        GeneratedFile(str(tmp_path / "work" / "child.log")),
    )


def test_snapshot_collect_refuses_an_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "triton_cache"
    cache_root.mkdir()
    kernel = cache_root / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    _pin_generated_cache_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(
        snapshot_attachments, "fd_targets", lambda _pid: (str(kernel),)
    )
    monkeypatch.setattr(
        snapshot_attachments, "_mapping_targets", lambda _pid: ()
    )
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())

    # The limit is enforced while the inventory is assembled, so a cache
    # holding a file past it fails create before CRIU runs at all.
    with pytest.raises(SnapshotAttachmentError, match="byte limit"):
        collect_generated_files(
            (100,),
            artifact,
            tmp_path / "work",
            tuple(),
            (16, 1 << 30),
        )


def test_snapshot_collect_refuses_a_total_size_past_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "triton_cache"
    cache_root.mkdir()
    kernel = cache_root / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    stale = cache_root / "stale.so"
    stale.write_bytes(b"unreferenced compiled kernel")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    _pin_generated_cache_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(
        snapshot_attachments, "fd_targets", lambda _pid: (str(kernel),)
    )
    monkeypatch.setattr(
        snapshot_attachments, "_mapping_targets", lambda _pid: ()
    )
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())

    with pytest.raises(SnapshotAttachmentError, match="byte limit"):
        collect_generated_files(
            (100,),
            artifact,
            tmp_path / "work",
            tuple(),
            (1 << 30, 20),
        )


def test_snapshot_capture_refuses_a_file_that_changed_since_inventory(
    tmp_path: Path,
):
    cache = tmp_path / "triton_cache"
    cache.mkdir()
    kernel = cache / "kernel.so"
    kernel.write_bytes(b"a different kernel")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)

    # The inventory digested one version, then something rewrote the file
    # before capture copied it: staging the new bytes would record a digest
    # the fingerprints can never verify at restore.
    with pytest.raises(SnapshotAttachmentError, match="changed since inventory"):
        capture_attachments(
            artifact,
            (
                GeneratedFile(
                    str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()
                ),
            ),
        )


def test_snapshot_dump_stages_attachments_after_the_image_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = tmp_path / "triton_cache"
    cache.mkdir()
    kernel = cache / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    kernel.chmod(0o755)
    lock = cache / "launcher.lock"
    lock.write_bytes(b"held by the steady point")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    child_log = artifact / "child.log"
    child_log.parent.mkdir(mode=0o700)
    child_log.write_bytes(b"")
    staged_before_dump: list[bool] = []

    def record_dump(
        _action: str, _artifact: Path, _arguments: list[str], _work_dir: Path
    ) -> None:
        staged_before_dump.append((artifact / "attachments").exists())

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_criu", record_dump)
    monkeypatch.setattr(tools, "_link_remap_names", lambda: set())
    monkeypatch.setattr(tools, "_capture_link_remaps", lambda *_args: None)
    inventory = ProcessInventory(
        100,
        (100,),
        (100,),
        "GPU-abc",
        generated_files=(
            GeneratedFile(str(child_log)),
            GeneratedFile(str(kernel)),
            GeneratedFile(str(lock)),
        ),
    )

    captured = tools.dump(artifact, artifact, inventory, portable=True)

    # Nothing was staged while the captured tree was still running to rewrite it.
    assert staged_before_dump == [False]
    assert [item.path for item in captured] == [
        str(child_log),
        str(kernel),
        str(lock),
    ]
    assert captured[0].digest == "cache"
    staged = artifact / "attachments" / Path(str(kernel).lstrip("/"))
    assert staged.read_bytes() == b"compiled kernel"
    assert staged.stat().st_mode & 0o777 == 0o755


def test_snapshot_inventory_carries_unreferenced_cache_files_as_warm_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "triton_cache"
    cache_root.mkdir()
    kernel = cache_root / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    stale = cache_root / "stale.so"
    stale.write_bytes(b"unreferenced compiled kernel")
    lock = cache_root / "launcher.lock"
    lock.write_bytes(b"held by pid 100")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    _pin_generated_cache_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(
        snapshot_attachments,
        "fd_targets",
        lambda _pid: (str(kernel), str(lock)),
    )
    monkeypatch.setattr(snapshot_attachments, "_mapping_targets", lambda _pid: ())
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())

    inventory = tools.inventory(100, artifact, tmp_path / "work", portable=True)

    # A file the image does not reference still travels as warm state, marked
    # differently from the bytes the restore itself depends on.
    assert inventory.generated_files == (
        GeneratedFile(
            str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()
        ),
        GeneratedFile(
            str(lock), hashlib.sha256(b"held by pid 100").hexdigest()
        ),
        GeneratedFile(str(stale)),
        GeneratedFile(str(tmp_path / "work" / "child.log")),
    )
    # The warm-state entries carry no fingerprint: a target machine's own
    # compiled state may differ without breaking anything.


def test_snapshot_in_place_checkpoint_walks_no_cache_and_carries_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "triton_cache"
    cache_root.mkdir()
    kernel = cache_root / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    stale = cache_root / "stale.so"
    stale.write_bytes(b"unreferenced compiled kernel")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    (artifact / "child.log").write_bytes(b"")
    _pin_generated_cache_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_tree_pids", lambda _root_pid: (100,))
    monkeypatch.setattr(
        snapshot_attachments, "fd_targets", lambda _pid: (str(kernel),)
    )
    monkeypatch.setattr(snapshot_attachments, "_mapping_targets", lambda _pid: ())
    monkeypatch.setattr(tools, "_cuda_process_rows", lambda: ("100, GPU-abc",))
    monkeypatch.setattr(tools, "_tcp_records", lambda: ())
    monkeypatch.setattr(tools, "_criu", lambda *_args: None)
    monkeypatch.setattr(tools, "_link_remap_names", lambda: set())
    monkeypatch.setattr(tools, "_capture_link_remaps", lambda *_args: None)

    # Without --portable the cache roots are not walked: only the files the
    # tree holds open are fingerprinted, and nothing is carried at all.
    inventory = tools.inventory(100, artifact, artifact)
    assert inventory.generated_files == ()
    assert inventory.external_cache_files == (
        (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
    )
    assert tools.dump(artifact, artifact, inventory) == ()
    assert not (artifact / "attachments").exists()


def test_snapshot_materialize_leaves_a_targets_own_cache_state_alone(
    tmp_path: Path,
):
    from vllm.snapshot.attachments import materialize_attachments

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    compiled = target / "compiled.so"
    compiled.write_bytes(b"target machine compiled this")
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    captured = capture_attachments(
        artifact, (GeneratedFile(str(compiled)),)
    )
    manifest_entry = captured[0]

    # A cache attachment does not back the restore, so a target's own content
    # is never overwritten, whatever it holds.
    assert materialize_attachments(artifact, (manifest_entry,)) == []
    assert compiled.read_bytes() == b"target machine compiled this"

    # Only a missing file is written back.
    compiled.unlink()
    assert materialize_attachments(artifact, (manifest_entry,)) == [compiled]
    assert compiled.read_bytes() == b"target machine compiled this"


def test_snapshot_restore_rematerializes_a_cache_file_the_host_rotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    artifact = _local_restore_artifact(tmp_path)
    tools = LocalSnapshotTools()
    captured = capture_attachments(
        artifact, (GeneratedFile(str(kernel)),)
    )
    manifest = _manifest(attachments=tuple(captured), work_dir=str(artifact))
    kernel.unlink()
    reopened: list[bool] = []

    class Restored(RuntimeError):
        """Stands in for CRIU so the test can see what it was handed."""

    def fail_restore(*_args: object) -> None:
        reopened.append(kernel.exists())
        raise Restored("criu stopped here")

    monkeypatch.setattr(tools, "_criu", fail_restore)

    with pytest.raises(Restored, match="criu stopped here"):
        tools.restore(artifact, manifest)

    # The path was back before CRIU ran, then taken away again because this
    # restore attempt failed.
    assert reopened == [True]
    assert not kernel.exists()
    marker = json.loads((artifact / "release.json").read_text())
    assert marker == {"release": False}


def test_snapshot_restore_rematerializes_a_work_dir_child_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    child_log = work_dir / "child.log"
    child_log.write_bytes(b"startup log")
    artifact = _local_restore_artifact(tmp_path)
    tools = LocalSnapshotTools()
    captured = capture_attachments(
        artifact, (GeneratedFile(str(child_log)),)
    )
    manifest = _manifest(
        portable=True,
        work_dir=str(work_dir),
        cleanup_work_dir=False,
        attachments=tuple(captured),
    )
    # A work directory a create (or an earlier failed restore) left behind
    # carries no log, so restore has to put the captured one back before CRIU
    # reopens it by path.
    child_log.unlink()
    reopened: list[bool] = []

    class Restored(RuntimeError):
        """Stands in for CRIU so the test can see what it was handed."""

    def fail_restore(*_args: object) -> None:
        reopened.append(child_log.exists())
        raise Restored("criu stopped here")

    monkeypatch.setattr(tools, "_criu", fail_restore)

    with pytest.raises(Restored, match="criu stopped here"):
        tools.restore(artifact, manifest)

    # The log was back before CRIU ran, then taken away again because this
    # restore attempt failed.
    assert reopened == [True]
    assert not child_log.exists()


def test_snapshot_manifest_round_trips_attachments(tmp_path: Path):
    artifact = tmp_path / "snapshot"
    artifact.mkdir(mode=0o700)
    attachments = (
        SnapshotAttachment(
            path="/cache/kernel.so",
            digest="a" * 64,
            size=108936,
        ),
    )
    write_manifest_atomic(artifact, _manifest(attachments=attachments))

    assert read_manifest(artifact).attachments == attachments

    # An artifact captured before this field existed still restores, carrying
    # nothing.
    legacy = _manifest().model_dump(mode="json")
    del legacy["attachments"]
    (artifact / "manifest.json").write_text(json.dumps(legacy))

    assert read_manifest(artifact).attachments == ()


def test_snapshot_restore_rejects_rotated_cache_file_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    lock = tmp_path / "launcher.lock"
    lock.write_bytes(b"held by pid 100")
    artifact = _local_restore_artifact(tmp_path)
    manifest = _manifest(
        work_dir=str(artifact),
        external_cache_files=(
            (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
            (str(lock), hashlib.sha256(b"held by pid 100").hexdigest()),
        ),
    )
    before = {
        path.name: path.read_bytes()
        for path in artifact.iterdir()
        if path.is_file()
    }
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))

    kernel.write_bytes(b"recompiled kernel")
    with pytest.raises(SnapshotRestoreError, match="changed since capture") as changed:
        tools.restore(artifact, manifest)
    assert str(kernel) in str(changed.value)

    # Restoring the digest gets past the first entry, so the missing lock file
    # is the second failure: it proves the first one passed.
    kernel.write_bytes(b"compiled kernel")
    lock.unlink()
    with pytest.raises(SnapshotRestoreError, match="vanished since capture") as gone:
        tools.restore(artifact, manifest)
    assert str(lock) in str(gone.value)

    # A refused restore leaves the image alone and marks it unreleased, so CRIU
    # never sees a half-checked artifact.
    assert {
        path.name: path.read_bytes()
        for path in artifact.iterdir()
        if path.is_file()
    } == before
    assert json.loads((artifact / "release.json").read_text()) == {
        "release": False
    }


def test_snapshot_restore_rejects_a_cache_file_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mark_snapshot_pids_free(monkeypatch)
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(b"compiled kernel")
    artifact = _local_restore_artifact(tmp_path)
    manifest = _manifest(
        work_dir=str(artifact),
        external_cache_files=(
            (str(kernel), hashlib.sha256(b"compiled kernel").hexdigest()),
        ),
    )
    tools = LocalSnapshotTools()
    monkeypatch.setattr(tools, "_criu", lambda *_args: pytest.fail("CRIU called"))
    monkeypatch.setattr(snapshot_attachments, "sha256_file", _unreadable)

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
