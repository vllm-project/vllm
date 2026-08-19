# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create and restore transaction orchestration for local snapshots."""

import argparse
import shutil
import sys
from pathlib import Path

from vllm.snapshot import runtime as _runtime
from vllm.snapshot.manifest import SnapshotRuntimeIdentity as SnapshotRuntimeIdentity
from vllm.snapshot.manifest import validate_artifact_root, validate_identity
from vllm.snapshot.types import Oracle, oracles_match

LocalSnapshotTools = _runtime.LocalSnapshotTools
ProcessInventory = _runtime.ProcessInventory
SnapshotCreateError = _runtime.SnapshotCreateError
SnapshotRestoreError = _runtime.SnapshotRestoreError
_TcpSocketRecord = _runtime._TcpSocketRecord
_error_detail = _runtime._error_detail


def _remove_snapshot_option(argv: tuple[str, ...]) -> tuple[str, ...]:
    remaining: list[str] = []
    iterator = iter(argv)
    for item in iterator:
        if item == "--snapshot-dir":
            next(iterator, None)
        elif item.startswith("--snapshot-dir=") or item == "--include-model-state":
            continue
        else:
            remaining.append(item)
    return tuple(remaining)


def _current_engine_argv(args: argparse.Namespace) -> tuple[str, ...]:
    process_argv = tuple(sys.argv[1:])
    if len(process_argv) >= 2 and process_argv[:2] == ("snapshot", "create"):
        return process_argv[2:]
    model = getattr(args, "model_tag", None) or getattr(args, "model", None)
    if not model:
        raise SnapshotCreateError("snapshot create requires a model argument")
    return (str(model),)


def create_snapshot(
    args: argparse.Namespace,
    *,
    engine_argv: tuple[str, ...] | None = None,
    tools: "LocalSnapshotTools | None" = None,
) -> None:
    target = Path(args.snapshot_dir).absolute()
    toolset = tools or LocalSnapshotTools()
    toolset.preflight("create", target)
    if target.exists() or target.is_symlink():
        raise SnapshotCreateError(f"snapshot target already exists: {target}")
    validate_artifact_root(target, creating=True)
    target.parent.mkdir(mode=0o700, parents=False, exist_ok=True)
    validate_artifact_root(target.parent, creating=False)
    target.mkdir(mode=0o700)
    validate_artifact_root(target, creating=False)
    published = False
    root_pid: int | None = None
    try:
        child_argv = _remove_snapshot_option(engine_argv or _current_engine_argv(args))
        include_model_state = bool(getattr(args, "include_model_state", False))
        if not include_model_state and "--enable-sleep-mode" not in child_argv:
            child_argv = (*child_argv, "--enable-sleep-mode")
        root_pid = toolset.launch_child(
            target,
            child_argv,
            include_model_state=include_model_state,
        )
        oracle = toolset.wait_ready(target, root_pid)
        inventory = toolset.inventory(root_pid)
        toolset.dump(target, inventory)
        toolset.verify_dead(inventory)
        manifest = toolset.make_manifest(args, child_argv, inventory, oracle, target)
        toolset.publish(target, manifest)
        published = True
    except BaseException:
        if root_pid is not None:
            toolset.abort_create(root_pid)
        raise
    finally:
        if not published and target.exists():
            shutil.rmtree(target)


def restore_snapshot(
    args: argparse.Namespace, *, tools: "LocalSnapshotTools | None" = None
) -> None:
    artifact = Path(args.snapshot_dir).absolute()
    toolset = tools or LocalSnapshotTools()
    toolset.preflight("restore", artifact)
    from vllm.snapshot.manifest import read_manifest

    manifest = read_manifest(artifact)
    current = toolset.current_identity(manifest.gpu_uuid)
    validate_identity(manifest, current)

    root_pid: int | None = None
    try:
        root_pid = toolset.restore(artifact, manifest)
        toolset.release(artifact, args.host, args.port)
        toolset.wait_listener(root_pid, args.host, args.port)
        actual = toolset.request_oracle(args.host, args.port, manifest)
        expected = Oracle(
            token_ids=manifest.oracle_token_ids,
            text=manifest.oracle_text,
            sampled_token_logprob=manifest.oracle_sampled_token_logprob,
        )
        if not oracles_match(expected, actual):
            raise SnapshotRestoreError(
                f"snapshot oracle mismatch: expected={expected!r}, actual={actual!r}"
            )
        toolset.complete_restore(root_pid)
    except BaseException as error:
        if root_pid is not None:
            try:
                toolset.cleanup(root_pid)
            except BaseException as cleanup_error:
                raise SnapshotRestoreError(
                    f"snapshot restore failed: {_error_detail(error)}; "
                    f"cleanup failed: {_error_detail(cleanup_error)}"
                ) from error
        if isinstance(error, SnapshotRestoreError):
            raise
        raise SnapshotRestoreError(str(error)) from error
