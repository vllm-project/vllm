# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create and restore transaction orchestration for local snapshots."""

import argparse
import shutil
import sys
from pathlib import Path

from vllm.snapshot.manifest import (
    read_manifest,
    validate_artifact_root,
    validate_identity,
)
from vllm.snapshot.runtime import (
    LocalSnapshotTools,
    SnapshotCreateError,
    SnapshotRestoreError,
    _error_detail,
)
from vllm.snapshot.types import Oracle, oracles_match


def _child_engine_argv(
    args: argparse.Namespace, engine_argv: tuple[str, ...] | None
) -> tuple[str, ...]:
    """Build the child argv from the first argv source that is present.

    In order: an explicit ``engine_argv`` from the caller, the process argv
    tail when invoked as ``vllm snapshot create ...``, then the bare model from
    the parsed args. Whichever source wins, snapshot options are stripped and
    sleep mode is appended.
    """
    if not engine_argv:
        process_argv = tuple(sys.argv[1:])
        if process_argv[:2] == ("snapshot", "create"):
            engine_argv = process_argv[2:]
        else:
            model = getattr(args, "model_tag", None) or getattr(args, "model", None)
            if not model:
                raise SnapshotCreateError("snapshot create requires a model argument")
            engine_argv = (str(model),)
    child_argv: list[str] = []
    iterator = iter(engine_argv)
    for item in iterator:
        option, separator, _ = item.partition("=")
        option = option.replace("_", "-")
        if len(option) > 2 and (
            "--snapshot-dir".startswith(option) or "--work-dir".startswith(option)
        ):
            if not separator:
                next(iterator, None)
        elif option == "--portable":
            # The create parser rejects abbreviations, so only the full
            # spelling reaches the child argv.
            continue
        else:
            child_argv.append(item)
    if "--enable-sleep-mode" not in child_argv:
        child_argv.append("--enable-sleep-mode")
    return tuple(child_argv)


def create_snapshot(
    args: argparse.Namespace,
    *,
    engine_argv: tuple[str, ...] | None = None,
    tools: LocalSnapshotTools | None = None,
) -> None:
    target = Path(args.snapshot_dir).absolute()
    toolset = tools or LocalSnapshotTools()
    toolset.preflight("create", target)
    if target.exists() or target.is_symlink():
        raise SnapshotCreateError(f"snapshot target already exists: {target}")
    validate_artifact_root(target, creating=True)
    target.parent.mkdir(mode=0o700, parents=False, exist_ok=True)
    validate_artifact_root(target.parent, creating=False)
    published = False
    root_pid: int | None = None
    work_dir: Path | None = None
    cleanup_work_dir = False
    portable = bool(getattr(args, "portable", False))
    target.mkdir(mode=0o700)
    try:
        validate_artifact_root(target, creating=False)
        work_dir, cleanup_work_dir = toolset.create_work_dir(
            getattr(args, "work_dir", None),
            target,
            portable=portable,
        )
        child_argv = _child_engine_argv(args, engine_argv)
        root_pid = toolset.launch_child(work_dir, child_argv)
        oracle = toolset.wait_ready(work_dir, root_pid)
        inventory = toolset.inventory(root_pid, target, work_dir, portable=portable)
        attachments = toolset.dump(target, work_dir, inventory, portable=portable)
        toolset.verify_dead(inventory)
        manifest = toolset.make_manifest(
            args,
            child_argv,
            inventory,
            oracle,
            target,
            attachments,
            work_dir=work_dir,
            cleanup_work_dir=cleanup_work_dir,
        )
        toolset.publish(target, manifest)
        published = True
    except BaseException:
        if root_pid is not None:
            toolset.abort_create(root_pid)
        raise
    finally:
        # Runs even when abort_create raised, so no partial artifact is left.
        if not published and target.exists():
            shutil.rmtree(target)
        if work_dir is not None and cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def restore_snapshot(
    args: argparse.Namespace, *, tools: LocalSnapshotTools | None = None
) -> None:
    artifact = Path(args.snapshot_dir).absolute()
    toolset = tools or LocalSnapshotTools()
    toolset.preflight("restore", artifact)
    manifest = read_manifest(artifact)
    requested_portable = bool(getattr(args, "portable", False))
    if requested_portable != manifest.portable:
        raise SnapshotRestoreError(
            "snapshot restore requires --portable to match the artifact: "
            f"artifact is {'portable' if manifest.portable else 'in place'}, "
            f"restore requested "
            f"{'portable' if requested_portable else 'in place'}"
        )
    validate_identity(manifest, toolset.current_identity(manifest.gpu_uuid))

    # The manifest is the only source for the work directory: the barrier
    # paths are frozen into the captured argv, so restore takes no --work-dir.
    work_dir = Path(manifest.work_dir)
    root_pid: int | None = None
    failed = False
    try:
        root_pid = toolset.restore(artifact, manifest)
        # The tree is still waiting at the barrier.
        toolset.release(work_dir, args.host, args.port)
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
        failed = True
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
    finally:
        # Like a create, a restore removes a work directory it drew no
        # record for keeping. On success the released tree holds open files
        # there, so their bytes live on as unlinked inodes. A failed restore
        # keeps the directory instead: a tree the abort could not signal —
        # or could not kill — polls its release marker to exit, and the
        # marker is the only signal that directory holds.
        if manifest.cleanup_work_dir and not failed:
            shutil.rmtree(work_dir, ignore_errors=True)
