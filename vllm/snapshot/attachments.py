# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Carry the generated files a snapshot image reopens by path."""

import hashlib
import os
import shutil
import stat
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vllm.snapshot.manifest import (
    SnapshotAttachment,
    SnapshotSecurityError,
    _require_trusted_path_chain,
)

ATTACHMENT_DIR = "attachments"
_DIGEST_CHUNK = 1 << 20


def _blob_path(artifact: Path, path: Path) -> Path:
    """Where the artifact carries a captured file: its own path mirrored
    under the attachment root, so each cache root keeps its tree and no two
    files can collide."""
    return artifact / ATTACHMENT_DIR / Path(*path.parts[1:])
# Roots of machine state a captured process pins open: the image reopens
# these paths, but the bytes are the host's own, so restore only puts back a
# missing one and never compares content. util-linux's libuuid opens
# /var/lib/libuuid/clock.txt once and keeps the descriptor for the process
# lifetime, and the watermark it holds differs between hosts by design; the
# directory is uuidd-owned and group-writable, so the private-path contract
# cannot hold for it.
_MACHINE_STATE_ROOTS = (os.path.realpath("/var/lib/libuuid"),)
DEFAULT_MAX_FILE_BYTES = 1 << 30
DEFAULT_MAX_TOTAL_BYTES = 1 << 30
MAX_FILE_BYTES_ENV = "VLLM_SNAPSHOT_ATTACHMENT_MAX_FILE_BYTES"
MAX_TOTAL_BYTES_ENV = "VLLM_SNAPSHOT_ATTACHMENT_MAX_TOTAL_BYTES"


def _is_machine_state(path: str) -> bool:
    """Whether ``path`` names a machine-state root or something inside it."""
    return any(
        path == root or path.startswith(f"{root}{os.sep}")
        for root in _MACHINE_STATE_ROOTS
    )


def _env_limit(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        limit = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer number of bytes") from error
    if limit <= 0:
        raise ValueError(f"{name} must be a positive number of bytes")
    return limit


def attachment_limits() -> tuple[int, int]:
    """Per-file and total attachment size limits, from the environment."""
    return (
        _env_limit(MAX_FILE_BYTES_ENV, DEFAULT_MAX_FILE_BYTES),
        _env_limit(MAX_TOTAL_BYTES_ENV, DEFAULT_MAX_TOTAL_BYTES),
    )


@dataclass(frozen=True)
class GeneratedFile:
    """A file considered for snapshot capture and its content fingerprint.

    ``fingerprint`` is a SHA-256 hex digest for content verification,
    ``None`` when restore skips content verification, or
    ``_CACHE_UNREADABLE`` when the file could not be read during capture.
    """

    path: str
    fingerprint: str | None = None


def fd_targets(pid: int) -> tuple[str, ...]:
    """Return the symlink targets of entries in /proc/<pid>/fd."""
    descriptor_dir = Path("/proc") / str(pid) / "fd"
    try:
        descriptors = tuple(descriptor_dir.iterdir())
    except FileNotFoundError as error:
        raise SnapshotAttachmentError(
            f"snapshot process exited during descriptor inventory: {pid}"
        ) from error
    targets: list[str] = []
    for descriptor in descriptors:
        try:
            targets.append(os.readlink(descriptor))
        except FileNotFoundError:
            continue
    return tuple(targets)


def _mapping_targets(pid: int) -> tuple[str, ...]:
    """Read file-backed mapping paths from /proc/<pid>/maps.

    File mappings can outlive their file descriptors, so scanning
    /proc/<pid>/fd alone may miss files CRIU needs to reopen during restore.
    """
    maps = Path("/proc") / str(pid) / "maps"
    try:
        payload = maps.read_text()
    except FileNotFoundError as error:
        raise SnapshotAttachmentError(
            f"snapshot process exited during mapping inventory: {pid}"
        ) from error
    targets: list[str] = []
    for line in payload.splitlines():
        # permissions offset dev inode pathname
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            targets.append(fields[5])
    return tuple(targets)


def _path_references(pid: int) -> tuple[tuple[str, str], ...]:
    """Collect descriptor targets and mapped file paths for snapshot capture.

    Each entry is a (target, kind) pair, where kind is "descriptor" or
    "mapping", identifying how the process references the target.
    """
    referenced = [(target, "descriptor") for target in fd_targets(pid)]
    referenced += [(target, "mapping") for target in _mapping_targets(pid)]
    return tuple(referenced)


def _cache_root_directories(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Expand cache path prefixes into directories to scan for attachments.

    Paths ending in "/" are used directly. Other paths are matched against
    directory names in the parent directory: "/tmp/torchinductor_" matches
    all directories named "torchinductor_*" under "/tmp".
    """
    roots: list[str] = list(_MACHINE_STATE_ROOTS)
    for prefix in prefixes:
        if prefix.endswith(os.sep):
            roots.append(prefix.rstrip(os.sep))
            continue
        parent = Path(prefix).parent
        stem = Path(prefix).name
        roots.extend(str(path) for path in parent.glob(f"{stem}*") if path.is_dir())
    return tuple(sorted(set(roots)))


def collect_generated_files(
    process_tree: tuple[int, ...],
    artifact: Path,
    work_dir: Path,
    cache_prefixes: tuple[str, ...],
    limits: tuple[int, int] | None = None,
) -> tuple[GeneratedFile, ...]:
    """Collect files to include as snapshot attachments.

    Include cache files opened or mapped by the process tree, plus additional
    cache files to avoid recompilation after restore. Referenced cache files
    are fingerprinted when readable; additional cache files are included
    without content verification.

    Also include the child log and files under known machine-state roots.
    Only paths under a cache prefix are in scope, so model weights and files
    some unrelated helper holds stay out; files inside the artifact belong to
    the snapshot itself, and a deleted target is already covered by CRIU's
    link remap. Deduplicate paths and enforce per-file and total size limits.
    """
    if limits is None:
        limits = attachment_limits()
    artifact_prefix = f"{os.path.abspath(artifact)}{os.sep}"
    references = _referenced_generated_files(
        process_tree, artifact_prefix, cache_prefixes
    )
    # The image reopens the child log by path as well, and the work
    # directory holding it does not travel with the artifact. Cache
    # semantics fit: a log an earlier restore appended to is left
    # alone, and the restoring reset crops it back.
    child_log = str(work_dir / "child.log")
    references[child_log] = GeneratedFile(child_log)
    files = {
        **references,
        **_walked_cache_files(references, artifact_prefix, cache_prefixes),
    }
    ordered = _enforce_size_limits(sorted(files.items()), limits)
    return tuple(entry for _path, entry in ordered)


def _enforce_size_limits(
    ordered: list[tuple[str, GeneratedFile]], limits: tuple[int, int]
) -> list[tuple[str, GeneratedFile]]:
    """Enforce per-file and total size limits for snapshot attachments.

    Raise SnapshotAttachmentError if any file or the combined total exceeds
    the corresponding limit in bytes. The check runs on the assembled
    inventory, so a cache that does not fit fails create before CRIU runs.
    """
    max_file_bytes, max_total_bytes = limits
    total = 0
    for path, _entry in ordered:
        size = os.stat(path).st_size
        if size > max_file_bytes:
            raise SnapshotAttachmentError(
                f"attachment exceeds the {max_file_bytes} byte limit: {path}; "
                f"point the cache root into the artifact instead"
            )
        total += size
        if total > max_total_bytes:
            raise SnapshotAttachmentError(
                f"attachments exceed the {max_total_bytes} byte limit at {path}; "
                f"point the cache root into the artifact instead"
            )
    return ordered


def _referenced_generated_files(
    process_tree: tuple[int, ...],
    artifact_prefix: str,
    prefixes: tuple[str, ...],
) -> dict[str, GeneratedFile]:
    """Collect cache and machine-state files referenced by the process tree.

    Inspect open file descriptors and file-backed mappings to identify files
    to include in the snapshot. Skip deleted files and paths inside the
    snapshot artifact.
    """
    references: dict[str, GeneratedFile] = {}
    for pid in process_tree:
        for target, _source in _path_references(pid):
            if target.endswith(" (deleted)") or target.startswith(artifact_prefix):
                continue
            if target in references:
                # Reached through several pids, or through both a descriptor
                # and a mapping; one fingerprint of the bytes is enough.
                continue
            if _is_machine_state(target):
                references[target] = GeneratedFile(target)
                continue
            if not target.startswith(prefixes):
                continue
            references[target] = GeneratedFile(
                target, cache_fingerprint(Path(target))
            )
    return references


def _walked_cache_files(
    references: dict[str, GeneratedFile],
    artifact_prefix: str,
    prefixes: tuple[str, ...],
) -> dict[str, GeneratedFile]:
    """Collect unreferenced cache files for reuse after restore.

    Scan cache directories, skipping files already in ``references`` and
    paths inside the snapshot artifact; an in-place checkpoint keeps every
    unreferenced cache file on the host instead. Record these files without
    fingerprints so restore can reuse existing cache files without content
    verification; one that is not a readable regular file stays out.
    """
    walked: dict[str, GeneratedFile] = {}
    for root in _cache_root_directories(prefixes):
        for directory, _subdirs, files in os.walk(root, followlinks=False):
            for name in files:
                target = str(Path(directory) / name)
                if target in references or target.startswith(artifact_prefix):
                    continue
                target_fingerprint = cache_fingerprint(Path(target))
                if target_fingerprint is None:
                    # Non-regular or unreadable files are not carryable;
                    # leave them for the host rather than failing create.
                    continue
                walked[target] = GeneratedFile(target)
    return walked


class SnapshotAttachmentError(RuntimeError):
    """An attachment could not be captured or reproduced faithfully."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


# Recorded instead of a digest for files whose content is volatile by design.
_CACHE_LOCKED = "locked"
# Recorded instead of a digest when the bytes are warm state restore never
# verifies; a target that already has its own is left alone.
_CACHE = "cache"
# Recorded instead of a digest for a file capture found but could not read.
_CACHE_UNREADABLE = "unreadable"


def cache_fingerprint(path: Path) -> str | None:
    """Digest a cache file, or mark the ones capture cannot read."""
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            return None
    except OSError:
        return None
    try:
        return sha256_file(path)
    except OSError:
        # Restore can only check that this one still exists, so record the
        # downgrade rather than passing it off as a volatile file.
        return _CACHE_UNREADABLE


def _regular_stat(path: Path) -> os.stat_result:
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode):
        raise SnapshotSecurityError(f"attachment is a symlink: {path}")
    if not stat.S_ISREG(path_stat.st_mode):
        raise SnapshotSecurityError(f"attachment is not a regular file: {path}")
    return path_stat


def _check_target(path: Path) -> None:
    """Validate that the attachment target has a trusted path chain.

    Paths under known machine-state roots are exempt from this check.
    """
    if _is_machine_state(str(path)):
        # Machine state the whole host already shares through the uuid-runtime
        # package; the regular-file check below still applies to it.
        return
    _require_trusted_path_chain(path, what="attachment target")


def _install(source: Path, target: Path, *, fsync: bool) -> None:
    """Copy ``source`` onto ``target`` through a same-directory temporary.

    ``fsync`` publishes the bytes durably; a transient restore target skips
    it, because a restore that dies mid-copy rolls the file back anyway.
    """
    temporary = target.parent / f".{target.name}.partial"
    try:
        shutil.copyfile(source, temporary)
        if fsync:
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IMODE(source.lstat().st_mode))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _stage(source: Path, blob: Path) -> Path:
    """Copy a validated payload into the artifact and return where it landed."""
    _install(source, blob, fsync=True)
    return blob


def capture_attachments(
    artifact: Path,
    generated_files: Iterable[GeneratedFile],
) -> list[SnapshotAttachment]:
    """Copy files into the snapshot and record their paths, sizes, and digests.

    If a copied file differs from its recorded SHA-256 digest, abort capture.
    If an I/O or security check fails, or the tree let go of the file before
    the image was written, skip it and print the reason to stderr: CRIU will
    still need that file on the restore host if it references it.

    Return metadata for the files successfully captured.
    """
    artifact = Path(artifact)
    attachments: list[SnapshotAttachment] = []
    for file in sorted(generated_files, key=lambda file: file.path):
        path = Path(file.path)
        staged = _blob_path(artifact, path)
        try:
            _check_target(path)
            _regular_stat(path)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged = _stage(path, staged)
            digest = _CACHE if file.fingerprint is None else sha256_file(staged)
            attachment = SnapshotAttachment(
                path=str(path),
                digest=digest,
                size=staged.stat().st_size,
            )
            if (
                file.fingerprint is not None
                and file.fingerprint != _CACHE_UNREADABLE
                and file.fingerprint != attachment.digest
            ):
                raise SnapshotAttachmentError(
                    f"attachment changed since inventory: {path}"
                )
        except (OSError, SnapshotSecurityError) as error:
            print(f"snapshot attachment skipped: {path}: {error}", file=sys.stderr)
            continue
        attachments.append(attachment)
    return attachments


def materialize_attachments(
    artifact: Path, attachments: Iterable[SnapshotAttachment]
) -> list[Path]:
    """Prepare snapshot attachments at their original paths for restore.

    Copy files from the artifact only when the target is missing. For existing
    files, skip content verification for ``cache`` attachments and require
    all other attachments to match their recorded SHA-256 digests.

    Return the newly created file paths. On failure, attempt to remove those
    files and report any cleanup failures.
    """
    created: list[Path] = []
    artifact = Path(artifact)
    try:
        for attachment in attachments:
            target = Path(attachment.path)
            _check_target(target)
            source = _blob_path(artifact, Path(attachment.path))
            if not source.exists():
                raise SnapshotAttachmentError(
                    f"attachment payload is missing from the artifact: {target}"
                )
            if target.exists():
                _regular_stat(target)
                if (
                    attachment.digest == _CACHE
                    or sha256_file(target) == attachment.digest
                ):
                    continue
                raise SnapshotAttachmentError(
                    f"attachment target file differs since capture: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            # The first walk only saw components that already existed.
            _check_target(target.parent)
            _install(source, target, fsync=False)
            created.append(target)
    except BaseException as error:
        leftovers = rollback_attachments(created)
        if leftovers:
            raise SnapshotAttachmentError(
                f"{error}; attachments left behind: {', '.join(leftovers)}"
            ) from error
        raise
    return created


def rollback_attachments(created: Iterable[Path]) -> list[str]:
    """Remove only what this restore wrote."""
    failures: list[str] = []
    for path in reversed(list(created)):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            failures.append(f"{path}: {error}")
    return failures
