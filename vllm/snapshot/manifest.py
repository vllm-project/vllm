# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast


class SnapshotError(RuntimeError):
    """Base error for snapshot operations."""


class SnapshotCompatibilityError(SnapshotError):
    """The snapshot identity does not match the requested runtime."""


class SnapshotSecurityError(SnapshotError):
    """The snapshot artifact does not meet the private-path contract."""


@dataclass(frozen=True)
class SocketIdentity:
    family: str
    socket_type: str
    local_address: str
    remote_address: str | None
    state: str


@dataclass(frozen=True)
class SnapshotManifest:
    schema_version: int
    boundary: str
    complete: bool
    created_at: str
    artifact_bytes: int
    source_revision: str
    binary_revision: str
    python_version: str
    torch_version: str
    cuda_runtime: str
    driver_version: str
    criu_version: str
    cuda_checkpoint_sha256: str
    kernel_release: str
    host_id: str
    gpu_name: str
    gpu_uuid: str
    model: str
    served_model_name: str
    model_revision: str
    tokenizer_revision: str
    engine_argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    process_tree: tuple[int, ...]
    cuda_holders: tuple[int, ...]
    socket_inventory: tuple[SocketIdentity, ...]
    oracle_token_ids: tuple[int, ...]
    oracle_text: str
    oracle_sampled_token_logprob: float


_NON_IDENTITY_FIELDS = frozenset(
    {
        "complete",
        "created_at",
        "artifact_bytes",
        "process_tree",
        "cuda_holders",
        "socket_inventory",
        "oracle_token_ids",
        "oracle_text",
        "oracle_sampled_token_logprob",
        "served_model_name",
    }
)
_MAX_PID = 2**31 - 1


def validate_identity(expected: SnapshotManifest, actual: SnapshotManifest) -> None:
    """Require an exact match for every field that can affect compatibility."""
    for field in dataclasses.fields(SnapshotManifest):
        if field.name in _NON_IDENTITY_FIELDS:
            continue
        if getattr(expected, field.name) != getattr(actual, field.name):
            raise SnapshotCompatibilityError(f"snapshot mismatch: {field.name}")


def _existing_path_chain(path: Path):
    current = path
    while True:
        if current.exists() or current.is_symlink():
            yield current
        if current.parent == current:
            break
        current = current.parent


def _is_trusted_sticky_directory(path_stat: os.stat_result) -> bool:
    return (
        path_stat.st_uid == 0
        and stat.S_ISDIR(path_stat.st_mode)
        and bool(path_stat.st_mode & stat.S_ISVTX)
    )


def validate_artifact_root(path: Path, *, creating: bool) -> None:
    """Validate that an artifact path cannot be replaced by another user."""
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        raise SnapshotSecurityError(f"snapshot path is a symlink: {path}")
    if not creating and not path.exists():
        raise SnapshotSecurityError(f"snapshot path does not exist: {path}")

    effective_uid = os.geteuid()
    for component in _existing_path_chain(path):
        component_stat = component.lstat()
        if stat.S_ISLNK(component_stat.st_mode):
            raise SnapshotSecurityError(
                f"snapshot path contains a symlink: {component}"
            )
        if component_stat.st_uid not in (0, effective_uid):
            raise SnapshotSecurityError(
                f"snapshot path has an untrusted owner: {component}"
            )
        if component_stat.st_mode & (
            stat.S_IWGRP | stat.S_IWOTH
        ) and not _is_trusted_sticky_directory(component_stat):
            raise SnapshotSecurityError(
                f"snapshot path has a group- or world-writable ancestor: {component}"
            )

    if path.exists():
        path_stat = path.lstat()
        if not stat.S_ISDIR(path_stat.st_mode):
            raise SnapshotSecurityError(f"snapshot path is not a directory: {path}")
        if path_stat.st_uid != effective_uid:
            raise SnapshotSecurityError(
                f"snapshot directory is not owned by the current user: {path}"
            )
        if stat.S_IMODE(path_stat.st_mode) & 0o077:
            raise SnapshotSecurityError(
                f"snapshot directory must have mode 0700: {path}"
            )


def _manifest_to_dict(manifest: SnapshotManifest) -> dict[str, object]:
    return dataclasses.asdict(manifest)


def _invalid_manifest(field: str) -> NoReturn:
    raise SnapshotCompatibilityError(f"invalid snapshot manifest: {field}")


def _validate_manifest_dict(value: dict[str, object]) -> None:
    expected_fields = {field.name for field in dataclasses.fields(SnapshotManifest)}
    if set(value) != expected_fields:
        _invalid_manifest("fields")

    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        _invalid_manifest("schema_version")
    if value["complete"] is not True:
        _invalid_manifest("complete")
    boundary = value["boundary"]
    if type(boundary) is not str or boundary not in {
        "post-engine-init-pre-http-bind",
        "post-engine-init-reloadable-state-released",
    }:
        _invalid_manifest("boundary")

    artifact_bytes = value["artifact_bytes"]
    if type(artifact_bytes) is not int or artifact_bytes < 0:
        _invalid_manifest("artifact_bytes")
    for field in (
        "created_at",
        "source_revision",
        "binary_revision",
        "python_version",
        "torch_version",
        "cuda_runtime",
        "driver_version",
        "criu_version",
        "cuda_checkpoint_sha256",
        "kernel_release",
        "host_id",
        "gpu_name",
        "gpu_uuid",
        "model",
        "served_model_name",
        "model_revision",
        "tokenizer_revision",
    ):
        if type(value[field]) is not str:
            _invalid_manifest(field)

    engine_argv = value["engine_argv"]
    if type(engine_argv) is not list or not all(
        type(item) is str for item in engine_argv
    ):
        _invalid_manifest("engine_argv")
    environment = value["environment"]
    if type(environment) is not list or not all(
        type(item) is list
        and len(item) == 2
        and all(type(part) is str for part in item)
        for item in environment
    ):
        _invalid_manifest("environment")

    process_tree = value["process_tree"]
    process_tree_values = cast(list[int], process_tree)
    if (
        type(process_tree) is not list
        or not process_tree
        or any(type(pid) is not int or not 0 < pid <= _MAX_PID for pid in process_tree)
        or len(set(process_tree)) != len(process_tree)
    ):
        _invalid_manifest("process_tree")
    cuda_holders = value["cuda_holders"]
    cuda_holder_values = cast(list[int], cuda_holders)
    if (
        type(cuda_holders) is not list
        or not cuda_holders
        or any(type(pid) is not int or not 0 < pid <= _MAX_PID for pid in cuda_holders)
        or len(set(cuda_holders)) != len(cuda_holders)
        or not set(cuda_holder_values).issubset(process_tree_values)
    ):
        _invalid_manifest("cuda_holders")

    socket_inventory = value["socket_inventory"]
    socket_fields = {field.name for field in dataclasses.fields(SocketIdentity)}
    if type(socket_inventory) is not list:
        _invalid_manifest("socket_inventory")
    for item in cast(list[dict[str, object]], socket_inventory):
        if type(item) is not dict or set(item) != socket_fields:
            _invalid_manifest("socket_inventory")
        if any(
            type(item[field]) is not str for field in socket_fields - {"remote_address"}
        ) or (
            item["remote_address"] is not None
            and type(item["remote_address"]) is not str
        ):
            _invalid_manifest("socket_inventory")

    token_ids = value["oracle_token_ids"]
    oracle_text = value["oracle_text"]
    logprob = value["oracle_sampled_token_logprob"]
    if (
        type(token_ids) is not list
        or len(token_ids) != 1
        or type(token_ids[0]) is not int
        or token_ids[0] < 0
        or type(oracle_text) is not str
        or type(logprob) is not float
        or not math.isfinite(logprob)
    ):
        _invalid_manifest("oracle")


def _manifest_from_dict(value: dict[str, object]) -> SnapshotManifest:
    value = dict(value)
    _validate_manifest_dict(value)
    engine_argv = cast(list[str], value["engine_argv"])
    environment = cast(list[list[str]], value["environment"])
    socket_inventory = cast(list[dict[str, Any]], value["socket_inventory"])
    value["engine_argv"] = tuple(str(item) for item in engine_argv)
    value["environment"] = tuple((str(key), str(item)) for key, item in environment)
    value["process_tree"] = tuple(cast(list[int], value["process_tree"]))
    value["cuda_holders"] = tuple(cast(list[int], value["cuda_holders"]))
    value["socket_inventory"] = tuple(
        SocketIdentity(**item) for item in socket_inventory
    )
    value["oracle_token_ids"] = tuple(cast(list[int], value["oracle_token_ids"]))
    return SnapshotManifest(**value)  # type: ignore[arg-type]


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_manifest_atomic(path: Path, manifest: SnapshotManifest) -> None:
    """Write a new private manifest and make its directory entry durable."""
    path = Path(path)
    validate_artifact_root(path, creating=False)
    destination = path / "manifest.json"
    temporary = path / "manifest.json.tmp"
    if destination.exists() or destination.is_symlink():
        raise SnapshotSecurityError(f"manifest already exists: {destination}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(temporary, flags, 0o600)
    except FileExistsError as error:
        raise SnapshotSecurityError(
            f"manifest temporary file already exists: {temporary}"
        ) from error

    try:
        payload = json.dumps(
            _manifest_to_dict(manifest), sort_keys=True, separators=(",", ":")
        ).encode()
        with os.fdopen(file_descriptor, "wb", closefd=True) as manifest_file:
            manifest_file.write(payload)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary, destination)
        _fsync_directory(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_manifest(path: Path) -> SnapshotManifest:
    path = Path(path)
    validate_artifact_root(path, creating=False)
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise SnapshotSecurityError(f"manifest is a symlink: {manifest_path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(manifest_path, flags)
    except FileNotFoundError as error:
        raise SnapshotSecurityError(
            f"manifest does not exist: {manifest_path}"
        ) from error
    try:
        with os.fdopen(file_descriptor, encoding="utf-8") as manifest_file:
            value = json.load(manifest_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SnapshotCompatibilityError("invalid snapshot manifest: JSON") from error
    if not isinstance(value, dict):
        _invalid_manifest("root")
    try:
        return _manifest_from_dict(value)
    except SnapshotCompatibilityError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        raise SnapshotCompatibilityError("invalid snapshot manifest") from error


def inspect_snapshot(path: Path) -> dict[str, object]:
    inspected = dataclasses.asdict(read_manifest(path))
    inspected["support_boundary"] = "same-host Linux x86_64 TP1"
    inspected["private_artifact_path_validated"] = True
    return inspected
