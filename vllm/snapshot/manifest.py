# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
import stat
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError


class SnapshotError(RuntimeError):
    """Base error for snapshot operations."""


class SnapshotCompatibilityError(SnapshotError):
    """The snapshot identity does not match the requested runtime."""


class SnapshotSecurityError(SnapshotError):
    """The snapshot artifact does not meet the private-path contract."""


class SocketIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    family: str
    socket_type: str
    local_address: str
    remote_address: str | None
    state: str


_MAX_PID = 2**31 - 1
_PositivePid = Annotated[int, Field(gt=0, le=_MAX_PID)]
_OracleTokenId = Annotated[int, Field(ge=0)]


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    boundary: Literal[
        "post-engine-init-pre-http-bind",
        "post-engine-init-reloadable-state-released",
    ]
    complete: Literal[True]
    created_at: str
    artifact_bytes: Annotated[int, Field(ge=0)]
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
    process_tree: Annotated[tuple[_PositivePid, ...], Field(min_length=1)]
    cuda_holders: Annotated[tuple[_PositivePid, ...], Field(min_length=1)]
    socket_inventory: tuple[SocketIdentity, ...]
    oracle_token_ids: Annotated[
        tuple[_OracleTokenId, ...], Field(min_length=1, max_length=1)
    ]
    oracle_text: str
    oracle_sampled_token_logprob: float

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_schema_version_one(cls, value: Any) -> Any:
        if type(value) is not int or value != 1:
            raise ValueError("schema version must be integer 1")
        return value

    @field_validator("complete", mode="before")
    @classmethod
    def _require_complete_true(cls, value: Any) -> Any:
        if type(value) is not bool or value is not True:
            raise ValueError("complete must be boolean true")
        return value

    @field_validator("oracle_sampled_token_logprob", mode="before")
    @classmethod
    def _require_finite_float(cls, value: Any) -> Any:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("sampled token log probability must be a finite float")
        return value

    @model_validator(mode="after")
    def _validate_process_relationships(self) -> Self:
        if len(set(self.process_tree)) != len(self.process_tree):
            raise PydanticCustomError(
                "snapshot_process_tree", "process_tree contains duplicate PIDs"
            )
        if len(set(self.cuda_holders)) != len(self.cuda_holders):
            raise PydanticCustomError(
                "snapshot_cuda_holders", "cuda_holders contains duplicate PIDs"
            )
        if not set(self.cuda_holders).issubset(self.process_tree):
            raise PydanticCustomError(
                "snapshot_cuda_holders",
                "cuda_holders must be a subset of process_tree",
            )
        return self


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


def validate_identity(expected: SnapshotManifest, actual: SnapshotManifest) -> None:
    """Require an exact match for every field that can affect compatibility."""
    for field_name in SnapshotManifest.model_fields:
        if field_name in _NON_IDENTITY_FIELDS:
            continue
        if getattr(expected, field_name) != getattr(actual, field_name):
            raise SnapshotCompatibilityError(f"snapshot mismatch: {field_name}")


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


_ORACLE_FIELDS = frozenset(
    {"oracle_token_ids", "oracle_text", "oracle_sampled_token_logprob"}
)
_ROOT_VALIDATION_FIELDS = {
    "snapshot_process_tree": "process_tree",
    "snapshot_cuda_holders": "cuda_holders",
}


def _validation_diagnostic(error: ValidationError) -> str | None:
    for detail in error.errors(include_url=False):
        error_type = detail["type"]
        location = detail["loc"]
        if error_type == "json_invalid":
            return "JSON"
        if error_type == "model_type" and not location:
            return "root"
        if error_type in _ROOT_VALIDATION_FIELDS:
            return _ROOT_VALIDATION_FIELDS[error_type]
        if not location:
            continue
        field = location[0]
        if len(location) == 1 and error_type in {"extra_forbidden", "missing"}:
            return "fields"
        if field in _ORACLE_FIELDS:
            return "oracle"
        if isinstance(field, str):
            return field
    return None


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
            manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
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
            payload = manifest_file.read()
    except UnicodeDecodeError as error:
        raise SnapshotCompatibilityError("invalid snapshot manifest: JSON") from error
    try:
        return SnapshotManifest.model_validate_json(payload, strict=True)
    except ValidationError as error:
        diagnostic = _validation_diagnostic(error)
        suffix = f": {diagnostic}" if diagnostic is not None else ""
        raise SnapshotCompatibilityError(
            f"invalid snapshot manifest{suffix}"
        ) from error


def inspect_snapshot(path: Path) -> dict[str, object]:
    inspected: dict[str, object] = read_manifest(path).model_dump(mode="json")
    inspected["support_boundary"] = "same-host Linux x86_64 TP1"
    inspected["private_artifact_path_validated"] = True
    return inspected
