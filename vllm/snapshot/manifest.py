# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
import stat
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self


class SnapshotCompatibilityError(RuntimeError):
    """The snapshot identity does not match the requested runtime."""


class SnapshotSecurityError(RuntimeError):
    """The snapshot artifact does not meet the private-path contract."""


_MAX_PID = 2**31 - 1
_PositivePid = Annotated[int, Field(gt=0, le=_MAX_PID)]
_OracleTokenId = Annotated[int, Field(ge=0)]


class SnapshotRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    vllm_version: str
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
    environment: tuple[tuple[str, str], ...]


class SnapshotManifest(SnapshotRuntimeIdentity):
    schema_version: Literal[1]
    boundary: Literal["post-engine-init-reloadable-state-released"]
    created_at: str
    artifact_bytes: Annotated[int, Field(ge=0)]
    model: str
    served_model_name: str
    model_revision: str
    tokenizer_revision: str
    engine_argv: tuple[str, ...]
    process_tree: Annotated[tuple[_PositivePid, ...], Field(min_length=1)]
    cuda_holders: Annotated[tuple[_PositivePid, ...], Field(min_length=1)]
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

    @field_validator("oracle_sampled_token_logprob", mode="before")
    @classmethod
    def _require_finite_float(cls, value: Any) -> Any:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("sampled token log probability must be a finite float")
        return value

    @model_validator(mode="after")
    def _validate_process_relationships(self) -> Self:
        if len(set(self.process_tree)) != len(self.process_tree):
            raise ValueError("process_tree contains duplicate PIDs")
        if len(set(self.cuda_holders)) != len(self.cuda_holders):
            raise ValueError("cuda_holders contains duplicate PIDs")
        if not set(self.cuda_holders).issubset(self.process_tree):
            raise ValueError("cuda_holders must be a subset of process_tree")
        return self


def validate_identity(
    expected: SnapshotRuntimeIdentity, actual: SnapshotRuntimeIdentity
) -> None:
    """Require an exact match for every field that can affect compatibility."""
    for field_name in SnapshotRuntimeIdentity.model_fields:
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


def _validation_path(error: ValidationError) -> str:
    detail = error.errors(include_url=False)[0]
    location = detail["loc"]
    if not location:
        return "JSON" if detail["type"] == "json_invalid" else "root"
    return ".".join(map(str, location))


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: Path, payload: object, *, overwrite: bool = False) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if overwrite:
        temporary.unlink(missing_ok=True)
    elif path.exists() or path.is_symlink():
        raise FileExistsError(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_manifest_atomic(path: Path, manifest: SnapshotManifest) -> None:
    """Write a new private manifest and make its directory entry durable."""
    path = Path(path)
    validate_artifact_root(path, creating=False)
    destination = path / "manifest.json"
    try:
        _write_json_atomic(destination, manifest.model_dump(mode="json"))
    except FileExistsError as error:
        raise SnapshotSecurityError(
            f"manifest already exists: {destination}"
        ) from error


def read_manifest(path: Path) -> SnapshotManifest:
    """Read a snapshot published by atomically installing its manifest."""
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
        raise SnapshotCompatibilityError(
            f"invalid snapshot manifest: {_validation_path(error)}"
        ) from error


def inspect_snapshot(path: Path) -> dict[str, object]:
    return read_manifest(path).model_dump(mode="json")
