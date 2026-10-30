# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared compiled-cache manifest and installation helpers."""

import hashlib
import io
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

BUNDLE_LIMIT = 8 << 30


@dataclass(frozen=True)
class CacheRoot:
    name: str
    path: Path


@dataclass(frozen=True)
class CacheManifest:
    entries: tuple[dict[str, Any], ...]
    total_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "total_size": self.total_size,
            "entries": list(self.entries),
        }


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe cache path: {value!r}")
    return path


def _cache_roots() -> tuple[CacheRoot, ...]:
    roots = []
    for name, variable in (
        ("flashinfer-cubins", "FLASHINFER_CUBIN_DIR"),
        ("flashinfer-workspace", "FLASHINFER_WORKSPACE_BASE"),
    ):
        value = os.environ.get(variable)
        if value:
            roots.append(CacheRoot(name, Path(value).expanduser()))
    return tuple(roots)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(roots: tuple[CacheRoot, ...]) -> CacheManifest:
    entries: list[dict[str, Any]] = []
    total = 0
    nested_roots = {
        root.path: tuple(
            other.path
            for other in roots
            if other.path != root.path and root.path in other.path.parents
        )
        for root in roots
    }
    for root in roots:
        if not root.path.is_dir():
            continue
        for path in sorted(root.path.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.name.endswith((".lock", ".tmp"))
                or any(other in path.parents for other in nested_roots[root.path])
            ):
                continue
            relative = path.relative_to(root.path).as_posix()
            _safe_path(relative)
            stat = path.stat()
            total += stat.st_size
            if total > BUNDLE_LIMIT:
                raise ValueError("compiled cache is too large")
            entries.append(
                {
                    "root": root.name,
                    "path": relative,
                    "size": stat.st_size,
                    "sha256": _digest(path),
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return CacheManifest(tuple(entries), total)


def _bundle(roots: tuple[CacheRoot, ...], manifest: CacheManifest) -> bytes:
    by_name = {root.name: root for root in roots}
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:") as archive:
        for entry in manifest.entries:
            root = by_name[entry["root"]]
            path = root.path / _safe_path(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"compiled cache changed: {path}")
            info = tarfile.TarInfo(f"{entry['root']}/{entry['path']}")
            info.size, info.mode = entry["size"], 0o600
            info.mtime = entry["mtime_ns"] / 1e9
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    return output.getvalue()


def _install(
    bundle: bytes, manifest: CacheManifest, roots: tuple[CacheRoot, ...]
) -> None:
    by_name = {root.name: root for root in roots}
    expected = {f"{entry['root']}/{entry['path']}": entry for entry in manifest.entries}
    staged: list[tuple[Path, Path, dict[str, Any]]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != len(expected):
                raise ValueError("compiled cache bundle entry count mismatch")
            for member in members:
                entry = expected.get(member.name)
                if entry is None or not member.isfile():
                    raise ValueError(f"unexpected compiled cache member: {member.name}")
                root_name, relative = member.name.split("/", 1)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("cannot read compiled cache member")
                data = extracted.read()
                if (
                    len(data) != entry["size"]
                    or hashlib.sha256(data).hexdigest() != entry["sha256"]
                ):
                    raise ValueError(f"compiled cache checksum mismatch: {member.name}")
                destination = by_name[root_name].path / _safe_path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                temporary.write_bytes(data)
                staged.append((temporary, destination, entry))
        for temporary, destination, entry in staged:
            os.replace(temporary, destination)
            os.utime(destination, ns=(entry["mtime_ns"], entry["mtime_ns"]))
    except Exception:
        for temporary, _, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
