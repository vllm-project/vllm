# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared compiled-cache manifest and installation helpers."""

import hashlib
import io
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

BUNDLE_LIMIT = 8 << 30


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe cache path: {value!r}")
    return path


def _cache_roots() -> dict[str, Path]:
    return {
        name: Path(value).expanduser()
        for name, variable in (
            ("flashinfer-cubins", "FLASHINFER_CUBIN_DIR"),
            ("flashinfer-workspace", "FLASHINFER_WORKSPACE_BASE"),
        )
        if (value := os.environ.get(variable))
    }


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(roots: dict[str, Path]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total = 0
    for root_name, root in roots.items():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.name.endswith((".lock", ".tmp"))
                or any(
                    other != root and other in path.parents
                    for other in roots.values()
                    if root in other.parents
                )
            ):
                continue
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            total += stat.st_size
            if total > BUNDLE_LIMIT:
                raise ValueError("compiled cache is too large")
            entries.append(
                {
                    "root": root_name,
                    "path": relative,
                    "size": stat.st_size,
                    "sha256": _digest(path),
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"version": 1, "total_size": total, "entries": entries}


def _bundle(roots: dict[str, Path], manifest: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:") as archive:
        for entry in manifest["entries"]:
            path = roots[entry["root"]] / _safe_path(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"compiled cache changed: {path}")
            info = tarfile.TarInfo(f"{entry['root']}/{entry['path']}")
            info.size, info.mode = entry["size"], 0o600
            info.mtime = entry["mtime_ns"] / 1e9
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    return output.getvalue()


def _install(bundle: bytes, manifest: dict[str, Any], roots: dict[str, Path]) -> None:
    expected = {
        f"{entry['root']}/{entry['path']}": entry for entry in manifest["entries"]
    }
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
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("cannot read compiled cache member")
                data = extracted.read()
                if (
                    len(data) != entry["size"]
                    or hashlib.sha256(data).hexdigest() != entry["sha256"]
                ):
                    raise ValueError(f"compiled cache checksum mismatch: {member.name}")
                destination = roots[entry["root"]] / _safe_path(entry["path"])
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
