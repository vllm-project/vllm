# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integrity checks for materialized Hugging Face Hub cache artifacts."""

import os
from pathlib import Path

import regex as re
from huggingface_hub._tree_cache import TreeCacheEntry, read_tree_cache
from huggingface_hub.utils._verification import compute_file_hash

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
FileIdentity = tuple[int, int, int, int, int]
VerificationKey = tuple[str, str, FileIdentity]


class HfCacheIntegrityError(RuntimeError):
    pass


class HfCacheVerifier:
    """Verify a Hub snapshot or one materialized file beneath it."""

    def __init__(self) -> None:
        self._verified: set[VerificationKey] = set()

    def verify(self, artifact: str | os.PathLike[str]) -> Path:
        path = Path(os.path.abspath(os.fspath(artifact)))
        snapshot = next(
            (item for item in (path, *path.parents) if item.parent.name == "snapshots"),
            None,
        )
        if snapshot is None or not _COMMIT_RE.fullmatch(snapshot.name):
            raise HfCacheIntegrityError(f"Not a snapshots/<commit> path: {path}")
        if not snapshot.is_dir():
            raise HfCacheIntegrityError(f"Cached snapshot does not exist: {snapshot}")

        entries = [path]
        if path == snapshot:
            entries = sorted(
                (
                    entry
                    for entry in snapshot.rglob("*")
                    if entry.is_symlink() or entry.is_file()
                ),
                key=os.fspath,
            )
        if not entries:
            raise HfCacheIntegrityError(f"Cached snapshot is empty: {snapshot}")

        tree = self._read_tree(snapshot)
        for entry in entries:
            self._verify_entry(entry, snapshot, tree)
        return path

    @staticmethod
    def _read_tree(snapshot: Path) -> dict[str, TreeCacheEntry] | None:
        repo = snapshot.parent.parent
        tree_path = repo / "trees" / f"{snapshot.name}.json"
        tree = read_tree_cache(str(repo), snapshot.name)
        if tree_path.exists() and tree is None:
            raise HfCacheIntegrityError(f"Invalid Hub tree metadata: {tree_path}")
        return tree

    @staticmethod
    def _expected(
        entry: Path,
        snapshot: Path,
        tree: dict[str, TreeCacheEntry] | None,
    ) -> tuple[str, str, int | None]:
        if tree is None:
            if not entry.is_symlink():
                raise HfCacheIntegrityError(f"No tree metadata: {entry}")
            try:
                target = entry.resolve(strict=True)
                blobs = (snapshot.parent.parent / "blobs").resolve(strict=True)
            except OSError as error:
                raise HfCacheIntegrityError(f"Unreadable link: {entry}") from error
            if target.parent != blobs:
                raise HfCacheIntegrityError(f"Cache link escapes blobs: {entry}")
            expected = target.name
            algorithm = "sha256" if len(expected) == 64 else "git-sha1"
            expected_size = None
        else:
            metadata = tree.get(entry.relative_to(snapshot).as_posix())
            if metadata is None:
                raise HfCacheIntegrityError(f"Tree metadata has no entry for {entry}")
            if metadata.lfs_sha256 is not None:
                algorithm, expected = "sha256", metadata.lfs_sha256
                expected_size = metadata.lfs_size
            else:
                algorithm, expected = "git-sha1", metadata.blob_id
                expected_size = metadata.size
            if expected_size is None or expected_size < 0:
                raise HfCacheIntegrityError(f"Invalid cached size for {entry}")

        digest_length = 64 if algorithm == "sha256" else 40
        if not re.fullmatch(rf"[0-9a-f]{{{digest_length}}}", expected):
            raise HfCacheIntegrityError(f"Invalid cached digest for {entry}")
        return algorithm, expected, expected_size

    def _verify_entry(
        self,
        entry: Path,
        snapshot: Path,
        tree: dict[str, TreeCacheEntry] | None,
    ) -> None:
        algorithm, expected, expected_size = self._expected(entry, snapshot, tree)
        identity = self._identity(entry.stat())
        if expected_size is not None and identity[2] != expected_size:
            raise HfCacheIntegrityError(f"Wrong cached size for {entry}")

        key = (algorithm, expected, identity)
        if key not in self._verified:
            actual = compute_file_hash(entry, algorithm)  # type: ignore[arg-type]
            if self._identity(entry.stat()) != identity:
                raise HfCacheIntegrityError(f"Cache changed while hashing: {entry}")
            if actual != expected:
                raise HfCacheIntegrityError(f"Cache checksum mismatch for {entry}")
            self._verified.add(key)

    @staticmethod
    def _identity(file_stat: os.stat_result) -> FileIdentity:
        return (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )
