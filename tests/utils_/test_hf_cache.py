# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest
from huggingface_hub._tree_cache import TreeCacheEntry, write_tree_cache
from huggingface_hub.utils.sha import git_hash

import tests.hf_cache_utils as cache_utils
from tests.hf_cache_utils import HfCacheIntegrityError, HfCacheVerifier

_COMMIT = "a" * 40


def _make_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "models--org--model"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / _COMMIT
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    return repo, blobs, snapshot


def _add_file(
    snapshot: Path,
    blobs: Path,
    name: str,
    data: bytes,
    *,
    lfs: bool,
) -> tuple[Path, Path, TreeCacheEntry]:
    digest = hashlib.sha256(data).hexdigest() if lfs else git_hash(data)
    blob = blobs / digest
    blob.write_bytes(data)
    entry = snapshot / name
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.symlink_to(os.path.relpath(blob, entry.parent))
    metadata = TreeCacheEntry(
        size=len(data),
        blob_id=git_hash(data),
        lfs_sha256=digest if lfs else None,
        lfs_size=len(data) if lfs else None,
    )
    return entry, blob, metadata


def _write_tree(repo: Path, files: dict[str, TreeCacheEntry]) -> None:
    write_tree_cache(str(repo), _COMMIT, files)


def test_fixture_verifies_partial_git_and_lfs_snapshot(
    tmp_path: Path, verify_hf_cache_artifacts
) -> None:
    repo, blobs, snapshot = _make_snapshot(tmp_path)
    _, _, config = _add_file(snapshot, blobs, "config.json", b"config", lfs=False)
    _, _, weights = _add_file(
        snapshot, blobs, "weights/model.safetensors", b"weights", lfs=True
    )
    _write_tree(
        repo,
        {
            "config.json": config,
            "weights/model.safetensors": weights,
            "not-materialized.json": TreeCacheEntry(1, "f" * 40),
        },
    )
    assert verify_hf_cache_artifacts(snapshot) == snapshot


@pytest.mark.parametrize("lfs", [False, True])
def test_corrupt_blob_fails_checksum(tmp_path: Path, lfs: bool) -> None:
    repo, blobs, snapshot = _make_snapshot(tmp_path)
    entry, blob, metadata = _add_file(snapshot, blobs, "artifact.bin", b"good", lfs=lfs)
    _write_tree(repo, {"artifact.bin": metadata})
    blob.write_bytes(b"evil")
    with pytest.raises(HfCacheIntegrityError, match="checksum mismatch"):
        HfCacheVerifier().verify(entry)


@pytest.mark.parametrize("failure", ["invalid-tree", "missing-entry", "wrong-size"])
def test_invalid_tree_metadata_fails_closed(tmp_path: Path, failure: str) -> None:
    repo, blobs, snapshot = _make_snapshot(tmp_path)
    entry, _, metadata = _add_file(snapshot, blobs, "artifact.bin", b"data", lfs=False)
    if failure == "missing-entry":
        _write_tree(repo, {})
    elif failure == "wrong-size":
        _write_tree(repo, {"artifact.bin": replace(metadata, size=99)})
    else:
        trees = repo / "trees"
        trees.mkdir()
        (trees / f"{_COMMIT}.json").write_text('{"format_version": 2, "files": {}}')

    with pytest.raises(HfCacheIntegrityError):
        HfCacheVerifier().verify(entry)


def test_tree_less_content_address_fallback(tmp_path: Path) -> None:
    _, blobs, snapshot = _make_snapshot(tmp_path)
    entry, _, _ = _add_file(snapshot, blobs, "config.json", b"config", lfs=False)
    assert HfCacheVerifier().verify(entry) == entry

    regular_file = snapshot / "copied.json"
    regular_file.write_bytes(b"copy")
    with pytest.raises(HfCacheIntegrityError, match="No tree metadata"):
        HfCacheVerifier().verify(regular_file)


def test_stable_identity_is_memoized_and_replacement_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, blobs, snapshot = _make_snapshot(tmp_path)
    entry, blob, metadata = _add_file(
        snapshot, blobs, "model.safetensors", b"weights", lfs=True
    )
    _write_tree(repo, {"model.safetensors": metadata})
    original_hash = cache_utils.compute_file_hash
    calls = 0

    def counted_hash(path: Path, algorithm: str) -> str:
        nonlocal calls
        calls += 1
        return original_hash(path, algorithm)  # type: ignore[arg-type]

    monkeypatch.setattr(cache_utils, "compute_file_hash", counted_hash)
    verifier = HfCacheVerifier()
    verifier.verify(entry)
    verifier.verify(entry)
    assert calls == 1

    replacement = blob.with_suffix(".replacement")
    replacement.write_bytes(b"corrupt")
    os.replace(replacement, blob)
    with pytest.raises(HfCacheIntegrityError, match="checksum mismatch"):
        verifier.verify(entry)
    assert calls == 2
