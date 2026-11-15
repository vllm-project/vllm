# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The checkpoint digest identifies the loaded weights by content, so engines
can tell whether they compute the same KV cache."""

import hashlib
import os
from pathlib import Path

import pytest

from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.model_loader.checkpoint_digest import (
    CheckpointRecording,
    record_checkpoint_files,
)

MODEL = "facebook/opt-125m"
SHARDS = {
    "model-00001-of-00002.safetensors": b"first shard",
    "model-00002-of-00002.safetensors": b"second shard",
}


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig(MODEL)


def _hf_cache_snapshot(
    root: Path, shards: dict[str, bytes], shared_store: bool = False
) -> Path:
    """Lay out shards as `huggingface_hub` caches them: snapshot entries link
    to blobs named by their SHA-256. With `shared_store`, each blob links on
    into a store that names files by another hash."""
    blobs = root / "blobs"
    store = root / "store"
    snapshot = root / "snapshots" / "0123abcd"
    for folder in (blobs, store, snapshot):
        folder.mkdir(parents=True)
    for name, data in shards.items():
        blob = blobs / hashlib.sha256(data).hexdigest()
        if shared_store:
            stored = store / hashlib.sha3_256(data).hexdigest()
            stored.write_bytes(data)
            blob.symlink_to(os.path.relpath(stored, blobs))
        else:
            blob.write_bytes(data)
        (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))
    return snapshot


def _local_copy(folder: Path, shards: dict[str, bytes]) -> Path:
    folder.mkdir(parents=True)
    for name, data in shards.items():
        (folder / name).write_bytes(data)
    return folder


def _record(owner: str, folder: Path) -> None:
    files = sorted(str(path) for path in folder.iterdir())
    record_checkpoint_files(owner, str(folder), files)


def _digest(
    model_config: ModelConfig,
    folder: Path,
    hash_local_files: bool = False,
    load_format: str = "auto",
    draft: Path | None = None,
) -> dict[str, str]:
    with CheckpointRecording(model_config.model, hash_local_files) as recording:
        _record(model_config.model, folder)
        if draft is not None:
            _record("org/draft", draft)
    return recording.digest(model_config, LoadConfig(load_format=load_format))


@pytest.mark.parametrize("shared_store", [False, True])
def test_hf_cache_and_hashed_local_copy_match(model_config, tmp_path, shared_store):
    snapshot = _hf_cache_snapshot(tmp_path / "hub", SHARDS, shared_store)
    cached = _digest(model_config, snapshot)
    local = _digest(
        model_config,
        _local_copy(tmp_path / "local", SHARDS),
        hash_local_files=True,
    )

    assert cached["weights_source"] == "hf_cache"
    assert local["weights_source"] == "hashed"
    assert cached["weights"] == local["weights"]


def test_changed_shard_changes_digest(model_config, tmp_path):
    changed = {**SHARDS, "model-00002-of-00002.safetensors": b"fine-tuned"}

    original = _digest(model_config, _hf_cache_snapshot(tmp_path / "a", SHARDS))
    updated = _digest(model_config, _hf_cache_snapshot(tmp_path / "b", changed))

    assert original["weights"] != updated["weights"]


def test_unhashed_local_copy_is_identified_by_path(model_config, tmp_path):
    """Without hashing, files outside the HF cache cannot be told apart, so
    the digest says so instead of claiming to cover their content."""
    original = _digest(model_config, _local_copy(tmp_path / "a", SHARDS))
    changed = {**SHARDS, "model-00002-of-00002.safetensors": b"fine-tuned"}
    updated = _digest(model_config, _local_copy(tmp_path / "b", changed))

    assert original["weights_source"] == "path"
    assert original == updated


def test_other_models_files_are_excluded(model_config, tmp_path):
    """A draft model loaded in the same worker is not part of the target's
    digest."""
    snapshot = _hf_cache_snapshot(tmp_path / "target", SHARDS)
    alone = _digest(model_config, snapshot)

    draft = _hf_cache_snapshot(tmp_path / "draft", {"model.safetensors": b"draft"})
    with_draft = _digest(model_config, snapshot, draft=draft)

    assert alone == with_draft


def test_dummy_weights_are_marked(model_config, tmp_path):
    digest = _digest(
        model_config, _hf_cache_snapshot(tmp_path, SHARDS), load_format="dummy"
    )

    assert digest == {"weights": "dummy", "weights_source": "dummy"}


def test_files_loaded_outside_a_recording_are_not_recorded(model_config, tmp_path):
    """Only the rank that records digests the checkpoint; loads elsewhere
    leave nothing behind for a later digest."""
    _record(model_config.model, _local_copy(tmp_path / "other", SHARDS))

    digest = _digest(model_config, _hf_cache_snapshot(tmp_path / "hub", SHARDS))

    assert digest["weights_source"] == "hf_cache"
