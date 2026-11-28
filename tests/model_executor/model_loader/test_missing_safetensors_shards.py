# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import pytest

from vllm.model_executor.model_loader.weight_utils import (
    raise_on_missing_safetensors_shards,
)


def _make_shard_files(tmpdir: str, indices: list[int], total: int) -> list[str]:
    files = []
    for i in indices:
        path = os.path.join(tmpdir, f"model-{i:05d}-of-{total:05d}.safetensors")
        with open(path, "wb"):
            pass
        files.append(path)
    return files


def test_missing_shard_raises_with_missing_name():
    # Real-world shape of an interrupted download: shard 3 of 3 never
    # arrived and no model.safetensors.index.json exists, so the loader
    # would otherwise silently load 2 of 3 shards.
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _make_shard_files(tmpdir, [1, 2], total=3)
        with pytest.raises(FileNotFoundError) as exc_info:
            raise_on_missing_safetensors_shards(files)
        assert "model-00003-of-00003.safetensors" in str(exc_info.value)
        assert "2 of 3" in str(exc_info.value)


def test_complete_shards_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _make_shard_files(tmpdir, [1, 2, 3], total=3)
        raise_on_missing_safetensors_shards(files)


def test_single_unnumbered_file_is_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.safetensors")
        with open(path, "wb"):
            pass
        raise_on_missing_safetensors_shards([path])


def test_mixed_numbered_and_unnumbered_files_are_ignored():
    # e.g. sharded checkpoint plus a consolidated export in one folder:
    # too ambiguous to judge completeness from names alone.
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _make_shard_files(tmpdir, [1, 2], total=3)
        consolidated = os.path.join(tmpdir, "consolidated.safetensors")
        with open(consolidated, "wb"):
            pass
        raise_on_missing_safetensors_shards(files + [consolidated])


def test_mixed_shard_families_are_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _make_shard_files(tmpdir, [1, 2], total=3)
        other_dir = os.path.join(tmpdir, "other")
        os.makedirs(other_dir)
        other = _make_shard_files(other_dir, [1], total=2)
        raise_on_missing_safetensors_shards(files + other)


def test_different_prefixes_without_index_overlap_are_ignored():
    # Two shard families with distinct prefixes whose indices happen not to
    # collide (e.g. a model and an adapter each sharded 1-of-2, one shard of
    # each missing). Index/total alone would look "complete"; the prefix
    # mismatch must force the ambiguous no-op path.
    with tempfile.TemporaryDirectory() as tmpdir:
        model_shard = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        adapter_shard = os.path.join(tmpdir, "adapter-00002-of-00002.safetensors")
        for path in (model_shard, adapter_shard):
            with open(path, "wb"):
                pass
        raise_on_missing_safetensors_shards([model_shard, adapter_shard])


def test_out_of_range_shard_index_is_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _make_shard_files(tmpdir, [0, 1], total=2)
        raise_on_missing_safetensors_shards(files)


def test_empty_list_is_ignored():
    raise_on_missing_safetensors_shards([])
