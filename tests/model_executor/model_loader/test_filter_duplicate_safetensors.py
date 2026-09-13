# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
)


def test_filter_duplicate_safetensors_files_missing_weight():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        with open(existing_file, "wb") as f:
            f.write(b"")

        existing_file2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        with open(existing_file2, "wb") as f:
            f.write(b"")

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
                "layer.2.weight": "model-00003-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        hf_weights_files = [
            os.path.join(tmpdir, "model-00001-of-00002.safetensors"),
            os.path.join(tmpdir, "model-00002-of-00002.safetensors"),
        ]

        with pytest.raises(FileNotFoundError) as exc_info:
            filter_duplicate_safetensors_files(
                hf_weights_files=hf_weights_files,
                hf_folder=tmpdir,
                index_file="model.safetensors.index.json",
            )

        assert "model-00003-of-00002.safetensors" in str(exc_info.value)


def test_filter_duplicate_safetensors_files_all_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_files = []
        for i in range(1, 3):
            file_path = os.path.join(tmpdir, f"model-0000{i}-of-00002.safetensors")
            with open(file_path, "wb") as f:
                f.write(b"")
            existing_files.append(file_path)

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        filter_duplicate_safetensors_files(
            hf_weights_files=existing_files,
            hf_folder=tmpdir,
            index_file="model.safetensors.index.json",
        )


def _write_index(tmp_path, weight_map: dict[str, str]) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _write_reused_shard_checkpoint(tmp_path) -> None:
    """A checkpoint whose second shard is reused from another model.

    The index assigns ``main.weight`` to the first shard and ``ple.weight`` to
    the second, but the second shard also stores a stale ``main.weight`` and an
    entirely unindexed tensor.
    """
    main_shard = tmp_path / "model-00001-of-00002.safetensors"
    reused_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"main.weight": torch.tensor([1.0])}, main_shard)
    save_file(
        {
            "ple.weight": torch.tensor([2.0]),
            "main.weight": torch.tensor([99.0]),
            "unindexed.weight": torch.tensor([3.0]),
        },
        reused_shard,
    )
    _write_index(
        tmp_path,
        {"main.weight": main_shard.name, "ple.weight": reused_shard.name},
    )


def _write_ordinary_checkpoint(tmp_path) -> None:
    """A normal sharded checkpoint: each shard stores exactly its index entry."""
    first_shard = tmp_path / "model-00001-of-00002.safetensors"
    second_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"main.weight": torch.tensor([1.0])}, first_shard)
    save_file({"ple.weight": torch.tensor([2.0])}, second_shard)
    _write_index(
        tmp_path,
        {"main.weight": first_shard.name, "ple.weight": second_shard.name},
    )


def _prepare(tmp_path, **load_config_kwargs):
    loader = DefaultModelLoader(LoadConfig(**load_config_kwargs))
    return loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=False,
        allow_patterns_overrides=None,
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
@pytest.mark.parametrize("load_strategy", [None, "lazy", "eager"])
def test_default_loader_only_yields_tensors_assigned_by_index(
    tmp_path, load_format: str, load_strategy: str | None
) -> None:
    _write_reused_shard_checkpoint(tmp_path)

    loader = DefaultModelLoader(
        LoadConfig(
            load_format=load_format,
            safetensors_load_strategy=load_strategy,
            use_tqdm_on_load=False,
        )
    )
    source = DefaultModelLoader.Source(
        model_or_path=str(tmp_path),
        revision=None,
        fall_back_to_pt=False,
    )

    weights = list(loader._get_weights_iterator(source))

    assert [name for name, _ in weights] == ["main.weight", "ple.weight"]
    torch.testing.assert_close(weights[0][1], torch.tensor([1.0]))
    torch.testing.assert_close(weights[1][1], torch.tensor([2.0]))


@pytest.mark.skip_global_cleanup
def test_multithread_loader_only_yields_tensors_assigned_by_index(tmp_path) -> None:
    _write_reused_shard_checkpoint(tmp_path)

    loader = DefaultModelLoader(
        LoadConfig(
            load_format="safetensors",
            use_tqdm_on_load=False,
            model_loader_extra_config={"enable_multithread_load": True},
        )
    )
    source = DefaultModelLoader.Source(
        model_or_path=str(tmp_path),
        revision=None,
        fall_back_to_pt=False,
    )

    weights = sorted(loader._get_weights_iterator(source), key=lambda kv: kv[0])

    assert [name for name, _ in weights] == ["main.weight", "ple.weight"]
    torch.testing.assert_close(weights[0][1], torch.tensor([1.0]))
    torch.testing.assert_close(weights[1][1], torch.tensor([2.0]))


@pytest.mark.skip_global_cleanup
def test_prepare_weights_skips_filtering_for_ordinary_checkpoint(tmp_path) -> None:
    # Every shard stores exactly what the index assigns to it, so no allowlist
    # is produced and the accelerated backends stay available.
    _write_ordinary_checkpoint(tmp_path)

    *_, indexed_weights_by_file = _prepare(tmp_path, load_format="safetensors")

    assert indexed_weights_by_file is None


@pytest.mark.skip_global_cleanup
def test_prepare_weights_engages_filtering_for_reused_shard(tmp_path) -> None:
    # The reused shard stores tensors the index did not assign to it, so the
    # per-file allowlist is produced.
    _write_reused_shard_checkpoint(tmp_path)

    *_, indexed_weights_by_file = _prepare(tmp_path, load_format="safetensors")

    assert indexed_weights_by_file == {
        os.path.normpath(str(tmp_path / "model-00001-of-00002.safetensors")): {
            "main.weight"
        },
        os.path.normpath(str(tmp_path / "model-00002-of-00002.safetensors")): {
            "ple.weight"
        },
    }


@pytest.mark.skip_global_cleanup
def test_prepare_weights_skips_filtering_without_index(tmp_path) -> None:
    save_file({"main.weight": torch.tensor([1.0])}, tmp_path / "model.safetensors")

    *_, indexed_weights_by_file = _prepare(tmp_path, load_format="safetensors")

    assert indexed_weights_by_file is None


if __name__ == "__main__":
    test_filter_duplicate_safetensors_files_missing_weight()
    test_filter_duplicate_safetensors_files_all_exist()
