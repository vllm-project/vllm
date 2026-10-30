# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_safetensors_files_by_weight_name,
)


@pytest.mark.parametrize("remote", [False, True])
def test_mixed_shard_filter_is_opt_in(tmp_path, remote):
    """The default path loads both tensors; remote mode excludes only its table."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "max_position_embeddings": 64,
            }
        )
    )
    save_file(
        {"table": torch.ones(8), "projection": torch.zeros(4)},
        str(tmp_path / "model.safetensors"),
    )

    class TinyModel(torch.nn.Module):
        @property
        def filter_unused_checkpoint_weights(self):
            return remote

        def is_unused_checkpoint_weight(self, name):
            return name == "table"

    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, max_model_len=32, dtype="float32"
    )
    loader = DefaultModelLoader(
        LoadConfig(
            load_format="safetensors",
            safetensors_load_strategy="lazy",
            use_tqdm_on_load=False,
        )
    )
    names = {name for name, _ in loader.get_all_weights(model_config, TinyModel())}
    assert names == ({"projection"} if remote else {"table", "projection"})


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


if __name__ == "__main__":
    test_filter_duplicate_safetensors_files_missing_weight()
    test_filter_duplicate_safetensors_files_all_exist()


def _skip_non_mtp(name: str) -> bool:
    return not name.startswith("mtp.")


@pytest.fixture
def mtp_checkpoint(tmp_path):
    """Three shards: target only, target + MTP, MTP only."""
    shards = {
        "model-00001-of-00003.safetensors": ["model.layers.0.weight"],
        "model-00002-of-00003.safetensors": ["model.layers.1.weight", "mtp.a.weight"],
        "model-00003-of-00003.safetensors": ["mtp.b.weight"],
    }
    for filename, names in shards.items():
        save_file({name: torch.zeros(1) for name in names}, tmp_path / filename)
    return tmp_path


def test_filter_by_weight_name_drops_shards_without_wanted_weights(mtp_checkpoint):
    files = sorted(str(f) for f in mtp_checkpoint.glob("*.safetensors"))

    assert filter_safetensors_files_by_weight_name(files, _skip_non_mtp) == files[1:]
    # A rule that rejects everything must not leave the loader without files.
    assert filter_safetensors_files_by_weight_name(files, lambda _: True) == files


def test_loader_does_not_read_shards_the_model_skips(mtp_checkpoint, monkeypatch):
    """Whole-file loaders such as InstantTensor only see the kept shards."""
    opened: list[list[str]] = []

    def fake_iterator(hf_weights_files, *args, **kwargs):
        opened.append(sorted(os.path.basename(f) for f in hf_weights_files))
        return iter(())

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.default_loader."
        "instanttensor_weights_iterator",
        fake_iterator,
    )

    class Drafter(torch.nn.Module):
        def is_unused_checkpoint_weight(self, name: str) -> bool:
            return _skip_non_mtp(name)

    loader = DefaultModelLoader(LoadConfig(load_format="instanttensor"))
    model_config = SimpleNamespace(model=str(mtp_checkpoint), revision=None)
    list(loader.get_all_weights(model_config, Drafter()))
    list(loader.get_all_weights(model_config, torch.nn.Module()))

    assert opened == [
        ["model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"],
        sorted(f.name for f in mtp_checkpoint.glob("*.safetensors")),
    ]


def test_secondary_source_with_explicit_patterns_skips_backbone_index(tmp_path):
    """A source with explicit allow patterns (e.g. a standalone head file in
    the repo root) must not be validated against the backbone's index."""
    save_file(
        {"backbone.weight": torch.zeros(1)},
        tmp_path / "model-00001-of-00001.safetensors",
    )
    save_file({"head.weight": torch.zeros(1)}, tmp_path / "joint_head.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"backbone.weight": "model-00001-of-00001.safetensors"}}
        )
    )

    loader = DefaultModelLoader(LoadConfig())
    secondary = DefaultModelLoader.Source(
        model_or_path=str(tmp_path),
        revision=None,
        prefix="pooler.head.",
        allow_patterns_overrides=["joint_head.safetensors"],
    )
    assert [n for n, _ in loader._get_weights_iterator(secondary)] == [
        "pooler.head.head.weight"
    ]
    # The primary source still filters files the backbone index doesn't list.
    primary = DefaultModelLoader.Source(model_or_path=str(tmp_path), revision=None)
    assert [n for n, _ in loader._get_weights_iterator(primary)] == ["backbone.weight"]
