# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import DefaultModelLoader, weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_safetensors_files_by_weight_name,
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
    """InstantTensor only sees shards with at least one wanted tensor."""
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


@pytest.mark.parametrize("selection", [None, "mtp", "all", "empty", "remote"])
@pytest.mark.parametrize("prefix", ["", "draft."])
@pytest.mark.parametrize("local_ids", [None, {0}])
def test_instanttensor_filters_before_native_io(
    tmp_path, monkeypatch, selection, prefix, local_ids
):
    """Intersect MTP and EP selections before native I/O and source prefixes."""
    weights = {
        name: torch.ones(4)
        for name in (
            "model.layers.0.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "mtp.layers[0].weight",
            "mtp.layers0.weight",
            "mtp.mlp.experts.0.gate_proj.weight",
            "mtp.mlp.experts.1.gate_proj.weight",
            "mtp.mlp.experts.1.gate_proj.weight_scale",
            "mtp.mlp.experts.1.gate_proj.input_scale",
            "mtp.mlp.shared_experts.gate_proj.weight",
            "mtp.mlp.experts.gate_proj.weight",
        )
    }
    save_file(weights, tmp_path / "model.safetensors")
    wanted = {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "mtp.layers[0].weight",
        "mtp.mlp.experts.0.gate_proj.weight",
        "mtp.mlp.experts.1.gate_proj.weight",
        "mtp.mlp.experts.1.gate_proj.weight_scale",
        "mtp.mlp.shared_experts.gate_proj.weight",
    }
    selected = set(weights) if selection in (None, "all") else wanted
    if selection == "empty":
        selected = set()
    elif selection == "remote":
        selected = {"mtp.mlp.experts.1.gate_proj.weight"}
    expected = selected - (
        {"mtp.mlp.experts.1.gate_proj.weight"} if local_ids is not None else set()
    )
    process_group = object()
    opened = []
    progress = MagicMock()
    monkeypatch.setattr(weight_utils, "tqdm", progress)

    @contextmanager
    def fake_open(files, *, tensor_filter, process_group, **kwargs):
        opened.append((tensor_filter, process_group, kwargs))
        names = weights if tensor_filter is None else tensor_filter
        tensors = {name: weights[name] for name in names}
        yield SimpleNamespace(
            total_tensor_size=sum(
                t.numel() * t.element_size() for t in tensors.values()
            ),
            tensors=lambda: iter(tensors.items()),
        )

    monkeypatch.setitem(
        sys.modules, "instanttensor", SimpleNamespace(safe_open=fake_open)
    )
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(
        weight_utils,
        "get_world_group",
        lambda: SimpleNamespace(world_size=8, device_group=process_group),
    )

    class Drafter(torch.nn.Module):
        def is_unused_checkpoint_weight(self, name):
            return name not in selected

    model = torch.nn.Module() if selection is None else Drafter()
    loader = DefaultModelLoader(LoadConfig(load_format="instanttensor"))
    loader.local_expert_ids = local_ids
    model_config = SimpleNamespace(model=str(tmp_path), revision=None)
    if prefix:
        source = DefaultModelLoader.Source(
            str(tmp_path),
            revision=None,
            prefix=prefix,
            is_unused_weight=getattr(model, "is_unused_checkpoint_weight", None),
        )
        result = dict(loader._get_weights_iterator(source))
    else:
        result = dict(loader.get_all_weights(model_config, model))
    assert result.keys() == {prefix + name for name in expected}
    for name, tensor in result.items():
        torch.testing.assert_close(tensor, weights[name.removeprefix(prefix)])
    if not expected:
        assert opened == []
        progress.assert_not_called()
    else:
        assert len(opened) == 1
        tensor_filter, group, kwargs = opened[0]
        assert kwargs["copy"] is True
        assert kwargs["tensor_name_match_mode"] == "exact"
        assert progress.call_args.kwargs["total"] == sum(
            weights[name].numel() * weights[name].element_size() for name in expected
        )
        if selection is None and local_ids is None:
            assert tensor_filter is None
            assert group is process_group
        else:
            assert set(tensor_filter) == expected
            assert group is None


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
