# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.fixture
def mock_instanttensor(monkeypatch):
    from vllm.model_executor.model_loader import weight_utils

    native_open = MagicMock()
    reader = native_open.return_value.__enter__.return_value
    reader.total_tensor_size = 0
    reader.tensors.return_value = ()
    monkeypatch.setitem(
        sys.modules, "instanttensor", SimpleNamespace(safe_open=native_open)
    )
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setattr(weight_utils, "get_world_group", lambda: group)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "current_device", lambda: 0)
    return native_open, group


@pytest.mark.parametrize(
    ("weight_name", "local_ids", "collective"),
    [
        ("mlp.experts.gate_proj.weight", {0}, True),
        ("mlp.experts.0.gate_proj.weight", {0}, False),
        ("mlp.experts.0.gate_proj.weight", set(), False),
    ],
)
def test_instanttensor_ep_collective_loading(
    mock_instanttensor, tmp_path, weight_name, local_ids, collective
):
    """Keep fused checkpoints collective and expert decisions rank-independent."""
    native_open, group = mock_instanttensor
    path = tmp_path / "model.safetensors"
    save_file({weight_name: torch.ones(1)}, str(path))

    list(instanttensor_weights_iterator([str(path)], False, local_expert_ids=local_ids))
    kwargs = native_open.call_args.kwargs
    assert kwargs["process_group"] is (group.device_group if collective else None)


@pytest.mark.parametrize("local_ids", [None, {0}])
@pytest.mark.parametrize("separate_shard", [False, True])
def test_instanttensor_engram_tables_load_on_cpu(
    mock_instanttensor, tmp_path, local_ids, separate_shard
):
    """Exclude tables before GPU buffer sizing, preserving all local weights."""
    native_open, _ = mock_instanttensor
    cpu_weights = {
        "layers.1.engram.embed.weight": torch.arange(16)
        .reshape(4, 4)
        .to(torch.float8_e4m3fn),
        "layers.1.engram.embed.scale": torch.tensor(
            [0, 127, 128, 254], dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    gpu_weights = {
        "layers.1.engram.wkv.weight": torch.ones(2),
        "mlp.experts.0.gate_proj.weight": torch.ones(2),
        "mlp.experts.1.gate_proj.weight": torch.full((2,), 2.0),
        "mlp.experts.1.gate_proj.weight_scale": torch.ones(1),
    }
    path = tmp_path / "model.safetensors"
    save_file(gpu_weights if separate_shard else gpu_weights | cpu_weights, path)
    paths = [str(path)]
    if separate_shard:
        cpu_path = tmp_path / "engram.safetensors"
        save_file(cpu_weights, cpu_path)
        paths.append(str(cpu_path))

    expected_gpu = gpu_weights.copy()
    if local_ids is not None:
        del expected_gpu["mlp.experts.1.gate_proj.weight"]

    reader = native_open.return_value.__enter__.return_value
    reader.tensors.return_value = iter(expected_gpu.items())

    loaded = list(instanttensor_weights_iterator(paths, False, local_ids))
    kwargs = native_open.call_args.kwargs
    assert kwargs["process_group"] is None
    assert kwargs["tensor_name_match_mode"] == "exact"
    assert set(kwargs["tensor_filter"]) == expected_gpu.keys()
    expected = expected_gpu | cpu_weights
    assert len(loaded) == len(expected)
    assert dict(loaded).keys() == expected.keys()
    for name, tensor in loaded:
        assert tensor.dtype == expected[name].dtype
        assert tensor.shape == expected[name].shape
        assert torch.equal(tensor.view(torch.uint8), expected[name].view(torch.uint8))
        if name in cpu_weights:
            assert tensor.device.type == "cpu"


def test_instanttensor_engram_only_checkpoint(mock_instanttensor, tmp_path):
    """An Engram-only source needs no GPU buffer."""
    native_open, _ = mock_instanttensor
    path = tmp_path / "engram.safetensors"
    weights = {"layers.1.engram.embed.weight": torch.arange(4).reshape(2, 2)}
    save_file(weights, path)

    loaded = dict(instanttensor_weights_iterator([str(path)], False))
    native_open.assert_not_called()
    assert loaded.keys() == weights.keys()
    torch.testing.assert_close(
        loaded["layers.1.engram.embed.weight"], weights["layers.1.engram.embed.weight"]
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_ep_filter_cuda(tmp_path):
    weights = {
        "mlp.experts.0.gate_proj.weight": torch.ones(2),
        "mlp.experts.1.gate_proj.weight": torch.ones(2),
        "mlp.experts.1.gate_proj.weight_scale": torch.ones(1),
        "model.layers[0].attn.weight": torch.ones(2),
        "layers.1.engram.embed.weight": torch.ones(4, 2),
        "layers.1.engram.embed.scale": torch.ones(4, 1),
    }
    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))
    loaded = dict(
        instanttensor_weights_iterator([str(path)], False, local_expert_ids={0})
    )
    del weights["mlp.experts.1.gate_proj.weight"]
    assert loaded.keys() == weights.keys()
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor.cpu(), weights[name], rtol=0, atol=0)
        assert tensor.device.type == ("cpu" if ".engram.embed." in name else "cuda")


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


if __name__ == "__main__":
    test_instanttensor_model_loader()
