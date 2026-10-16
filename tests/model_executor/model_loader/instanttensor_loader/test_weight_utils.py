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


@pytest.mark.parametrize(
    ("weight_name", "local_ids", "collective"),
    [
        ("mlp.experts.gate_proj.weight", {0}, True),
        ("mlp.experts.0.gate_proj.weight", {0}, False),
        ("mlp.experts.0.gate_proj.weight", set(), False),
    ],
)
def test_instanttensor_ep_collective_loading(
    monkeypatch, tmp_path, weight_name, local_ids, collective
):
    """Keep fused checkpoints collective and expert decisions rank-independent."""
    from vllm.model_executor.model_loader import weight_utils

    path = tmp_path / "model.safetensors"
    save_file({weight_name: torch.ones(1)}, str(path))
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
    list(instanttensor_weights_iterator([str(path)], False, local_expert_ids=local_ids))
    kwargs = native_open.call_args.kwargs
    assert kwargs["process_group"] is (group.device_group if collective else None)


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
