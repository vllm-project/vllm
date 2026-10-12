# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("local_ids", [None, {0, 2}, set()])
def test_instanttensor_ep_filter(monkeypatch, local_ids):
    """Select experts before native I/O and avoid rank-dependent NCCL layouts."""
    from vllm.model_executor.model_loader import weight_utils

    weights = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.ones(2),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.ones(2),
        "model.layers.0.mlp.experts.1.gate_proj.weight_packed": torch.ones(2),
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale": torch.ones(1),
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.ones(2),
        "model.layers.0.mlp.experts.gate_proj.weight": torch.ones(2),
    }
    native_open = MagicMock()
    reader = native_open.return_value.__enter__.return_value

    def open_reader(*args, **kwargs):
        predicate = kwargs.get("tensor_filter", lambda name: True)
        selected = {name: tensor for name, tensor in weights.items() if predicate(name)}
        reader.total_tensor_size = sum(
            tensor.numel() * tensor.element_size() for tensor in selected.values()
        )
        reader.tensors.return_value = iter(selected.items())
        return native_open.return_value

    native_open.side_effect = open_reader
    monkeypatch.setitem(
        sys.modules, "instanttensor", SimpleNamespace(safe_open=native_open)
    )
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setattr(weight_utils, "get_world_group", lambda: group)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "current_device", lambda: 0)

    loaded = dict(instanttensor_weights_iterator([], False, local_expert_ids=local_ids))
    expected = set(weights)
    if local_ids is not None:
        expected -= {
            name
            for name in weights
            if ".experts.1." in name and name.endswith((".weight", ".weight_packed"))
        }
        if not local_ids:
            expected.remove("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert set(loaded) == expected
    kwargs = native_open.call_args.kwargs
    assert kwargs["copy"] is True
    if local_ids is None:
        assert kwargs["process_group"] is group.device_group
        assert "tensor_filter" not in kwargs
    else:
        assert kwargs["process_group"] is None


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
