# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch

import vllm.model_executor.model_loader.weight_utils as weight_utils
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


def test_default_loader_filters_fastsafetensors_before_materializing(monkeypatch):
    class FakeProcessGroup:
        def size(self):
            return 1

    class FakeFileBuffer:
        def __init__(self):
            self.key_to_rank_lidx = {
                "model.layers.0.self_attn.q_proj.weight": (0, 0),
                "model.layers.0.mlp.experts.0.gate_proj.weight": (0, 1),
                "model.layers.0.mlp.experts.1.gate_proj.weight": (0, 2),
                "model.mtp.0.weight": (0, 3),
            }
            self.loaded_keys: list[str] = []
            self.closed = False

        def get_tensor(self, key: str):
            self.loaded_keys.append(key)
            return torch.tensor([len(self.loaded_keys)])

        def close(self):
            self.closed = True

    class FakeLoader:
        def __init__(self, file_buffer):
            self.file_buffer = file_buffer
            self.closed = False

        def copy_files_to_device(self):
            return self.file_buffer

        def close(self):
            self.closed = True

    file_buffer = FakeFileBuffer()
    loader = FakeLoader(file_buffer)

    model_loader = DefaultModelLoader(LoadConfig(load_format="fastsafetensors"))
    model_loader.local_expert_ids = {0}
    monkeypatch.setattr(
        model_loader,
        "_prepare_weights",
        lambda *_args: ("/weights", ["model.safetensors"], True),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(weight_utils, "SingleGroup", FakeProcessGroup)
    monkeypatch.setattr(
        weight_utils,
        "_init_fastsafetensors_loader",
        lambda *_args, **_kwargs: loader,
    )

    loaded = dict(
        model_loader._get_weights_iterator(
            DefaultModelLoader.Source("model", revision=None),
            weight_name_filter=lambda name: "model.mtp." in name,
        )
    )

    assert set(loaded) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
    }
    assert file_buffer.loaded_keys == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
    ]
    assert file_buffer.closed
    assert loader.closed


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_model_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
        )
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0

        fastsafetensors_tensors = {}
        hf_safetensors_tensors = {}

        for name, tensor in fastsafetensors_weights_iterator(safetensors, True):
            fastsafetensors_tensors[name] = tensor

        for name, tensor in safetensors_weights_iterator(safetensors, True):
            hf_safetensors_tensors[name] = tensor

        assert len(fastsafetensors_tensors) == len(hf_safetensors_tensors)

        for name, fastsafetensors_tensor in fastsafetensors_tensors.items():
            fastsafetensors_tensor = fastsafetensors_tensor.to("cpu")
            assert fastsafetensors_tensor.dtype == hf_safetensors_tensors[name].dtype
            assert fastsafetensors_tensor.shape == hf_safetensors_tensors[name].shape
            assert torch.all(fastsafetensors_tensor.eq(hf_safetensors_tensors[name]))


if __name__ == "__main__":
    test_fastsafetensors_model_loader()
