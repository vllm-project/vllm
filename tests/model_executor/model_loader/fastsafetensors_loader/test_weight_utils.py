# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
@pytest.mark.parametrize("queue_size", [0, 1])
def test_fastsafetensors_model_loader(monkeypatch, queue_size):
    monkeypatch.setenv("VLLM_FASTSAFETENSORS_QUEUE_SIZE", str(queue_size))
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
