# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
        )
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0

        instanttensor_tensors = {}
        hf_safetensors_tensors = {}

        for name, tensor in instanttensor_weights_iterator(safetensors, True):
            # Copy the tensor immediately as it is a reference to the internal
            # buffer of instanttensor.
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
