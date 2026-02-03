# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for LoRA Async Loading with Forward Compute using Nvshmem
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.model_manager import LoRAModelManager
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


DEVICES = (
    [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
    if current_platform.is_cuda_alike()
    else ["cpu"]
)

DEFAULT_DTYPE = torch.get_default_dtype()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="NVSHMEM requires CUDA"
)
@pytest.mark.parametrize("device", DEVICES)
def test_nvshmem_init_when_enabled(dist_init, dummy_model, device):
    """Verify NVSHMEM is initialized when async loading enabled."""
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": "1"}),
        patch("vllm.lora.model_manager.nvshmem", create=True) as mock_nvshmem,
        patch("vllm.lora.layers.base.nvshmem_tensor", create=True) as mock_tensor,
    ):
        mock_nvshmem.get_unique_id.return_value = MagicMock()
        mock_tensor.return_value = torch.zeros(1, dtype=torch.int64)
        _ = LoRAModelManager(
            dummy_model,
            2,
            2,
            2,
            LoRAConfig(
                max_lora_rank=8,
                max_cpu_loras=8,
                max_loras=8,
                lora_dtype=DEFAULT_DTYPE,
            ),
            device=device,
        )
        mock_nvshmem.init.assert_called_once()


@pytest.mark.parametrize("device", DEVICES)
def test_nvshmem_not_init_when_disabled(dist_init, dummy_model, device):
    """Verify NVSHMEM is NOT initialized when async loading disabled."""
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": "0"}),
        patch("vllm.lora.model_manager.nvshmem", create=True) as mock_nvshmem,
    ):
        _ = LoRAModelManager(
            dummy_model,
            2,
            2,
            2,
            LoRAConfig(
                max_lora_rank=8,
                max_cpu_loras=8,
                max_loras=8,
                lora_dtype=DEFAULT_DTYPE,
            ),
            device=device,
        )
        mock_nvshmem.init.assert_not_called()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="NVSHMEM requires CUDA"
)
@pytest.mark.parametrize("device", DEVICES)
def test_loading_stream_created_when_enabled(dist_init, dummy_model, device):
    """Verify loading stream is created when async loading enabled."""
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": "1"}),
        patch("vllm.lora.model_manager.nvshmem", create=True) as mock_nvshmem,
        patch("vllm.lora.layers.base.nvshmem_tensor", create=True) as mock_tensor,
    ):
        mock_nvshmem.get_unique_id.return_value = MagicMock()
        mock_tensor.return_value = torch.zeros(1, dtype=torch.int64)
        manager = LoRAModelManager(
            dummy_model,
            2,
            2,
            2,
            LoRAConfig(
                max_lora_rank=8,
                max_cpu_loras=8,
                max_loras=8,
                lora_dtype=DEFAULT_DTYPE,
            ),
            device=device,
        )
        assert hasattr(manager, "lora_loading_stream")
        assert manager.lora_loading_stream is not None


@pytest.mark.parametrize("device", DEVICES)
def test_loading_stream_not_created_when_disabled(dist_init, dummy_model, device):
    """Verify loading stream is NOT created when async loading disabled."""
    with patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": "0"}):
        manager = LoRAModelManager(
            dummy_model,
            2,
            2,
            2,
            LoRAConfig(
                max_lora_rank=8, max_cpu_loras=8, max_loras=8, lora_dtype=DEFAULT_DTYPE
            ),
            device=device,
        )
        assert not hasattr(manager, "lora_loading_stream")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="NVSHMEM requires CUDA"
)
def test_sync_lora_loads_calls_wait_when_enabled():
    """Verify _sync_lora_loads calls nvshmem wait when async loading enabled."""
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": "1"}),
        patch("vllm.lora.layers.base.nvshmem_wait_for_lora_flag") as mock_wait,
    ):
        from vllm.lora.layers.base import BaseLayerWithLoRA

        layer = BaseLayerWithLoRA()
        layer.lora_ready = torch.ones(1, dtype=torch.int64)
        layer._sync_lora_loads()
        mock_wait.assert_called_once_with(layer.lora_ready, 1)
