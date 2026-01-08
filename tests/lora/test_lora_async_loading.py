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


def create_lora_manager(dummy_model, device):
    """Helper to create LoRAModelManager with standard test config."""
    return LoRAModelManager(
        dummy_model,
        2,
        2,
        2,
        LoRAConfig(
            max_lora_rank=8, max_cpu_loras=8, max_loras=8, lora_dtype=DEFAULT_DTYPE
        ),
        device=device,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("async_enabled", [True, False])
def test_nvshmem_initialization(dist_init, dummy_model, device, async_enabled):
    """Verify NVSHMEM initialization based on async loading setting."""
    if async_enabled and not current_platform.is_cuda_alike():
        pytest.skip("NVSHMEM requires CUDA")

    env_value = "1" if async_enabled else "0"
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": env_value}),
        patch("vllm.lora.model_manager.nvshmem", create=True) as mock_nvshmem,
        patch("vllm.lora.async_loading.nvshmem_tensor", create=True) as mock_tensor,
    ):
        if async_enabled:
            mock_nvshmem.get_unique_id.return_value = MagicMock()
            mock_tensor.return_value = torch.zeros(1, dtype=torch.int64)

        create_lora_manager(dummy_model, device)

        if async_enabled:
            mock_nvshmem.init.assert_called_once()
        else:
            mock_nvshmem.init.assert_not_called()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("async_enabled", [True, False])
def test_loading_stream_creation(dist_init, dummy_model, device, async_enabled):
    """Verify loading stream creation based on async loading setting."""
    if async_enabled and not current_platform.is_cuda_alike():
        pytest.skip("NVSHMEM requires CUDA")

    env_value = "1" if async_enabled else "0"
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": env_value}),
        patch("vllm.lora.model_manager.nvshmem", create=True) as mock_nvshmem,
        patch("vllm.lora.async_loading.nvshmem_tensor", create=True) as mock_tensor,
    ):
        if async_enabled:
            mock_nvshmem.get_unique_id.return_value = MagicMock()
            mock_tensor.return_value = torch.zeros(1, dtype=torch.int64)

        manager = create_lora_manager(dummy_model, device)

        if async_enabled:
            assert hasattr(manager, "lora_loading_stream")
            assert manager.lora_loading_stream is not None
        else:
            assert not hasattr(manager, "lora_loading_stream")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="NVSHMEM requires CUDA"
)
def test_sync_lora_loads_calls_wait_when_enabled():
    """Verify _sync_lora_loads calls nvshmem wait when async loading enabled."""
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": "1"}),
        patch("vllm.lora.async_loading.nvshmem_wait_for_lora_flag") as mock_wait,
    ):
        from vllm.lora.async_loading import AsyncLoRAMixin

        class TestLayer(AsyncLoRAMixin):
            pass

        layer = TestLayer()
        layer.lora_ready = torch.ones(1, dtype=torch.int64)
        layer._sync_lora_loads()
        mock_wait.assert_called_once_with(layer.lora_ready, 1)
