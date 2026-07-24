# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for LoRA Async Loading with CUDA stream-based pipelining.
"""

import os
from unittest.mock import patch

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.async_loading import AsyncLoadLoRAMixin
from vllm.lora.model_manager import LoRAModelManager
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


DEVICES = (
    [f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)]
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
def test_cuda_stream_initialization(dist_init, dummy_model, device, async_enabled):
    """Verify CUDA loading stream and lora_ready flags are created
    when async loading is enabled, and absent when disabled."""

    if async_enabled and not current_platform.is_cuda_alike():
        pytest.skip("Async LoRA loading requires CUDA")

    env_value = "1" if async_enabled else "0"
    with patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": env_value}):
        manager = create_lora_manager(dummy_model, device)

        if async_enabled:
            # Loading stream must exist
            assert hasattr(manager, "lora_loading_stream")
            assert isinstance(manager.lora_loading_stream, torch.cuda.Stream)

            # Every LoRA module should have a lora_ready flag
            for module_name, module in manager.modules.items():
                assert hasattr(module, "lora_ready"), (
                    f"Module {module_name} missing lora_ready flag"
                )
                assert module.lora_ready.dtype == torch.uint32
        else:
            assert not hasattr(manager, "lora_loading_stream")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Async LoRA loading requires CUDA"
)
def test_sync_lora_loads_calls_wait_when_enabled():
    """Verify _sync_lora_loads calls wait_for_lora_flag when async enabled."""
    with (
        patch.dict(os.environ, {"VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA": "1"}),
        patch("vllm.lora.async_loading.wait_for_lora_flag") as mock_wait,
    ):

        class TestLayer(AsyncLoadLoRAMixin):
            pass

        layer = TestLayer()
        layer.lora_ready = torch.ones(1, dtype=torch.uint32)
        layer._sync_lora_loads()
        mock_wait.assert_called_once_with(layer.lora_ready, 1)
