# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS Model Runner for Apple Silicon."""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.cpu_model_runner import (
    _set_global_compilation_settings,
    _torch_cuda_wrapper,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class MPSModelRunner(GPUModelRunner):
    """Model runner for MPS (Metal Performance Shaders) on Apple Silicon."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert device == torch.device("mps"), f"Expected MPS device, got {device}"

        # Initialize with MPS device - inputs and model will be on MPS
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert self.speculative_config is None, (
            "Speculative decoding is not supported on MPS."
        )

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

    def load_model(self, eep_scale_up: bool = False) -> None:
        """Load the model onto MPS device."""
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def warming_up_model(self) -> None:
        """Warm up the model for compilation."""
        logger.info("Warming up model for MPS...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )
        logger.info("Warming up done.")

    def _init_device_properties(self) -> None:
        """Initialize device properties for MPS."""
        # MPS doesn't have the same device properties as CUDA
        pass

    def _sync_device(self) -> None:
        """Synchronize MPS device."""
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        """For MPS backend, dp padding is not required for now."""
        return 0, None

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        """Convert sampled token IDs from MPS tensor to list.

        Direct tolist() is faster than sync + pinned memory copy for small
        tensors like sampled token IDs (batch_size x 1).
        """
        return sampled_token_ids.tolist()
