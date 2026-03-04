# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.tracing import instrument
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class MPSModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("mps")
        assert self.speculative_config is None, "Spec decode is not supported on MPS."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

    @instrument(span_name="Loading (MPS)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        # Load model on CPU first to avoid MPS placeholder storage issues
        # (MPS tensors created via `with torch.device("mps"):` use lazy
        # allocation that can break copy_ and uniform_ during weight init).
        # After loading, move the whole model to MPS.
        import dataclasses

        cpu_load_config = dataclasses.replace(self.load_config, device="cpu")
        self.model = get_model(
            vllm_config=self.vllm_config,
            load_config=cpu_load_config,
        )
        self.model = self.model.to(self.device)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

    @instrument(span_name="Warmup (MPS)")
    def warming_up_model(self) -> None:
        logger.info("Warming up model on MPS...")
        self._dummy_run(
            min(
                max(16, self.max_num_reqs),
                self.scheduler_config.max_num_batched_tokens,
            )
        )
        # Flush all lazy MPS operations from warmup so they don't
        # surface as errors in the first real forward pass.
        torch.mps.synchronize()
        logger.info("Warmup done.")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        torch.mps.synchronize()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # The base class uses non_blocking copy to pinned CPU memory, but MPS
        # has pin_memory=False (unified memory) and the non_blocking MPS→CPU
        # copy through MPSGraph crashes on certain tensor shapes.
        # Use Event-based sync with a blocking copy instead.
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return sampled_token_ids.cpu().tolist()

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        return 0, None
