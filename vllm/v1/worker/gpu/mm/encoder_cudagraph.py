# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

logger = init_logger(__name__)


class EncoderCudaGraph:
    """V2 lifecycle adapter for encoder CUDA graphs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: "SupportsEncoderCudaGraph",
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.device = device
        from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

        self.manager: EncoderCudaGraphManager = EncoderCudaGraphManager(
            vllm_config=vllm_config,
            device=device,
            dtype=dtype,
            model=model,
        )

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig | None,
        model: SupportsMultiModal,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "EncoderCudaGraph | None":
        if vllm_config is None:
            return None

        from vllm.model_executor.models.interfaces import (
            SupportsEncoderCudaGraph,
            supports_encoder_cudagraph,
        )

        if (
            vllm_config.model_config.enforce_eager
            or not vllm_config.compilation_config.cudagraph_mm_encoder
            or not supports_encoder_cudagraph(model)
        ):
            return None
        return cls(
            vllm_config,
            cast(SupportsEncoderCudaGraph, model),
            device,
            dtype,
        )

    def execute(
        self,
        modality: str,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor] | None:
        if not self.manager.supports_modality(modality):
            return None
        return self.manager.execute(mm_kwargs)

    @torch.inference_mode()
    def profile_memory(self) -> int:
        logger.info(
            "Profiling encoder CUDA graph memory for %d graphs",
            self.manager.get_num_graphs_to_capture(),
        )
        try:
            gc.collect()
            torch.accelerator.empty_cache()
            memory_before = torch.accelerator.get_memory_info()[0]
            self.capture()
            memory_after = torch.accelerator.get_memory_info()[0]
            memory_used = max(memory_before - memory_after, 0)
        finally:
            self.clear()
            gc.collect()
            torch.accelerator.empty_cache()

        logger.info(
            "Estimated encoder CUDA graph memory: %.2f GiB",
            memory_used / (1 << 30),
        )
        return memory_used

    @torch.inference_mode()
    def capture(self) -> None:
        from vllm.distributed.parallel_state import graph_capture
        from vllm.platforms import current_platform

        with graph_capture(device=self.device):
            self.manager.capture(graph_pool=current_platform.graph_pool_handle())
            torch.accelerator.synchronize()

    def clear(self) -> None:
        self.manager.clear()
