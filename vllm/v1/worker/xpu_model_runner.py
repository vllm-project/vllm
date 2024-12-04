import torch
import numpy as np
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.attention.backends.ipex_attn import (IPEXAttentionBackend)
if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.use_cuda_graph = False

    @torch.inference_mode()
    def profile_run(self) -> None:
        # self._dummy_run(self.model, self.max_num_tokens)
        torch.xpu.synchronize()

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = IPEXAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device))
