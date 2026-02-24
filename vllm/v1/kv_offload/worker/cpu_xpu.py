# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager

import torch

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers


class CpuXpuOffloadingHandlers(CpuGpuOffloadingHandlers):
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        with _torch_cuda_wrapper():
            super().__init__(
                gpu_block_size,
                cpu_block_size,
                num_cpu_blocks,
                gpu_caches,
                attn_backends,
            )


@contextmanager
def _torch_cuda_wrapper():
    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.cuda.Stream = torch.xpu.Stream
        torch.cuda.stream = torch.xpu.stream
        torch.cuda.current_stream = torch.xpu.current_stream
        yield
    finally:
        pass
