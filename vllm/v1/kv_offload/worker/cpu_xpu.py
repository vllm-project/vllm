# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager

import torch

from vllm.v1.kv_offload.spec import CanonicalKVCaches
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers


class CpuXpuOffloadingHandlers(CpuGpuOffloadingHandlers):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
    ):
        with _torch_cuda_wrapper():
            super().__init__(
                kv_caches,
                block_size_factor,
                num_cpu_blocks,
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
