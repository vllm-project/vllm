# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU implementation of the CPU offload platform services."""

from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.engines.base import CopyEngine
from vllm.v1.kv_offload.cpu.engines.factory import CopyEngineFactory
from vllm.v1.kv_offload.cpu.engines.torch import TorchEngineConfig
from vllm.v1.kv_offload.cpu.platforms.base import (
    DeviceRuntime,
    RuntimeCapabilities,
)

logger = init_logger(__name__)


class XPURuntime(DeviceRuntime):
    """Platform services for Intel XPUs.

    XPU lacks CUDA's unified virtual address space, so the Triton kernel's
    `tl.load(cpu_ptr)` is invalid and the DMA engine is the only path.
    """

    capabilities = RuntimeCapabilities(cpu_offload=True, shared_region=True)

    def create_copy_engine(
        self,
        *,
        device_to_cpu: bool,
        host_memory_is_pinned: bool,
        page_sizes: Sequence[int],
    ) -> CopyEngine:
        # Resolved here, not at module import, so importing this module on a
        # non-XPU build is safe.
        return CopyEngineFactory.create(
            "torch",
            config=TorchEngineConfig(
                device_module=torch.xpu,
                copy_fn=ops.swap_blocks_batch,
                ptr_dtype=torch.uint64,
            ),
            device_to_cpu=device_to_cpu,
        )

    def register_host(self, ptr: int, num_bytes: int) -> bool:
        if not ptr or num_bytes <= 0:
            return False
        if not torch.ops._C.xpu_host_register(ptr, num_bytes):
            logger.warning(
                "xpu_host_register failed — transfers will still work "
                "but may be slower (unpinned DMA)"
            )
            return False
        return True

    def unregister_host(self, ptr: int) -> None:
        if not ptr:
            return
        if not torch.ops._C.xpu_host_unregister(ptr):
            logger.warning("xpu_host_unregister failed")
