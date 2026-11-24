# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm overrides for the CUDA-API implementation.

ROCm exposes the same `torch.cuda` and `CudaRTLibrary` (HIP) surface, so only
the differences live here: ROCm host mappings are not dereferenceable by the
Triton kernel, and neither the shared mmap region nor deferred host setup is
used.
"""

from collections.abc import Callable, Sequence

from vllm import _custom_ops as ops
from vllm.v1.kv_offload.cpu.platforms.base import RuntimeCapabilities
from vllm.v1.kv_offload.cpu.platforms.cuda import CudaRuntime


class RocmRuntime(CudaRuntime):
    """Platform services for ROCm GPUs."""

    # Deferred host setup, the only caller of capture relaxation, is off.
    capabilities = RuntimeCapabilities(cpu_offload=True)

    def _select_copy_fn(
        self,
        device_to_cpu: bool,
        host_memory_is_pinned: bool,
        page_sizes: Sequence[int],
    ) -> Callable[..., None]:
        return ops.swap_blocks_batch
