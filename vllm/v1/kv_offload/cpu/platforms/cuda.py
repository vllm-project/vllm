# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA implementation of the CPU offload platform services.

ROCm reuses these classes (see `rocm.py`); everything in this module is the
CUDA API surface that both implement, with NVIDIA defaults.
"""

import ctypes
import functools
from collections.abc import Callable, Sequence

import torch

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.kv_offload.cpu.engines.base import CopyEngine
from vllm.v1.kv_offload.cpu.engines.factory import CopyEngineFactory
from vllm.v1.kv_offload.cpu.engines.torch import TorchEngineConfig
from vllm.v1.kv_offload.cpu.platforms.base import (
    DeviceRuntime,
    RuntimeCapabilities,
)

logger = init_logger(__name__)

# Triton fast path constants, tuned empirically on H100 (PCIe Gen5):
#   NUM_SMS         - smallest SM slice within 5% of peak bandwidth at the
#                     8-32 KB block sizes that matter in practice
#   THRESHOLD_BYTES - max payload per descriptor where Triton beats DMA; above
#                     this the C++ cuMemcpyBatchAsync path takes the lead
#   MIN_N           - minimum batch size where Triton's per-launch cost is
#                     amortized; below this DMA wins
NUM_SMS = 12
THRESHOLD_BYTES = 28 * 1024
MIN_N = 16


@triton.jit
def _swap_blocks_kernel(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + job) // 8
        for start in range(0, words, WORDS_PER_CHUNK):
            idx = start + offsets
            mask = idx < words
            data = tl.load(src + idx, mask=mask, other=0)
            tl.store(dst + idx, data, mask=mask)
        job += num_progs


def swap_blocks_batch(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    *,
    bytes_per_chunk: int,
) -> None:
    """Triton implementation of `swap_blocks_batch` for small CPU->device batches.

    Selected only by the CUDA runtime, so the device is CUDA.
    """
    n = src_addrs.numel()
    # Too few descriptors to amortize Triton's launch cost.
    if n < MIN_N:
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        src_addrs.to("cuda", non_blocking=True),
        dst_addrs.to("cuda", non_blocking=True),
        sizes.to("cuda", non_blocking=True),
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )


def _clear_pending_device_error() -> None:
    """Clear the error a failed copy left pending for the next kernel launch."""
    try:
        CudaRTLibrary().cudaGetLastError()
    except (AssertionError, AttributeError, OSError):
        logger.warning("Could not clear the pending device error", exc_info=True)


def _relax_stream_capture() -> None:
    """Let this thread call cudaHostRegister while another captures a graph.

    A global-mode capture otherwise fails on such a call from any thread.
    """
    mode = ctypes.c_int(2)  # cudaStreamCaptureModeRelaxed
    lib = CudaRTLibrary().lib
    rc = lib.cudaThreadExchangeStreamCaptureMode(ctypes.byref(mode))
    if rc != 0:
        lib.cudaGetLastError()
        raise RuntimeError(f"cudaThreadExchangeStreamCaptureMode failed (code={rc})")


@functools.cache
def _get_cudart() -> CudaRTLibrary:
    return CudaRTLibrary()


def _load_cudart() -> CudaRTLibrary | None:
    try:
        return _get_cudart()
    except (AssertionError, AttributeError, OSError):
        logger.warning(
            "Could not load the CUDA runtime for host registration; "
            "the offload region stays pageable",
            exc_info=True,
        )
        return None


def _host_register(ptr: int, num_bytes: int) -> bool:
    """Register a host range as device-accessible pinned memory."""
    if not ptr or num_bytes <= 0:
        return False
    cudart = _load_cudart()
    if cudart is None:
        return False
    result = cudart.cudaHostRegister(ptr, num_bytes)
    if result != 0:
        cudart.cudaGetLastError()
        logger.warning(
            "cudaHostRegister failed (code=%d) — transfers will still work "
            "but may be slower (unpinned DMA)",
            result,
        )
        return False
    return True


def _host_unregister(ptr: int) -> None:
    """Release a host range previously registered by `_host_register`."""
    if not ptr:
        return
    cudart = _load_cudart()
    if cudart is None:
        return
    result = cudart.cudaHostUnregister(ptr)
    if result != 0:
        cudart.cudaGetLastError()
        logger.warning("cudaHostUnregister failed (code=%d)", result)


class CudaRuntime(DeviceRuntime):
    """Platform services for NVIDIA GPUs."""

    capabilities = RuntimeCapabilities(
        cpu_offload=True, shared_region=True, async_init=True
    )

    def _select_copy_fn(
        self,
        device_to_cpu: bool,
        host_memory_is_pinned: bool,
        page_sizes: Sequence[int],
    ) -> Callable[..., None]:
        # device->CPU is bandwidth-bound; the dedicated copy engine beats
        # Triton. The Triton kernel dereferences CPU pointers on the GPU,
        # which is only valid for pinned host memory.
        if device_to_cpu or not host_memory_is_pinned:
            return ops.swap_blocks_batch
        # Fall back to the C++ DMA path where Triton isn't usable.
        if not HAS_TRITON:
            return ops.swap_blocks_batch
        # Triton wins only on small, 8-byte-aligned payloads.
        if (
            not page_sizes
            or max(page_sizes) >= THRESHOLD_BYTES
            or any(s % 8 for s in page_sizes)
        ):
            return ops.swap_blocks_batch
        chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
        return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)

    def create_copy_engine(
        self,
        *,
        device_to_cpu: bool,
        host_memory_is_pinned: bool,
        page_sizes: Sequence[int],
    ) -> CopyEngine:
        return CopyEngineFactory.create(
            "torch",
            config=TorchEngineConfig(
                device_module=torch.cuda,
                copy_fn=self._select_copy_fn(
                    device_to_cpu, host_memory_is_pinned, page_sizes
                ),
                clear_error=_clear_pending_device_error,
            ),
            device_to_cpu=device_to_cpu,
        )

    def relax_stream_capture(self) -> None:
        _relax_stream_capture()

    def register_host(self, ptr: int, num_bytes: int) -> bool:
        return _host_register(ptr, num_bytes)

    def unregister_host(self, ptr: int) -> None:
        _host_unregister(ptr)
