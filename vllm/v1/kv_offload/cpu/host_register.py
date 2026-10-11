# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration of existing host memory for pinned DMA transfers.

Host allocations that the offloading workers share across processes (the
`SharedOffloadRegion` mmap) cannot be allocated through the accelerator's
pinned allocator, so they must be registered after the fact. Pinning is a
pure bandwidth optimization: when registration is unavailable the transfers
still run correctly, just as pageable (slower) copies.
"""

import torch

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _load_cudart() -> CudaRTLibrary | None:
    try:
        return CudaRTLibrary()
    except (AssertionError, AttributeError, OSError):
        logger.warning(
            "Could not load the CUDA runtime for host registration; "
            "the offload region stays pageable",
            exc_info=True,
        )
        return None


def host_register(ptr: int, num_bytes: int) -> bool:
    """Register a host range as device-accessible pinned memory.

    Args:
        ptr: Base address of the host range.
        num_bytes: Size of the host range in bytes.

    Returns:
        True if the range was registered and later needs `host_unregister`.

    """
    if not ptr or num_bytes <= 0:
        return False

    if current_platform.is_cuda_alike():
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

    if current_platform.is_xpu():
        if not torch.ops._C.xpu_host_register(ptr, num_bytes):
            logger.warning(
                "xpu_host_register failed — transfers will still work "
                "but may be slower (unpinned DMA)"
            )
            return False
        return True

    logger.info_once(
        "Host registration is not implemented on %s; KV offload transfers will "
        "use unpinned host memory.",
        current_platform.device_name,
    )
    return False


def host_unregister(ptr: int) -> None:
    """Release a host range previously registered by `host_register`."""
    if not ptr:
        return

    if current_platform.is_cuda_alike():
        cudart = _load_cudart()
        if cudart is None:
            return
        result = cudart.cudaHostUnregister(ptr)
        if result != 0:
            cudart.cudaGetLastError()
            logger.warning("cudaHostUnregister failed (code=%d)", result)
        return

    if current_platform.is_xpu():
        if not torch.ops._C.xpu_host_unregister(ptr):
            logger.warning("xpu_host_unregister failed")
        return
