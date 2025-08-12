import enum
import os
from functools import lru_cache
from typing import Type

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_cpu, is_hip

logger = init_logger(__name__)

VLLM_ATTENTION_BACKEND = "VLLM_ATTENTION_BACKEND"


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()


@lru_cache(maxsize=None)
def get_attn_backend(dtype: torch.dtype) -> Type[AttentionBackend]:
    backend = _which_attn_to_use(dtype)
    if backend == _Backend.FLASH_ATTN:
        logger.info("Using FlashAttention-2 backend.")
        from vllm.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    elif backend == _Backend.XFORMERS:
        logger.info("Using XFormers backend.")
        from vllm.attention.backends.xformers import (  # noqa: F401
            XFormersBackend)
        return XFormersBackend
    elif backend == _Backend.ROCM_FLASH:
        logger.info("Using ROCmFlashAttention backend.")
        from vllm.attention.backends.rocm_flash_attn import (  # noqa: F401
            ROCmFlashAttentionBackend)
        return ROCmFlashAttentionBackend
    elif backend == _Backend.TORCH_SDPA:
        logger.info("Using Torch SDPA backend.")
        from vllm.attention.backends.torch_sdpa import TorchSDPABackend
        return TorchSDPABackend
    else:
        raise ValueError("Invalid attention backend.")


def _which_attn_to_use(dtype: torch.dtype) -> _Backend:
    """Returns which flash attention backend to use."""
    if is_cpu():
        return _Backend.TORCH_SDPA

    if is_hip():
        # AMD GPUs.
        if torch.cuda.get_device_capability()[0] != 9:
            # not Instinct series GPUs.
            logger.info("flash_atten is not supported on NAVI GPUs.")
        return _Backend.ROCM_FLASH

    # NVIDIA GPUs.
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashAttention-2 backend for Volta and Turing "
                    "GPUs.")
        return _Backend.XFORMERS

    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention-2 backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return _Backend.XFORMERS

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info(
            "Cannot use FlashAttention-2 backend because the flash_attn "
            "package is not found. Please install it for better performance.")
        return _Backend.XFORMERS

    backend_by_env_var = os.getenv(VLLM_ATTENTION_BACKEND)
    if backend_by_env_var is not None:
        return _Backend[backend_by_env_var]

    # Default case.
    return _Backend.FLASH_ATTN
