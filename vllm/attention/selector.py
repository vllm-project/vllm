import os
from functools import lru_cache
from typing import Type

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_attn_backend(dtype: torch.dtype) -> Type[AttentionBackend]:
    if _which_attn_to_use(dtype) == "FlashAttention":
        logger.info("Using FlashAttention backend.")
        from vllm.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    elif _which_attn_to_use(dtype) == "TritonFlashAttention":
        logger.info("Using TritonFlashAttention backend.")
        from vllm.attention.backends.triton_flash_attn import (  # noqa: F401
            TritonFlashAttentionBackend)
        return TritonFlashAttentionBackend
    else:
        logger.info("Using XFormers backend.")
        from vllm.attention.backends.xformers import (  # noqa: F401
            XFormersBackend)
        return XFormersBackend


def _which_attn_to_use(dtype: torch.dtype) -> str:
    """Returns which flash attention backend to use.

    Returns:
        str: XFormers, FlashAttention, or TritonFlashAttention
    """

    # NOTE: Allow for switching between Triton and FA
    #       Defaulting to triton FA for AMD cards.
    use_triton_flash_attn = (os.environ.get("VLLM_USE_TRITON_FLASH_ATTN",
                                            "True").lower()
                             in ("true", "1")) and is_hip()

    if not is_hip() and torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashAttention backend for Volta and Turing "
                    "GPUs.")
        return "XFormers"

    if is_hip() and torch.cuda.get_device_capability()[0] != 9:
        # not Instinct series GPUs.
        logger.info("flash_atten is not supported on NAVI GPUs. "
                    "Using xformers backend.")
        return "XFormers"

    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return "XFormers"

    if not use_triton_flash_attn:
        # Only test for flash_attn if we are using it.
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.info(
                "Cannot use FlashAttention because the package is not found. "
                "Please install it for better performance.")
            return "XFormers"

    return "TritonFlashAttention" if use_triton_flash_attn else "FlashAttention"
