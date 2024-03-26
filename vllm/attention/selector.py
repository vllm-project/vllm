from functools import lru_cache

import torch
import os


from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_attn_backend(dtype: torch.dtype) -> AttentionBackend:
    if _which_attn_to_use(dtype) == "FlashAttention":
        logger.info("Using FlashAttention backend.")
        from vllm.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    elif _which_attn_to_use(dtype) == "FlashAttentionTriton":
        logger.info("Using FlashAttentionTriton backend.")
        from vllm.attention.backends.flash_attn_triton import (  # noqa: F401
            FlashAttentionTritonBackend)
        return FlashAttentionTritonBackend
    else:
        logger.info("Using XFormers backend.")
        from vllm.attention.backends.xformers import (  # noqa: F401
            XFormersBackend)
        return XFormersBackend


def _which_attn_to_use(dtype: torch.dtype) -> str:
    """Returns which flash attention backend to use.

    Returns:
        str: XFormers, FlashAttention, or FlashAttentionTriton
    """
     
    # NOTE: Defaulting to triton FA for AMD cards.
    use_flash_attn_triton = os.environ.get('VLLM_USE_FLASH_ATTN_TRITON', "True").lower() in ("true", "1")
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

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info("flash_attn is not found.")
        if is_hip() and use_flash_attn_triton:
            pass
        else:
            return "XFormers" 
    return "FlashAttentionTriton" if use_flash_attn_triton else "FlashAttention"
