from functools import lru_cache

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_attn_backend(dtype: torch.dtype) -> AttentionBackend:
    if _can_use_flash_attn(dtype):
        logger.info("Using FlashAttention backend.")
        from vllm.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    else:
        logger.info("Using XFormers backend.")
        from vllm.attention.backends.xformers import (  # noqa: F401
            XFormersBackend)
        return XFormersBackend


def _can_use_flash_attn(dtype: torch.dtype) -> bool:
    if is_hip():
        # AMD GPUs.
        logger.info("Cannot use FlashAttention backend for AMD GPUs.")
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashAttention backend for Volta and Turing "
                    "GPUs.")
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return False

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info("flash_attn is not found.")
        return False
    return True
