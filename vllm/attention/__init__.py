from functools import lru_cache

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def get_attention_backend(dtype: torch.dtype) -> AttentionBackend:
    if _can_use_flashinfer(dtype):
        logger.info("Using FlashInfer backend.")
        from vllm.attention.backends.flash_infer import FlashInferBackend  # noqa: F401
        return FlashInferBackend
    else:
        logger.info("Using XFormers backend.")
        from vllm.attention.backends.xformers import XFormersBackend  # noqa: F401
        return XFormersBackend


def _can_use_flashinfer(dtype: torch.dtype) -> bool:
    if is_hip():
        # AMD GPUs.
        logger.info("Cannot use FlashInfer backend for AMD GPUs.")
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashInfer backend for Volta and Turing GPUs.")
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashInfer backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return False

    try:
        import flashinfer  # noqa: F401
    except ImportError:
        logger.info("flashinfer is not found.")
        return False

    # TODO(woosuk): Use xFormers backend if flash_attn is not found.
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info("flash_attn is not found.")
        return False
