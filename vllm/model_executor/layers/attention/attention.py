"""Attention layer."""
from functools import lru_cache
from typing import List, Optional

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip
import os

logger = init_logger(__name__)


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        if use_triton := _use_flash_attn():
            from vllm.model_executor.layers.attention.backends.flash_attn import FlashAttentionBackend  # noqa: E501
            self.backend = FlashAttentionBackend(num_heads, head_size, scale,
                                                 num_kv_heads, alibi_slopes,
                                                 sliding_window,
                                                 use_triton == 2)
        else:
            from vllm.model_executor.layers.attention.backends.xformers import XFormersBackend  # noqa: E501
            self.backend = XFormersBackend(num_heads, head_size, scale,
                                           num_kv_heads, alibi_slopes,
                                           sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        return self.backend.forward(query, key, value, key_cache, value_cache,
                                    input_metadata)


@lru_cache(maxsize=1)
def _use_flash_attn() -> int:
    """Returns if and which flash attention to use.

    Returns:
        int: 0 for none, 1 for default implementation, 2 for triton implementation.
    """
    if not (os.environ.get('VLLM_USE_FLASH_ATTN_TRITON') and is_hip()):
        # AMD GPUs can use flash_attn package or triton impl.
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.info("flash_attn is not found. Using xformers backend.")
            return 0

    if (not is_hip()) and torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("flash_attn is not supported on Turing or older GPUs. "
                    "Using xformers backend.")
        return 0

    if is_hip() and torch.cuda.get_device_capability()[0] != 9:
        # not Instinct series GPUs.
        logger.info("flash_atten is not supported on NAVI GPUs. "
                    "Using xformers backend.")
        return 0

    if torch.get_default_dtype() not in (torch.float16, torch.bfloat16):
        logger.info(
            "flash_attn only supports torch.float16 or torch.bfloat16. "
            "Using xformers backend.")
        return 0

    logger.info(f"Using {'Triton' if os.environ.get('VLLM_USE_FLASH_ATTN_TRITON') else ''} flash_attn backend.")
    return 2 if os.environ.get('VLLM_USE_FLASH_ATTN_TRITON') else 1
