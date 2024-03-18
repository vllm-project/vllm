"""Attention layer."""
from functools import lru_cache
from typing import List, Optional

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip

logger = init_logger(__name__)


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.

    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens -------------->|	
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|

    Otherwise, the layout is as follows:	
    |<------------------ num_generation_tokens (M) ----------------->|	
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Output a flattened 1D tensor.
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
        if _use_flash_attn():
            from vllm.model_executor.layers.attention.backends.flash_attn import FlashAttentionBackend  # noqa: E501
            self.backend = FlashAttentionBackend(num_heads, head_size, scale,
                                                 num_kv_heads, alibi_slopes,
                                                 sliding_window)
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
def _use_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info("flash_attn is not found. Using xformers backend.")
        return False

    if is_hip():
        # AMD GPUs.
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("flash_attn is not supported on Turing or older GPUs. "
                    "Using xformers backend.")
        return False
    if torch.get_default_dtype() not in (torch.float16, torch.bfloat16):
        logger.info(
            "flash_attn only supports torch.float16 or torch.bfloat16. "
            "Using xformers backend.")
        return False

    logger.info("Using flash_attn backend.")
    return True
