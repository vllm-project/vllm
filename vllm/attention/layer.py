"""Attention layer."""
from typing import List, Optional

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import (AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.selector import get_attn_backend


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
        self.backend = get_attn_backend(torch.get_default_dtype())
        impl_cls = self.backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata[AttentionMetadataPerStage],
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        return self.impl.forward(query, key, value, kv_cache, attn_metadata,
                                 kv_scale)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        return s
