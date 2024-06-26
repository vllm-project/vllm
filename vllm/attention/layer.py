"""Attention layer."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


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
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            sliding_window = cache_config.sliding_window
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            sliding_window = None
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default kv_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized kv_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self._kv_scale = 1.0
        quant_method = quant_config.get_quant_method(
            self) if quant_config else None
        if quant_method is not None:
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # When FP8 quantization is enabled, we make a parameter
            # "kv_scale" so that it can be loaded from FP8 checkpoint.
            # The kv_scale will then be converted back
            # to self._kv_scale in a native float32 value after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(num_heads, head_size, num_kv_heads,
                                        sliding_window, dtype, kv_cache_dtype,
                                        block_size, blocksparse_params
                                        is not None)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.impl.forward(query, key, value, kv_cache, attn_metadata,
                                 self._kv_scale)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s
