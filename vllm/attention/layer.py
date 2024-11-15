"""Attention layer."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata, AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod


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
        logits_soft_cap: Optional[float] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            sliding_window = cache_config.sliding_window
            is_attention_free = cache_config.is_attention_free
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            sliding_window = None
            is_attention_free = False
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self._k_scale = 1.0
        self._v_scale = 1.0
        self._k_zero_point = 0
        self._v_zero_point = 0
        k_scaling_factor_lists = v_scaling_factor_lists = [1.0, 0.0]
        self._k_scaling_factor = torch.Tensor(k_scaling_factor_lists).type(torch.float32).to("cuda")
        self._v_scaling_factor = torch.Tensor(v_scaling_factor_lists).type(torch.float32).to("cuda")
        self._quant_group = cache_config.kv_quant_group
        if cache_config.cache_dtype == "int8":
            # self._k_scale = 0.16
            # self._v_scale = 0.005
            self._k_scale = 1.0
            self._v_scale = 1.0
            if cache_config.kv_quant_params_path is not None:
                if type(cache_config.kv_quant_params[0][0]) == list and len(cache_config.kv_quant_params[0][0])==1:
                    self._k_scale = cache_config.kv_quant_params[0].pop(0)[0]
                    self._v_scale = cache_config.kv_quant_params[1].pop(0)[0]
                    self._k_zero_point = cache_config.kv_quant_params[2].pop(0)[0]
                    self._v_zero_point = cache_config.kv_quant_params[3].pop(0)[0]
                elif type(cache_config.kv_quant_params[0][0]) == list:
                    k_scaling_factor_lists = cache_config.kv_quant_params[0].pop(0)
                    v_scaling_factor_lists = cache_config.kv_quant_params[1].pop(0)
                    self._k_scaling_factor = torch.Tensor(k_scaling_factor_lists).type(torch.float32).to("cuda")
                    self._v_scaling_factor = torch.Tensor(v_scaling_factor_lists).type(torch.float32).to("cuda")
                    self._k_scaling_factor = self._k_scaling_factor.reshape((-1, num_kv_heads, head_size//self._quant_group))
                    self._v_scaling_factor = self._v_scaling_factor.reshape((-1, num_kv_heads, head_size//self._quant_group))
        quant_method = quant_config.get_quant_method(
            self, prefix=prefix) if quant_config else None
        if quant_method is not None:
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype,
                                        block_size, is_attention_free,
                                        blocksparse_params is not None)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params, logits_soft_cap)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:

        return self.impl.forward(query,
                                 key,
                                 value,
                                 kv_cache,
                                 attn_metadata,
                                 self._quant_group,
                                 self._k_scaling_factor,
                                 self._v_scaling_factor,
                                 self._k_scale,
                                 self._v_scale,
                                 attn_type=attn_type)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s
