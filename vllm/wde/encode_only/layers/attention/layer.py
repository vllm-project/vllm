from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.wde.core.layers.attention.abstract import AttentionType
from vllm.wde.encode_only.layers.attention.backends.abstract import (
    EncodeOnlyAttentionBackend, EncodeOnlyAttentionMetadata)


class EncodeOnlyAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        prefix: str = "",
        attn_backend: Optional[EncodeOnlyAttentionBackend] = None,
    ) -> None:
        super().__init__()
        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self._k_scale = 1.0
        self._v_scale = 1.0
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

        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, None, blocksparse_params,
                             logits_soft_cap)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: EncodeOnlyAttentionMetadata,
        attn_type: AttentionType = AttentionType.ENCODER,
    ) -> torch.Tensor:

        return self.impl.forward(query,
                                 key,
                                 value,
                                 attn_metadata,
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
