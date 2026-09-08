# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import KVCacheSpec, RSWASpec, get_kv_quant_mode


class RSWAAttention(Attention):
    """Attention layer that reports ``RSWASpec`` as its KV cache spec.

    Drop-in replacement for the standard ``Attention`` layer when the model is
    configured with Reference Sliding Window Attention (R-SWA,
    ``rswa_window > 0``). The actual masking logic lives in the attention
    backend (FlexAttention or FA4 mask_mod); this layer only overrides
    ``get_kv_cache_spec`` so the KV cache manager instantiates ``RSWAManager``
    (instead of ``FullAttentionManager``) and can therefore evict "gap" blocks
    to keep per-request KV memory bounded at O(prefix + window).
    """

    def __init__(self, *args, rswa_window: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rswa_window = rswa_window

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        return RSWASpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            rswa_window=self._rswa_window,
        )
