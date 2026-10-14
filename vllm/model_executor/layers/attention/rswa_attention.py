# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.core.single_type_kv_cache_manager import RSWAManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    get_kv_quant_mode,
)


class RSWAAttention(Attention):
    """Attention layer whose KV cache is managed by ``RSWAManager``.

    Drop-in replacement for the standard ``Attention`` layer when the model is
    configured with Reference Sliding Window Attention (R-SWA,
    ``rswa_window > 0``). The actual masking logic lives in the attention
    backend (FlexAttention or FA4 mask_mod); this layer only overrides
    ``get_kv_cache_config`` so the KV cache manager instantiates ``RSWAManager``
    (instead of ``FullAttentionManager``) and can therefore evict "gap" blocks
    to keep per-request KV memory bounded at O(prefix + window).
    """

    def __init__(self, *args, rswa_window: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rswa_window = rswa_window

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig | None:
        if super().get_kv_cache_config(vllm_config) is None:
            return None
        spec = AttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
        return RSWAManager.layer_config(spec, rswa_window=self._rswa_window)
