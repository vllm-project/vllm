# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 MLA attention on ROCm.

Reuses the NVIDIA layer and only drops the CUDA-only `FLASHMLA_SPARSE` sink
fallback, so the learnable sink is applied only by a selected backend that
supports `attn_sink`.
"""

import torch

from vllm.logger import init_logger
from vllm.models.hy_v4.nvidia.attention import (
    HYV4MLAAttention as _HYV4MLAAttention,
)
from vllm.models.hy_v4.nvidia.attention import (
    is_skip_topk_indexer_weight,
    sparse_layer_ids,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.selector import get_attn_backend

logger = init_logger(__name__)

__all__ = ["HYV4MLAAttention", "is_skip_topk_indexer_weight", "sparse_layer_ids"]


class HYV4MLAAttention(_HYV4MLAAttention):
    def _resolve_sink_backend(
        self, kv_cache_dtype: str
    ) -> type[AttentionBackend] | None:
        """Return the selected MLA backend if it can apply the learnable sink.

        Args:
            kv_cache_dtype: The layer's KV cache dtype string.

        Returns:
            The selected backend, or None when it cannot apply sinks; the
            caller then loads the sink weight but disables the bias.

        """
        try:
            selected_cls = get_attn_backend(
                head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                dtype=torch.get_default_dtype(),
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                use_sparse=self.is_sparse,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:
            # Stringify so warning_once dedupes across layers.
            logger.warning_once(
                "HYV4 failed to select an MLA backend for the learnable sink "
                "(%s); the sink parameter is loaded but the sink bias is "
                "disabled.",
                str(exc),
            )
            return None

        if selected_cls.supports_sink():
            return selected_cls

        logger.warning_once(
            "HYV4 learnable sink is unavailable: the selected backend %s cannot "
            "apply sinks. The sink parameter is loaded but the sink bias is "
            "disabled.",
            selected_cls.get_name(),
        )
        return None
