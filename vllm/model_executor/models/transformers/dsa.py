# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Sparse Attention (DSA) for the Transformers modeling backend."""

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    DeepseekV32IndexerCache,
    SparseAttnIndexer,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from vllm.config import VllmConfig

TOPK_SELECTOR_ATTR = "topk_selector"
"""Attribute of a Transformers DSA indexer holding the module that scores the keys
and selects the top-k, which is the only part of the indexer that uses its cache."""

QUANT_BLOCK_SIZE = 128
"""Group size for quantizing indexer queries and keys to FP8."""


def record_topk_ready(topk_indices_buffer: torch.Tensor, layer_name: str) -> None:
    """Tell the sparse MLA layer `layer_name` that its top-k is in
    `topk_indices_buffer`. An op, because Dynamo cannot replay the CUDA event."""
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.impl.record_logical_topk_ready()


direct_register_custom_op(
    "transformers_record_topk_ready",
    record_topk_ready,
    mutates_args=["topk_indices_buffer"],
    fake_impl=lambda topk_indices_buffer, layer_name: None,
)


class TopkSelector(nn.Module):
    """Replaces a Transformers DSA indexer's `topk_selector`, keeping the indexer keys
    in the paged KV cache and writing the top-k to the buffer the sparse MLA layers
    read."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        config: "PreTrainedConfig",
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ):
        super().__init__()
        head_dim = config.index_head_dim
        if head_dim != QUANT_BLOCK_SIZE:
            # The query scale is folded into the per-head weights
            raise NotImplementedError(
                f"DSA indexers need `index_head_dim={QUANT_BLOCK_SIZE}`, "
                f"got {head_dim}."
            )
        # FP8 keys with one float32 scale per quantization block
        k_cache = DeepseekV32IndexerCache(
            head_dim=head_dim + head_dim // QUANT_BLOCK_SIZE * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=vllm_config.cache_config,
        )
        self.sparse_attn_indexer = SparseAttnIndexer(
            k_cache,
            QUANT_BLOCK_SIZE,
            "ue8m0",
            config.index_topk,
            head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            topk_indices_buffer,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        cache: None = None,
    ) -> torch.Tensor:
        # [batch=1, num_tokens, ...] -> [num_tokens, ...]
        query, key, weights = (x.flatten(0, 1) for x in (query, key, weights))
        query_fp8, query_scale = per_token_group_quant_fp8(
            query.flatten(0, 1), QUANT_BLOCK_SIZE, use_ue8m0=True
        )
        # Fold the per-head query scale into the head weights
        weights = weights * query_scale.view(weights.shape) * scaling
        # Keys are quantized as they are written to the indexer cache. The first
        # argument only provides the number of tokens and the device.
        return self.sparse_attn_indexer(key, query_fp8.view(query.shape), key, weights)
