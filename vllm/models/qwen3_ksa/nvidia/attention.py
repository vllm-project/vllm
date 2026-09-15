# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged-cache KSA attention primitives.

The scheduler owns only normal text rows. The model expands virtual Summary
rows internally and uses this module to read and write vLLM-managed caches.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.qwen3_ksa.common.cache import (
    build_ksa_summary_slot_mapping,
    split_ksa_kv_cache,
)
from vllm.models.qwen3_ksa.common.metadata import (
    KSA_SUMMARY_CHUNK_SIZE,
    KSAAttentionBackend,
    KSAAttentionMetadata,
)
from vllm.models.qwen3_ksa.nvidia.paged_attention import (
    ksa_paged_source_attention,
)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)


def register_ksa_cache_owner(
    *,
    prefix: str,
    module: AttentionLayerBase,
    vllm_config: VllmConfig,
) -> None:
    if not prefix:
        raise ValueError("KSA cache owners require a non-empty layer prefix")
    context = vllm_config.compilation_config.static_forward_context
    if prefix in context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    context[prefix] = module


def register_ksa_cache_scales(module: nn.Module) -> None:
    module.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
    module.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))


def resolve_ksa_cache_dtype(
    cache_config: CacheConfig,
    vllm_config: VllmConfig,
) -> tuple[str, torch.dtype]:
    cache_dtype = cache_config.cache_dtype
    if cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("KSA currently requires a BF16 KV cache")
    dtype = kv_cache_dtype_str_to_dtype(cache_dtype, vllm_config.model_config)
    if dtype is not torch.bfloat16:
        raise NotImplementedError("KSA currently requires a BF16 KV cache")
    return cache_dtype, dtype


class KSATextCacheLayer(nn.Module, AttentionLayerBase):
    """Text KV cache owner nested under one Qwen3 KSA attention layer."""

    supports_dcp = False

    def __init__(
        self,
        *,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_chunk_num: int,
        is_small_layer: bool,
        cache_config: CacheConfig,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()
        if cache_config.block_size % KSA_SUMMARY_CHUNK_SIZE != 0:
            raise ValueError("KSA cache block size must be divisible by 8")
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_chunk_num = sliding_chunk_num
        self.is_small_layer = is_small_layer
        self.cache_config = cache_config
        self.cache_dtype, self.cache_torch_dtype = resolve_ksa_cache_dtype(
            cache_config, vllm_config
        )
        self.kv_cache = torch.tensor([])
        register_ksa_cache_scales(self)
        register_ksa_cache_owner(
            prefix=prefix,
            module=self,
            vllm_config=vllm_config,
        )

    def get_attn_backend(self) -> type[KSAAttentionBackend]:
        return KSAAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        common: dict[str, Any] = {
            "block_size": self.cache_config.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "head_size_v": self.head_dim,
            "dtype": self.cache_torch_dtype,
            "kv_quant_mode": get_kv_quant_mode(self.cache_dtype),
        }
        if self.is_small_layer:
            return SlidingWindowSpec(
                **common,
                sliding_window=(self.sliding_chunk_num + 1) * KSA_SUMMARY_CHUNK_SIZE,
            )
        return FullAttentionSpec(**common)

    def forward(self) -> None:
        raise RuntimeError("KSATextCacheLayer is a cache owner, not a callable layer")


def get_ksa_summary_cache_spec(
    *,
    cache_config: CacheConfig,
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
) -> FullAttentionSpec:
    if cache_config.block_size % KSA_SUMMARY_CHUNK_SIZE != 0:
        raise ValueError("KSA cache block size must be divisible by 8")
    cache_dtype, torch_dtype = resolve_ksa_cache_dtype(cache_config, vllm_config)
    return FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        head_size_v=head_dim,
        dtype=torch_dtype,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
        tokens_per_state=KSA_SUMMARY_CHUNK_SIZE,
    )


def get_ksa_attention_metadata(layer_name: str) -> KSAAttentionMetadata | None:
    metadata = get_forward_context().attn_metadata
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise NotImplementedError("KSA does not support speculative or DBO metadata")
    layer_metadata = metadata[layer_name]
    if not isinstance(layer_metadata, KSAAttentionMetadata):
        raise TypeError(f"invalid KSA metadata for {layer_name}")
    return layer_metadata


def _write_ksa_cache(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_layer: nn.Module,
    cache_dtype: str,
    head_dim: int,
) -> None:
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

    key_cache, value_cache = split_ksa_kv_cache(kv_cache, head_dim=head_dim)
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        cache_dtype,
        cache_layer._k_scale,
        cache_layer._v_scale,
    )


def _inline_ksa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("KSA query heads must be divisible by KV heads")
    kv_head_indices = torch.div(
        torch.arange(num_query_heads, device=query.device),
        num_query_heads // num_kv_heads,
        rounding_mode="floor",
    )
    expanded_key = key.index_select(1, kv_head_indices)
    output = value.index_select(1, kv_head_indices)
    lse = (query.float() * expanded_key.float()).sum(dim=-1).mul(scale).transpose(0, 1)
    return output, lse


def _merge_ksa_sources(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(prefix_output)
    merge_attn_states(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    )
    return output


def paged_ksa_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    expanded_batch: Any,
    text_cache_layer: KSATextCacheLayer,
    summary_cache_owner: AttentionLayerBase | None,
    summary_cache_layer_name: str | None,
    is_small_layer: bool,
    scale: float,
) -> torch.Tensor:
    """Run batched KSA attention over vLLM-owned paged caches."""
    forward_context = get_forward_context()
    text_metadata = get_ksa_attention_metadata(text_cache_layer.prefix)
    if text_metadata is None:
        if forward_context.attn_metadata is None:
            # Memory profiling intentionally omits attention metadata. Keep the
            # dummy path cheap without a GPU-to-CPU padding-mask read.
            return torch.zeros_like(query)
        raise RuntimeError("KSA text metadata is missing")
    if text_cache_layer.kv_cache.numel() == 0:
        raise RuntimeError("KSA text KV cache is not bound")
    if text_metadata.num_actual_tokens != expanded_batch.text_row_indices.numel():
        raise ValueError("KSA logical metadata row count does not match model input")

    text_key = key.index_select(0, expanded_batch.text_row_indices)
    text_value = value.index_select(0, expanded_batch.text_row_indices)
    text_positions = expanded_batch.row_logical_positions.index_select(
        0, expanded_batch.text_row_indices
    ).to(torch.int64)
    text_requests = expanded_batch.row_to_request.index_select(
        0, expanded_batch.text_row_indices
    ).to(torch.int32)
    text_valid = expanded_batch.text_row_is_valid
    if is_small_layer:
        chunk_indices = torch.div(
            text_positions,
            KSA_SUMMARY_CHUNK_SIZE,
            rounding_mode="floor",
        )
        visible_summary_lens = torch.clamp_min(
            chunk_indices - text_cache_layer.sliding_chunk_num,
            0,
        )
        text_start_positions = visible_summary_lens * KSA_SUMMARY_CHUNK_SIZE
    else:
        visible_summary_lens = torch.zeros_like(text_positions)
        text_start_positions = torch.zeros_like(text_positions)
    text_start_positions = torch.where(
        text_valid,
        text_start_positions,
        torch.zeros_like(text_start_positions),
    )
    text_end_positions = torch.where(
        text_valid,
        text_positions + 1,
        torch.zeros_like(text_positions),
    )
    visible_summary_lens = torch.where(
        text_valid,
        visible_summary_lens,
        torch.zeros_like(visible_summary_lens),
    )
    _write_ksa_cache(
        key=text_key,
        value=text_value,
        kv_cache=text_cache_layer.kv_cache,
        slot_mapping=text_metadata.slot_mapping,
        cache_layer=text_cache_layer,
        cache_dtype=text_cache_layer.cache_dtype,
        head_dim=text_cache_layer.head_dim,
    )

    summary_metadata: KSAAttentionMetadata | None = None
    summary_key = key.index_select(0, expanded_batch.summary_row_indices)
    summary_value = value.index_select(0, expanded_batch.summary_row_indices)
    summary_positions = expanded_batch.row_logical_positions.index_select(
        0, expanded_batch.summary_row_indices
    ).to(torch.int64)
    summary_requests = expanded_batch.row_to_request.index_select(
        0, expanded_batch.summary_row_indices
    ).to(torch.int32)
    summary_active = expanded_batch.summary_row_is_active
    if is_small_layer:
        if summary_cache_owner is None or summary_cache_layer_name is None:
            raise RuntimeError("small KSA layer is missing its Summary cache owner")
        summary_metadata = get_ksa_attention_metadata(summary_cache_layer_name)
        if summary_metadata is None:
            raise RuntimeError("KSA Summary metadata is missing")
        if summary_cache_owner.kv_cache.numel() == 0:
            raise RuntimeError("KSA Summary KV cache is not bound")
        summary_slots = build_ksa_summary_slot_mapping(
            token_positions=summary_positions,
            token_to_request=summary_requests,
            boundary_mask=summary_active,
            block_table=summary_metadata.block_table,
            manager_block_size=summary_metadata.manager_block_size,
            states_per_block=summary_metadata.states_per_block,
            summary_chunk_size=KSA_SUMMARY_CHUNK_SIZE,
        )
        if summary_slots.numel():
            _write_ksa_cache(
                key=summary_key,
                value=summary_value,
                kv_cache=summary_cache_owner.kv_cache,
                slot_mapping=summary_slots,
                cache_layer=summary_cache_owner,
                cache_dtype=text_cache_layer.cache_dtype,
                head_dim=text_cache_layer.head_dim,
            )

    text_query = query.index_select(0, expanded_batch.text_row_indices)
    text_key_cache, text_value_cache = split_ksa_kv_cache(
        text_cache_layer.kv_cache,
        head_dim=text_cache_layer.head_dim,
    )
    text_output, text_lse = ksa_paged_source_attention(
        query=text_query,
        key_cache=text_key_cache,
        value_cache=text_value_cache,
        block_table=text_metadata.block_table,
        row_to_request=text_requests,
        kv_start=text_start_positions,
        kv_end=text_end_positions,
        softmax_scale=scale,
        query_start_loc=text_metadata.query_start_loc,
        max_query_len=text_metadata.max_query_len,
    )
    if is_small_layer:
        assert summary_metadata is not None
        assert summary_cache_owner is not None
        summary_key_cache, summary_value_cache = split_ksa_kv_cache(
            summary_cache_owner.kv_cache,
            head_dim=text_cache_layer.head_dim,
        )
        old_summary_output, old_summary_lse = ksa_paged_source_attention(
            query=text_query,
            key_cache=summary_key_cache,
            value_cache=summary_value_cache,
            block_table=summary_metadata.block_table,
            row_to_request=text_requests,
            kv_start=torch.zeros_like(visible_summary_lens),
            kv_end=visible_summary_lens,
            softmax_scale=scale,
            query_start_loc=text_metadata.query_start_loc,
            max_query_len=text_metadata.max_query_len,
        )
        text_output = _merge_ksa_sources(
            text_output,
            text_lse,
            old_summary_output,
            old_summary_lse,
        )

    output = torch.empty_like(query)
    output.index_copy_(0, expanded_batch.text_row_indices, text_output)
    if expanded_batch.summary_row_indices.numel():
        summary_query = query.index_select(0, expanded_batch.summary_row_indices)
        summary_text_start = torch.where(
            summary_active,
            summary_positions - KSA_SUMMARY_CHUNK_SIZE + 1,
            torch.zeros_like(summary_positions),
        )
        summary_text_end = torch.where(
            summary_active,
            summary_positions + 1,
            torch.zeros_like(summary_positions),
        )
        summary_text_output, summary_text_lse = ksa_paged_source_attention(
            query=summary_query,
            key_cache=text_key_cache,
            value_cache=text_value_cache,
            block_table=text_metadata.block_table,
            row_to_request=summary_requests,
            kv_start=summary_text_start,
            kv_end=summary_text_end,
            softmax_scale=scale,
        )
        summary_self_output, summary_self_lse = _inline_ksa_attention(
            summary_query,
            summary_key,
            summary_value,
            scale=scale,
        )
        summary_output = _merge_ksa_sources(
            summary_text_output,
            summary_text_lse,
            summary_self_output,
            summary_self_lse,
        )
        output.index_copy_(0, expanded_batch.summary_row_indices, summary_output)
    return output


__all__ = [
    "KSATextCacheLayer",
    "get_ksa_attention_metadata",
    "get_ksa_summary_cache_spec",
    "paged_ksa_attention",
    "register_ksa_cache_owner",
    "register_ksa_cache_scales",
    "resolve_ksa_cache_dtype",
]
