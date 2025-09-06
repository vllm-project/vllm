# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Optional

import torch
from transformers import CacheConfig

from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata, AttentionType)
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              subclass_attention_backend)


def _get_cross_slot_mapping(req_id: str, encoder_seq_len: int, requests: dict,
                            kv_cache_config) -> list[int]:
    """Get cross-attention slot mapping for a request."""
    from vllm.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    req_state = requests.get(req_id)
    if req_state is None:
        # During memory profiling or if request not found
        return [PAD_SLOT_ID] * encoder_seq_len

    # Find the KV cache group that uses CrossAttentionSpec
    cross_attn_group_idx = None
    for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        if isinstance(kv_cache_group.kv_cache_spec, CrossAttentionSpec):
            cross_attn_group_idx = i
            break

    if (cross_attn_group_idx is None
            or cross_attn_group_idx >= len(req_state.block_ids)):
        return [PAD_SLOT_ID] * encoder_seq_len

    # Get cross attention block IDs and calculate slot mapping
    cross_block_ids = req_state.block_ids[cross_attn_group_idx]
    block_size = kv_cache_config.kv_cache_groups[
        cross_attn_group_idx].kv_cache_spec.block_size

    slot_mapping = []
    for i in range(encoder_seq_len):
        block_number = cross_block_ids[i // block_size]
        block_offset = i % block_size
        slot = block_number * block_size + block_offset
        slot_mapping.append(slot)

    return slot_mapping


def _make_cross_attention_metadata(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """
    Transform common attention metadata for cross-attention.
    
    Cross-attention has specific requirements:
    - Non-causal (bidirectional) attention
    - Custom sequence lengths based on encoder length
    - Special slot mapping for encoder keys/values
    """
    # Create cross-attention specific slot mapping
    cross_slot_mapping = []
    for req_id in common_attn_metadata.scheduled_encoder_inputs:
        cross_slot_mapping.extend(
            _get_cross_slot_mapping(req_id,
                                    common_attn_metadata.max_encoder_len,
                                    common_attn_metadata.requests,
                                    common_attn_metadata.kv_cache_config))

    # Create tensors for cross-attention
    device = common_attn_metadata.device
    num_reqs = common_attn_metadata.num_reqs
    max_encoder_len = common_attn_metadata.max_encoder_len

    slot_mapping = torch.tensor(cross_slot_mapping,
                                dtype=torch.int64,
                                device=device)

    # Use encoder length for sequence lengths in cross-attention
    seq_lens_arg = torch.full(
        (num_reqs, ),
        max_encoder_len,
        dtype=torch.int32,
        device=device,
    )
    seq_lens_cpu_arg = torch.full(
        (num_reqs, ),
        max_encoder_len,
        dtype=torch.int32,
        device="cpu",
    )

    return CommonAttentionMetadata(
        query_start_loc=common_attn_metadata.query_start_loc,
        query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
        seq_lens=seq_lens_arg,
        seq_lens_cpu=seq_lens_cpu_arg,
        num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=common_attn_metadata.max_query_len,
        max_seq_len=max_encoder_len,
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=slot_mapping,
        causal=False,  # Cross-attention is non-causal
        logits_indices_padded=common_attn_metadata.logits_indices_padded,
        num_logits_indices=common_attn_metadata.num_logits_indices,
    )


@functools.lru_cache
def create_cross_attention_backend(
    underlying_attn_backend: AttentionBackend, ) -> type[AttentionBackend]:
    prefix = "CrossAttention_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class CrossAttentionBuilder(underlying_builder):  # type: ignore

        def build(self,
                  common_prefix_len: int,
                  common_attn_metadata: CommonAttentionMetadata,
                  fast_build: bool = False) -> AttentionMetadata:
            # Transform the metadata for cross-attention
            cross_attn_metadata = _make_cross_attention_metadata(
                common_attn_metadata)
            return super().build(common_prefix_len, cross_attn_metadata,
                                 fast_build)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=CrossAttentionBuilder)

    return attn_backend


class CrossAttention(Attention):
    """
    Cross-attention for encoder-decoder models.
    Handles attention between decoder queries and encoder keys/values.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 cache_config: Optional[CacheConfig] = None,
                 attn_type: Optional[str] = None,
                 **kwargs):
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        if envs.VLLM_USE_V1:
            underlying_attn_backend = get_attn_backend(head_size, dtype,
                                                       kv_cache_dtype,
                                                       block_size)

            attn_backend = create_cross_attention_backend(
                underlying_attn_backend)
        else:
            # in v0 cross attention is handled inside the backends
            attn_backend = None

        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_DECODER, (
                "CrossAttention only supports AttentionType.ENCODER_DECODER")

        super().__init__(num_heads=num_heads,
                         head_size=head_size,
                         scale=scale,
                         cache_config=cache_config,
                         attn_backend=attn_backend,
                         attn_type=AttentionType.ENCODER_DECODER,
                         **kwargs)
