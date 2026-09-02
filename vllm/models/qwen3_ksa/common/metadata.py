# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention metadata backend for Qwen3 KSA caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.models.qwen3_ksa.common.cache import build_ksa_summary_slot_mapping
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

KSA_SUMMARY_CHUNK_SIZE = 8


@dataclass
class KSAAttentionMetadata(AttentionMetadata):
    """Logical-row metadata for one KSA cache owner."""

    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_computed_tokens: torch.Tensor
    positions: torch.Tensor
    token_positions: torch.Tensor
    token_to_request: torch.Tensor
    boundary_mask: torch.Tensor
    chunk_indices: torch.Tensor
    text_start_positions: torch.Tensor
    visible_summary_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    manager_block_size: int
    states_per_block: int
    tokens_per_state: int
    is_cudagraph_capture: bool = False


def build_ksa_token_positions(
    *,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_actual_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build request-local absolute token indices for flattened query rows."""
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_computed_tokens = seq_lens - query_lens
    if torch.any(num_computed_tokens < 0):
        raise ValueError("KSA sequence lengths are smaller than query lengths")

    token_to_request = torch.repeat_interleave(
        torch.arange(
            query_lens.shape[0],
            device=query_start_loc.device,
            dtype=torch.int32,
        ),
        query_lens,
        output_size=num_actual_tokens,
    )
    query_starts = query_start_loc[:-1][token_to_request.to(torch.int64)]
    query_offsets = (
        torch.arange(num_actual_tokens, device=query_start_loc.device) - query_starts
    )
    token_positions = (
        num_computed_tokens[token_to_request.to(torch.int64)] + query_offsets
    ).to(torch.int64)
    return token_positions, token_to_request, num_computed_tokens


class KSAAttentionMetadataBuilder(AttentionMetadataBuilder[KSAAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if not isinstance(kv_cache_spec, AttentionSpec):
            raise TypeError("KSA requires an attention KV cache specification")
        tokens_per_state = kv_cache_spec.tokens_per_state
        if tokens_per_state not in (1, KSA_SUMMARY_CHUNK_SIZE):
            raise ValueError(
                "KSA cache tokens_per_state must be 1 or summary_chunk_size"
            )
        if kv_cache_spec.block_size % KSA_SUMMARY_CHUNK_SIZE != 0:
            raise ValueError("KSA cache block size must be divisible by 8")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KSAAttentionMetadata:
        del fast_build
        if common_prefix_len != 0:
            raise ValueError("KSA does not support prefix-cache cascade attention")
        if common_attn_metadata.positions is None:
            raise ValueError("KSA metadata requires logical RoPE positions")

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        if num_tokens == num_reqs and common_attn_metadata.max_query_len == 1:
            # Full CUDA Graph batches retain one physical token row for every
            # padded request, while query_start_loc uses zero-length padding
            # requests. A fixed request mapping keeps those rows addressable;
            # their cache slot remains -1.
            token_to_request = torch.arange(
                num_reqs,
                device=query_start_loc.device,
                dtype=torch.int32,
            )
            num_computed_tokens = torch.clamp_min(seq_lens - 1, 0)
            token_positions = num_computed_tokens.to(torch.int64)
        else:
            token_positions, token_to_request, num_computed_tokens = (
                build_ksa_token_positions(
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    num_actual_tokens=num_tokens,
                )
            )
        boundary_mask = (token_positions + 1).remainder(KSA_SUMMARY_CHUNK_SIZE) == 0

        spec = self.kv_cache_spec
        assert isinstance(spec, AttentionSpec)
        chunk_indices = torch.div(
            token_positions,
            KSA_SUMMARY_CHUNK_SIZE,
            rounding_mode="floor",
        )
        if isinstance(spec, SlidingWindowSpec):
            if spec.sliding_window % KSA_SUMMARY_CHUNK_SIZE != 0:
                raise ValueError("KSA sliding window must be divisible by 8")
            sliding_chunk_num = spec.sliding_window // KSA_SUMMARY_CHUNK_SIZE - 1
            text_start_positions = (
                torch.clamp_min(
                    chunk_indices - sliding_chunk_num,
                    0,
                )
                * KSA_SUMMARY_CHUNK_SIZE
            )
            visible_summary_lens = torch.clamp_min(
                chunk_indices - sliding_chunk_num,
                0,
            )
        else:
            text_start_positions = torch.zeros_like(token_positions)
            visible_summary_lens = torch.zeros_like(token_positions)
        tokens_per_state = int(spec.tokens_per_state)
        if tokens_per_state == KSA_SUMMARY_CHUNK_SIZE:
            slot_mapping = build_ksa_summary_slot_mapping(
                token_positions=token_positions,
                token_to_request=token_to_request,
                boundary_mask=boundary_mask,
                block_table=block_table,
                manager_block_size=spec.block_size,
                states_per_block=spec.num_states,
                summary_chunk_size=KSA_SUMMARY_CHUNK_SIZE,
            )
        else:
            slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]

        return KSAAttentionMetadata(
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_computed_tokens=num_computed_tokens,
            positions=common_attn_metadata.positions[:num_tokens],
            token_positions=token_positions,
            token_to_request=token_to_request,
            boundary_mask=boundary_mask,
            chunk_indices=chunk_indices,
            text_start_positions=text_start_positions,
            visible_summary_lens=visible_summary_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            manager_block_size=spec.block_size,
            states_per_block=spec.num_states,
            tokens_per_state=tokens_per_state,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> KSAAttentionMetadata:
        """Bind fixed-shape decode metadata to runner-owned graph buffers."""
        if common_attn_metadata.positions is None:
            raise ValueError("KSA CUDA Graph metadata requires logical positions")
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        if num_reqs != num_tokens or common_attn_metadata.max_query_len != 1:
            raise ValueError(
                "KSA CUDA Graph capture supports uniform single-token decode only"
            )

        spec = self.kv_cache_spec
        assert isinstance(spec, AttentionSpec)
        positions = common_attn_metadata.positions[:num_tokens]
        request_indices = torch.arange(
            num_reqs,
            device=positions.device,
            dtype=torch.int32,
        )
        zeros = torch.zeros_like(positions)
        boundary_mask = torch.zeros_like(positions, dtype=torch.bool)
        return KSAAttentionMetadata(
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=1,
            query_start_loc=common_attn_metadata.query_start_loc[: num_reqs + 1],
            seq_lens=common_attn_metadata.seq_lens[:num_reqs],
            # Derived values are recomputed inside the captured model. These
            # placeholders only preserve the fixed metadata structure.
            num_computed_tokens=zeros,
            positions=positions,
            token_positions=positions,
            token_to_request=request_indices,
            boundary_mask=boundary_mask,
            chunk_indices=zeros,
            text_start_positions=zeros,
            visible_summary_lens=zeros,
            block_table=common_attn_metadata.block_table_tensor[:num_reqs],
            slot_mapping=common_attn_metadata.slot_mapping[:num_tokens],
            manager_block_size=spec.block_size,
            states_per_block=spec.num_states,
            tokens_per_state=int(spec.tokens_per_state),
            is_cudagraph_capture=True,
        )


class KSAAttentionBackend(AttentionBackend):
    """Metadata-only backend used by model-owned KSA attention."""

    @staticmethod
    def get_name() -> str:
        return "KSA"

    @staticmethod
    def get_impl_cls() -> type[Any]:
        raise NotImplementedError("KSA attention is implemented by its model layer")

    @staticmethod
    def get_builder_cls() -> type[KSAAttentionMetadataBuilder]:
        return KSAAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(KSA_SUMMARY_CHUNK_SIZE)]

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True


__all__ = [
    "KSA_SUMMARY_CHUNK_SIZE",
    "KSAAttentionBackend",
    "KSAAttentionMetadata",
    "KSAAttentionMetadataBuilder",
    "build_ksa_token_positions",
]
