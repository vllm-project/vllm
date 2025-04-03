# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import CommonAttentionState


@torch.library.custom_op("mylib::neuron_paged_attn", mutates_args=())
def neuron_paged_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    # value_cache: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    from vllm.attention.ops.nki_flash_attn import flash_attn_varlen_nkifunc
    N, _, n_kv_head, _, head_size = kv_cache.shape
    assert N == 2, f"invalid {kv_cache.shape=}"
    output_nki = flash_attn_varlen_nkifunc(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        # value_cache=value_cache,
        block_table=block_table,
        attn_mask=attn_mask,
        n_kv_head=n_kv_head,
        head_size=head_size,
        mixed_precision=True,
    )
    return output_nki


@neuron_paged_attn.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    # value_cache: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query.transpose(-2, -1))


class NeuronAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "NEURON"

    @staticmethod
    def get_impl_cls() -> type["NeuronAttentionBackendImpl"]:
        return NeuronAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> type["NeuronAttentionMetadata"]:
        return NeuronAttentionMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["NeuronAttentionMetadataBuilder"]:
        return NeuronAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)


@dataclass
class NeuronAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_active_blocks: int
    active_block_table: torch.Tensor
    attn_mask: torch.Tensor
    num_input_tokens: int = 0  # Number of tokens including padding.


class NeuronAttentionMetadataBuilder(
        AttentionMetadataBuilder[NeuronAttentionMetadata]):
    ...


class NeuronAttentionBackendImpl(AttentionImpl[NeuronAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[list[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from vllm.attention.ops.nki_flash_attn import reshape_and_cache

        num_tokens = query.shape[1]
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)
            slot_mapping = attn_metadata.slot_mapping
            reshape_and_cache(key, value, kv_cache, slot_mapping)
        else:
            # profiling run
            return query

        query = query.view(num_tokens, self.num_heads, self.head_size)
        query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        key = key.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        value = value.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        input_args = (
            query,
            key,
            value,
            kv_cache,
            # kv_cache,
            attn_metadata.active_block_table,
            attn_metadata.attn_mask,
        )
        output = neuron_paged_attn(*input_args)
        output = output.transpose(1,
                                  2).reshape(1, num_tokens,
                                             self.num_heads * self.head_size)
        return output
