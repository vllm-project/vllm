# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch
# Required to register custom ops.
import torch_xla.experimental.custom_kernel  # noqa: F401

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState

# These are the 2 tunable parameters of the paged attention Pallas kernel.
NUM_QUERIES_PER_BLOCK = 32
NUM_KV_PAGES_PER_BLOCK = 128


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> type["PallasMetadata"]:
        return PallasMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")


@dataclass
class PallasMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Used in the PallasAttentionBackendImpl
    # slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_seqs: int
    slot_slices: torch.Tensor | None


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError("Paged attention Pallas kernel does "
                             "not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if head_size % 128 != 0:
            raise NotImplementedError("Head size must be a multiple of 128.")
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "Attention logits soft-capping is not supported.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        tpu_version = torch_xla.tpu.version()
        if tpu_version < 4:
            raise NotImplementedError("TPU version must be 4 or higher.")
        # NOTE(chengjiyao): the TPU v4's vmem capacity is 16MB
        # TODO(chengjiyao): autotune NUM_QUERIES_PER_BLOCK,
        # NUM_KV_PAGES_PER_BLOCK and vmem_limit_bytes
        if tpu_version == 4:
            self.vmem_limit_bytes = 16 * 1024 * 1024
        else:
            self.vmem_limit_bytes = 64 * 1024 * 1024

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: PallasMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = ([num_blocks, block_size, num_kv_heads * head_size], 
                        [num_blocks, block_size, num_kv_heads * head_size])
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # For determine_available_memory case.
        if kv_cache[0].numel() == 0:
            if output is None:
                output = torch.ones_like(query)
            return output

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens, hidden_size = query.shape
        query = query.view(num_tokens, self.num_heads, self.head_size)
        # key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        # value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        key_cache, value_cache = kv_cache
        if kv_cache[0].numel() > 0:
            slot_slices = attn_metadata.slot_slices
            write_to_kv_cache(key, value, key_cache, value_cache, slot_slices)

        # TODO(xw32): once Jevin changes key_cache and value_cache in the kernel from 
        # [num_blocks, block_size, num_kv_heads, head_size] to [num_blocks, block_size, num_kv_heads * head_size]
        # update the torch_xla wheel and start using the updated kernel.
        # output = torch.ops.xla.ragged_paged_attention(
        #     query,
        #     key_cache,
        #     value_cache,
        #     attn_metadata.context_lens,
        #     attn_metadata.block_tables,
        #     attn_metadata.query_start_loc,
        #     attn_metadata.num_seqs,
        #     num_kv_pages_per_block=NUM_KV_PAGES_PER_BLOCK,
        #     num_queries_per_block=NUM_QUERIES_PER_BLOCK,
        #     vmem_limit_bytes=self.vmem_limit_bytes,
        #     use_kernel=True,
        #     sm_scale=self.scale)

        # return output.reshape(num_tokens, hidden_size)
        return query.reshape(num_tokens, hidden_size)


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_slices: torch.Tensor,
) -> None:
    """ Write the key and values to the KV cache.

    Args:
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads * head_size]
        k_cache = [num_blocks, block_size, num_kv_heads * head_size]
        v_cache = [num_blocks, block_size, num_kv_heads * head_size]

    """
    # change kv_cache layout from [num_blocks, block_size, num_kv_heads, head_size] to [num_blocks, block_size, num_kv_heads*head_size]
    # remove the reshape op on kv.
    # Create slices as as in https://github.com/pytorch/xla/blob/4584a2134259d1f9074ef690315de1d541211f52/torch_xla/experimental/pallas_kernels/kv_insertion.py#L83
    torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

    # key_cache = key_cache.flatten(0, 1)
    # value_cache = value_cache.flatten(0, 1)
    # slot_mapping = slot_mapping.flatten()
    # key_cache.index_copy_(0, slot_mapping, key)
    # value_cache.index_copy_(0, slot_mapping, value)
    
    # TODO(xw32): once the write_to_kv_cache kernel is ready, update the torch_xla wheel and
    # call it here
    # torch.ops.xla.kv_insertion(key, value, key_cache, value_cache, slot_slices)
