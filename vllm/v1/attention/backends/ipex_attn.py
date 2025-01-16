from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.attention.backends.ipex_attn import use_gqa_kernel
import os


class IPEXAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "IPEX_V1"

    @staticmethod
    def get_impl_cls() -> Type["IPEXAttentionImpl"]:
        return IPEXAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        # if block_size % 16 != 0:
            # raise ValueError("Block size must be a multiple of 16.")
        # This needs to be changed...
        # return (2, num_blocks, block_size, num_kv_heads, head_size)
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)



class IPEXAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = IPEXAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

    # TODO(gc): implement this logic...
    # Where to invoke this logic? what is attn_metadata?
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with IPEXAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "IPEXAttentionImpl")

        # NOTE(woosuk): IPEXAttention does not support FP8 KV cache.
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in IPEXAttention.")

        output = torch.empty_like(query)
        # torch.ops.vllm.ipex_attn_chunked_prefill(
        ipex_llm_chunked_prefill(
            output,
            query,
            key,
            value,
            self.num_heads,
            self.head_size,
            self.num_kv_heads,
            kv_cache,
            self.kv_cache_dtype,
            k_scale,
            v_scale,
            self.scale,
            self.sliding_window,
            self.alibi_slopes,
            self.logits_soft_cap,
        )
        return output.view(-1, self.num_heads * self.head_size)

def split_kv_cache_ipexllm(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # For GQA kernel, key_cache and value_cache shape should be [num_blocks, num_kv_heads, head_size, block_size]
    num_blocks = kv_cache.shape[1]

    key_cache = kv_cache[0]
    key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)
    value_cache = kv_cache[1]
    value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)
    return key_cache, value_cache

def split_kv_cache(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = 16 // kv_cache.element_size()
    num_blocks = kv_cache.shape[1]

    key_cache = kv_cache[0]
    key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                -1, x)

    value_cache = kv_cache[1]
    value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
    return key_cache, value_cache



@torch.library.custom_op("vllm::ipex_attn_fake",
                         mutates_args=["output", "kv_cache"])
def ipex_attn_fake(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    scale: float,
    sliding_window: Optional[List[int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: Optional[float] = None,
) -> None:
    pass

def ipex_llm_chunked_prefill(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    scale: float,
    sliding_window: Optional[List[int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: Optional[float] = None,
) -> None:
    context = get_forward_context()
    current_metadata = context.dynamic_forward_context
    if current_metadata is None:
        # Profiling run.
        return
    assert current_metadata is not None
    assert isinstance(current_metadata, FlashAttentionMetadata)
    attn_metadata: FlashAttentionMetadata = current_metadata
    num_actual_tokens = attn_metadata.num_actual_tokens

    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    using_gqa_kernel = use_gqa_kernel(num_heads, num_kv_heads, head_size, logits_soft_cap)


    if using_gqa_kernel:
        key_cache, value_cache = split_kv_cache_ipexllm(
                kv_cache, num_kv_heads, head_size)
        ipex_ops.reshape_and_cache_ipexllm(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        key_cache, value_cache = split_kv_cache(
            kv_cache, num_kv_heads, head_size)   
        ipex_ops.reshape_and_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    # Invoke chunked prefill method...
    import vllm._C.ops
    assert head_size == 128 or head_size == 64
    value = os.environ.get('USE_CONTEXT_V1')
    query_len = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    seq_len = attn_metadata.seq_start_loc[1:] - attn_metadata.seq_start_loc[:-1]
    context_len = seq_len - query_len
    if using_gqa_kernel:
        # if using_gqa_kernel, then only the v1 kernel can be used
        out = vllm._C.ops.context_attention_forward_v1(query[:num_actual_tokens], key_cache, value_cache, attn_metadata.block_table, attn_metadata.query_start_loc, seq_len, context_len, attn_metadata.max_seq_len, torch.amax(context_len).item())
    elif value is None:
        # Otherwise, by default use v2 attention forward kernel...
        out = vllm._C.ops.context_attention_forward_v2(query[:num_actual_tokens], key_cache, value_cache, prefill_meta.block_tables, prefill_meta.query_start_loc, prefill_meta.seq_lens_tensor, prefill_meta.context_lens, prefill_meta.max_seqlen, torch.amax(prefill_meta.context_lens).item())
    else:
        out = vllm._C.ops.context_attention_forward_v1(query, key_cache, value_cache, prefill_meta.block_tables, prefill_meta.query_start_loc, prefill_meta.seq_lens_tensor, prefill_meta.context_lens, prefill_meta.max_seqlen, torch.amax(prefill_meta.context_lens).item())
    
    # output[:num_actual_tokens] = out
    output[:num_actual_tokens] = out.view(out.shape[0], -1)




@torch.library.custom_op("vllm::ipex_attn_chunked_prefill",
                         mutates_args=["output", "kv_cache"])
def ipex_attn_chunked_prefill(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    scale: float,
    sliding_window: Optional[List[int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: Optional[float] = None,
) -> None:
    context = get_forward_context()
    current_metadata = context.dynamic_forward_context
    if current_metadata is None:
        # Profiling run.
        return

    assert current_metadata is not None
    assert isinstance(current_metadata, FlashAttentionMetadata)
    attn_metadata: FlashAttentionMetadata = current_metadata
    num_actual_tokens = attn_metadata.num_actual_tokens

    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    # Reshape the input keys and values and store them in the cache.
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]

    ipex_ops.reshape_and_cache_flash(
        key[:num_actual_tokens],
        value[:num_actual_tokens],
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    ipex_ops.chunked_prefill(
        query[:num_actual_tokens],
        key_cache,
        value_cache,
        output[:num_actual_tokens],
        attn_metadata.query_start_loc,
        attn_metadata.seq_start_loc,
        None,
        attn_metadata.block_table,
        alibi_slopes,
        attn_metadata.max_query_len,
        attn_metadata.max_seq_len,
        0.0,
        scale,
        False,
        True,
        False,
        None,
    )
