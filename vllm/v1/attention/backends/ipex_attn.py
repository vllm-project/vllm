from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
import torch

from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.forward_context import get_forward_context


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
        return IPEXAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)


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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IPEXAttentionBackend,
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
        torch.ops.vllm.ipex_attn_chunked_prefill(
        # ipex_attn(
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

def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 1
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

@torch.library.custom_op("vllm::ipex_attn",
                         mutates_args=["output", "kv_cache"])
def ipex_attn_fake(output: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor, num_heads: int, head_size: int,
              num_kv_heads: int, kv_cache: torch.Tensor, kv_cache_dtype: str,
              k_scale: float, v_scale: float, scale: float,
              sliding_window: Optional[List[int]] = None,
              alibi_slopes: Optional[torch.Tensor] = None,
              logits_soft_cap: Optional[float] = None,) -> None:
    pass

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
    assert isinstance(current_metadata, IPEXAttentionMetadata)
    attn_metadata: IPEXAttentionMetadata = current_metadata
    num_actual_tokens = attn_metadata.num_actual_tokens
    
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)
    
    # Reshape the input keys and values and store them in the cache.
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    
    for i in range(num_actual_tokens):
        slot_idx = attn_metadata.slot_mapping[i]
        block_idx = slot_idx // key_cache.shape[1]
        block_offset = slot_idx % key_cache.shape[1]
        key_cache[block_idx, block_offset] = key[:num_actual_tokens][i]
        value_cache[block_idx, block_offset] = value[:num_actual_tokens][i]
    

    # ipex_ops.reshape_and_cache(
    #     key[:num_actual_tokens],
    #     value[:num_actual_tokens],
    #     key_cache,
    #     value_cache,
    #     attn_metadata.slot_mapping,
    #     kv_cache_dtype,
    #     k_scale,
    #     v_scale,
    # )
    
    attn_output = torch.ops.torch_ipex.chunked_prefill(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output,
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
    attn_output = attn_output.view(num_actual_tokens, -1)
    output[:num_actual_tokens].copy_(attn_output)


# @torch.library.custom_op("vllm::ipex_attn",
#                          mutates_args=["output", "kv_cache"])
def ipex_attn(output: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor, num_heads: int, head_size: int,
              num_kv_heads: int, kv_cache: torch.Tensor, kv_cache_dtype: str,
              k_scale: float, v_scale: float, scale: float,
              sliding_window: Optional[List[int]] = None,
              alibi_slopes: Optional[torch.Tensor] = None,
              logits_soft_cap: Optional[float] = None,) -> None:
    """
    Performs attention operation using Intel Extension for PyTorch (IPEX)
    optimized functions.

    Args:
        output (torch.Tensor): The output tensor to store the attention results.
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        num_heads (int): The number of attention heads.
        head_size (int): The size of each attention head.
        num_kv_heads (int): The number of key-value heads.
        kv_cache (tuple): A tuple containing the key and value cache tensors.
        kv_cache_dtype (torch.dtype): The data type of the key-value cache.
        k_scale (float): Scaling factor for the key tensor.
        v_scale (float): Scaling factor for the value tensor.
        scale (float): Scaling factor for the softmax operation.
        sliding_window (bool): Whether to use sliding window attention.
        alibi_slopes (torch.Tensor): The slopes for the ALiBi (Attention
        with Linear Biases) mechanism.
        logits_soft_cap (float): The soft cap for the logits.

    Returns:
        None
    """

    current_metadata = get_forward_context()
    if current_metadata is None:
        # Profiling run.
        return
    assert current_metadata is not None
    assert isinstance(current_metadata, IPEXAttentionMetadata)
    attn_metadata: IPEXAttentionMetadata = current_metadata
    num_actual_tokens = attn_metadata.num_actual_tokens

    # Reshape the query, key, and value tensors.
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)
    # Reshape the input keys and values and store them in the cache.
    
    key_cache, value_cache = split_kv_cache(kv_cache, num_kv_heads, head_size) 
    
    ipex_ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping.flatten(),
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    if attn_metadata.is_prefill_only:
        output = output.view(-1, num_heads, head_size)
        ipex_ops.varlen_attention(query,
                            key,
                            value,
                            output,
                            seqlen_q=attn_metadata.query_start_loc,
                            seqlen_k=attn_metadata.seq_start_loc,
                            max_seqlen_q=attn_metadata.max_query_len,
                            max_seqlen_k=attn_metadata.max_seq_len,
                            pdropout=0.0,
                            softmax_scale=scale,
                            zero_tensors=False,
                            is_causal=True,
                            return_softmax=False,
                            gen_=None,
                            logits_soft_cap=logits_soft_cap)

    elif attn_metadata.is_decode_only:
        block_size = value_cache.shape[3]
        actual_blocks_used = (attn_metadata.seq_len_q + block_size - 1) // block_size
        
        ipex_ops.paged_attention_v1(output, query, key_cache, value_cache, num_kv_heads, scale, attn_metadata.block_table, attn_metadata.seq_len_q,
                                    block_size, attn_metadata.max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale)
    else:
        raise ValueError("Invalid attention mode.")

    

@dataclass
class IPEXAttentionMetadata:
    is_prefill_only:bool
    is_decode_only:bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    seq_len_q: torch.Tensor  #??
    max_seq_len: int
    seq_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
