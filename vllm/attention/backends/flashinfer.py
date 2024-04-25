from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import flashinfer
import torch
from flash_attn import flash_attn_varlen_func

from vllm._C import cache_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)


class FlashInferBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "FlashInferMetadata":
        return FlashInferMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        raise NotImplementedError


@dataclass
class FlashInferMetadata(AttentionMetadataPerStage):

    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool

    use_cuda_graph: bool = False

    wrapper: Optional[flashinfer.BatchDecodeWithPagedKVCacheWrapper] = None

    # Metadata for prefill stage since we still use flash attention for prefill.
    seq_start_loc: Optional[torch.Tensor] = None
    max_prompt_len: Optional[int] = None


class FlashInferImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.alibi_slopes = alibi_slopes
        self.scale = scale
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache: Optional[torch.Tensor],
                attn_metadata: AttentionMetadata[FlashInferMetadata],
                kv_scale: float):
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            # Use the same reshape and cache kernel as flash attention.
            cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                attn_metadata.kv_cache_dtype,
            )

        if attn_metadata.prefill_metadata:
            output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.prefill_metadata.seq_start_loc,
                cu_seqlens_k=attn_metadata.prefill_metadata.seq_start_loc,
                max_seqlen_q=attn_metadata.prefill_metadata.max_prompt_len,
                max_seqlen_k=attn_metadata.prefill_metadata.max_prompt_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            assert attn_metadata.decode_metadata is not None
            query = query.contiguous(
            )  # Flashinfer requires query to be contiguous
            output = attn_metadata.decode_metadata.wrapper.forward(
                query,
                kv_cache,
                sm_scale=self.scale,
            )
        return output.view(num_tokens, hidden_size)
