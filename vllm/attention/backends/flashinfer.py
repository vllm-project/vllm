from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import flashinfer
import torch
from flash_attn import flash_attn_varlen_func

from vllm import _custom_ops as ops
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

    is_prompt: bool

    use_cuda_graph: bool = False

    decode_wrapper: Optional[
        flashinfer.BatchDecodeWithPagedKVCacheWrapper] = None

    # Metadata for the prefill stage since we still
    # use flash attention for prefill.
    seq_start_loc: Optional[torch.Tensor] = None
    max_prompt_len: Optional[int] = None

    # Metadata for the decode stage
    # Workspace buffer required by the kernel, the buffer should not
    # be allocated/deacollated by the FalshInfermetadata
    workspace_buffer: Optional[torch.Tensor] = None
    #  The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    # The number of query/output heads
    num_qo_heads: Optional[int] = None
    # The number of key/value heads
    num_kv_heads: Optional[int] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    # Block size of vllm
    page_size: Optional[int] = None
    # The data type of the paged kv cache
    data_type: torch.dtype = None

    def __post_init__(self):
        if not self.is_prompt:
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD")
            self.decode_wrapper.begin_forward(
                self.paged_kv_indptr,
                self.paged_kv_indices,
                self.paged_kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                # Disable flashinfer's pos encoding and use vllm's rope.
                pos_encoding_mode="NONE",
                data_type=self.data_type)

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        # We need to skip the decode_wrapper field since it cannot be
        # broadcasted with nccl when TP is enabled.
        skip_fields.add('decode_wrapper')
        return super().asdict_zerocopy(skip_fields)


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
        if sliding_window is not None:
            raise ValueError("Sliding window is not supported in FlashInfer.")
        self.sliding_window = (-1, -1)
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

        if attn_metadata.num_prefill_tokens > 0:
            assert attn_metadata.num_decode_tokens == 0, (
                "chunked prefill is not supported with flash infer yet")
        if attn_metadata.num_decode_tokens > 0:
            assert attn_metadata.num_prefill_tokens == 0, (
                "chunked prefill is not supported with flash infer yet")

        if kv_cache is not None:
            # Use the same reshape and cache kernel as flash attention.
            ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                attn_metadata.kv_cache_dtype,
            )

        if prefill_metadata := attn_metadata.prefill_metadata:
            output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_metadata.seq_start_loc,
                cu_seqlens_k=prefill_metadata.seq_start_loc,
                max_seqlen_q=prefill_metadata.max_prompt_len,
                max_seqlen_k=prefill_metadata.max_prompt_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            assert attn_metadata.decode_metadata is not None
            assert attn_metadata.decode_metadata.decode_wrapper is not None
            query = query.contiguous(
            )  # Flashinfer requires query to be contiguous
            output = attn_metadata.decode_metadata.decode_wrapper.forward(
                query,
                kv_cache,
                sm_scale=self.scale,
            )
        return output.view(num_tokens, hidden_size)
