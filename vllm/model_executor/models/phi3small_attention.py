# from vllm.attention import Attention, AttentionMetadata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from flash_attn import flash_attn_varlen_func

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)

from .phi3small_flash_blocksparse_attn_batch_inference import LocalStridedBlockSparseAttnInference
from .phi3small_paged_attn import LocalStridedBlockSparseAttnInferenceBT

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)


class BlockSparseFlashAttention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        local_blocks: int,
        vert_stride: int,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        max_seqlen: int = 8192,
        sparse_block_size: int = 64,
        layer_idx: int = 0,
        **kwargs
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.backend = BlocksparseFlashAttentionBackend
        impl_cls = self.backend.get_impl_cls()
        self.impl = impl_cls(local_blocks, vert_stride, num_heads, head_size, scale, num_kv_heads,
                             max_seqlen=max_seqlen, sparse_block_size=sparse_block_size, layer_idx=layer_idx)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        return self.impl.forward(query, key, value, kv_cache, attn_metadata,
                                 kv_scale)


class BlocksparseFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["BlocksparseFlashAttentionImpl"]:
        return BlocksparseFlashAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "BlocksparseFlashAttentionMetadata":
        return BlocksparseFlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class BlocksparseFlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]
    # The number of prompt tokens. Doesn't include padding.
    num_prompt_tokens: int
    # The number of generation tokens. Doesn't include padding.
    num_generation_tokens: int

    # NOTE(sang): Definition of context_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): context_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # slot_mapping
    # block_tables

    # blocksparse attributes:
    # TODO: metadata maker is from backend class, instead of backend instance.
    #       we can not set this info.
    #       Hack: is made optional, and will be set inside the real attention object initialization.

    # TODO: is block_tables padded? since it per sequence.

    # store the block_ids (relative sparse_block_size) of slot in block_tables
    sparse_block_ids: Optional[torch.Tensor]
    # Note that this is different from block_size in the block_table/block_manager.
    sparse_block_size: Optional[int]


    # the following two functions is only applable to blocksparse with local + vertical_stride patterns
    def check_if_remote(self, block_seq_id, head_id=0, n_heads=24, vert_stride=4):
        # does not check if the remote is in the current local or not
        head_sliding_step = max(1, int(vert_stride / n_heads))  # if vert_stride <= n_heads, rotating the heads
        return  (block_seq_id + head_id * head_sliding_step + 1 ) % vert_stride == 0

    def check_if_local(self, block_seq_id, block_size, total_seq_len, local_blocks=4):
        total_block = (total_seq_len + block_size - 1) // block_size
        # print(f'==>',  block_seq_id, block_size, total_seq_len, total_block)
        # return (block_seq_id + local_blocks) >= total_block
        return (total_block - local_blocks) <= block_seq_id < total_block



class BlocksparseFlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|

    Otherwise, the layout is as follows:
    |<------------------ num_generation_tokens (M) ----------------->|
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.
    """

    def __init__(
        self,
        local_blocks: int,
        vert_stride: int,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        max_seqlen: int = 8192,
        sparse_block_size: int = 64,
        alibi_slopes: Optional[List[float]] = None,
        layer_idx: int = 0,
        **kwargs, # for compatibiliy
    ) -> None:
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads


        if alibi_slopes is not None:
            assert ValueError('Alibi not support for blocksparse flash attention.')
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        active_head_range = (tp_rank * self.num_heads, (tp_rank + 1) * self.num_heads)

        total_num_heads = num_heads * tp_size
        self.bs_attn = LocalStridedBlockSparseAttnInference(total_num_heads, max_seqlen, local_blocks, vert_stride, sparse_block_size, active_head_range=active_head_range)
        self.bs_paged_attn = LocalStridedBlockSparseAttnInferenceBT(total_num_heads, max_seqlen, local_blocks, vert_stride, sparse_block_size, active_head_range=active_head_range)

        self.block_manager = None

    def set_block_manager(self, block_manager):
        self.block_manager = block_manager

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BlocksparseFlashAttentionMetadata,
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.

            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale)


            # torch.save({'q': query, 'k': key, 'v': value,
            # 'slot_mapping': attn_metadata.slot_mapping,
            # }, '/tmp/vllm_promt.pt')


        if attn_metadata.is_prompt:

            # Prompt run.
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.

            output = self.bs_attn(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.seq_start_loc,
                cu_seqlens_k=attn_metadata.seq_start_loc,
                sm_scale=self.scale
            )

        else:
            # Decoding run.

            # output = PagedAttention.forward_decode(
            #     query,
            #     key_cache,
            #     value_cache,
            #     attn_metadata.block_tables,
            #     attn_metadata.context_lens,
            #     attn_metadata.max_context_len,
            #     attn_metadata.kv_cache_dtype,
            #     self.num_kv_heads,
            #     self.scale,
            #     self.alibi_slopes,
            #     kv_scale,
            # )

            # print('\n\n2\n\n')
            # torch.save({'q': query, 'k': key_cache, 'v': value_cache,
            #             'output': output,
            #             'block_tables': attn_metadata.block_tables,
            #             'context_lens': attn_metadata.context_lens,
            #             'num_kv_heads':  self.num_kv_heads,
            #             'scale': self.scale,
            #             }, '/tmp/vllm.pt')

            # output = query
            output = self.bs_paged_attn(query, key_cache, value_cache,
                                        attn_metadata.block_tables,
                                        attn_metadata.context_lens,
                                        sm_scale=self.scale,
                                        kv_scale=kv_scale)

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)
