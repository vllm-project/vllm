import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

IS_COMPUTE_8_OR_ABOVE = current_platform.has_device_capability(80)


@dataclass
class BlocksparseParams:
    max_seqlen: int

    # Num q heads per tensor-parallel rank/partition
    num_heads: int  # per TP partition
    # Num kv heads per tensor-parallel rank/partition
    num_kv_heads: int

    # block size used for blocksparse attention.
    # This is the block_size used in `local_blocks`, `vert_stride`.
    block_size: int

    # Number of blocks for local attention, i.e., number of
    # local attended tokens / `sparse_block_size`
    local_blocks: int

    # Attend to one block per every `vert_stride` blocks.
    # Controlling the sparsity
    vert_stride: int
    """
    If to use the same vertical stride offset for all heads, 
    i.e., attend to the same block of tokens on all heads.
    By default, it is False, i.e., attention on the non-local 
    blocks depends on the `head_idx`, that is on
    blocks satisfying 
    `(block_idx + head_idx * head_sliding_step + 1) % vert_stride == 0`
    where `head_sliding_step=max(1, int(vert_stride / num_total_heads))`,
            `block_idx = position_id // sparse_block_size`.
    See `..ops.blocksparse_attention.utils:get_sparse_attn_mask`
    for more detail.
    """
    homo_head: bool = False

    # If within a group, the kv offsets that each q attends is the same or no.
    homo_head_group: bool = False

    # Decided by homo_head and homo_head group
    head_sliding_step: int = field(init=False)

    # range of q heads to for a TP rank
    active_head_range: Tuple = field(init=False)

    def __post_init__(self):
        assert self.block_size > 0
        assert self.local_blocks >= 0
        assert self.vert_stride >= 1
        assert self.num_heads % self.num_kv_heads == 0

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        total_heads = tp_size * self.num_heads
        total_kv_heads = tp_size * self.num_kv_heads

        from ..ops.blocksparse_attention.utils import get_head_sliding_step
        if self.homo_head:
            self.head_sliding_step = 0
        elif self.homo_head_group:
            head_sliding_step = get_head_sliding_step(total_kv_heads,
                                                      self.vert_stride)
            # negative indicates sliding along kv heads, i.e., homo q group
            self.head_sliding_step = -head_sliding_step
        else:
            self.head_sliding_step = get_head_sliding_step(
                total_heads, self.vert_stride)

        self.active_head_range = (
            tp_rank * self.num_heads,
            (tp_rank + 1) * self.num_heads,
        )


class BlocksparseFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["BlocksparseFlashAttentionImpl"]:
        return BlocksparseFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return BlocksparseFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["BlocksparseFlashAttentionMetadataBuilder"]:
        return BlocksparseFlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

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
class BlocksparseFlashAttentionMetadata(AttentionMetadata):
    """A copy of Metadata for FlashAttentionBackend,
    to avoid having to install flash_attn.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # Number of query tokens for each request in the batch.
    # Currently, we require that all requests have the same number of query
    # tokens during the decoding phase. When speculavie decoding is enabled,
    # decode_query_len might be greater than 1. In all other cases, it is 1.
    decode_query_len: Optional[int] = None

    _cached_prefill_metadata: Optional[
        "BlocksparseFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional[
        "BlocksparseFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(
            self) -> Optional["BlocksparseFlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = BlocksparseFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["BlocksparseFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = BlocksparseFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata


class BlocksparseFlashAttentionMetadataBuilder(
        CommonMetadataBuilder[BlocksparseFlashAttentionMetadata]):

    _metadata_cls = BlocksparseFlashAttentionMetadata


def transpose_and_pad(x, cu_seqlens, maxlen, head_repeats=1):
    """
    :param x: (total_tokens, n_heads, head_size)
    :return: (batch, n_heads, length, head_size)
    """
    x_padded = x.new_empty(
        len(cu_seqlens) - 1, x.size(1), head_repeats, maxlen, x.size(2))
    cu_seqlens = cu_seqlens.cpu()
    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        x_padded[i, :, :, :e - s].copy_(x[s:e].transpose(0, 1).unsqueeze(1))
    return x_padded.flatten(1, 2)


def transpose_and_unpad(x_padded, cu_seqlens):
    """
    :param x_padded: (batch, n_heads, length, head_size)
    :return: (total_tokens, n_heads, head_size)
    """
    cu_seqlens = cu_seqlens.cpu()
    total_n_tokens = cu_seqlens[-1]
    x = x_padded.new_empty(total_n_tokens, x_padded.size(1), x_padded.size(3))
    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        x[s:e].copy_(x_padded[i, :, :e - s].transpose(0, 1))
    return x


def get_attn_pattern(n_heads, max_seqlen, block_size, local_blocks,
                     vert_stride, homo_head, use_spda, active_head_range,
                     dtype, device):
    from ..ops.blocksparse_attention.utils import get_sparse_attn_mask
    sparse_layout, sparse_pattern, dense_attn_mask = get_sparse_attn_mask(
        n_heads,
        max_seqlen,
        max_seqlen,
        dtype,
        device,
        block_size=block_size,
        local_blocks=local_blocks,
        vert_stride=vert_stride,
        homo_head=homo_head,
        return_dense=use_spda,
        dense_mask_type="bias",
    )
    if (not homo_head) and (active_head_range is not None):
        assert isinstance(active_head_range, tuple)
        assert (len(active_head_range) == 2)
        h_start, h_end = active_head_range
        sparse_layout = tuple(x[h_start:h_end] for x in sparse_layout)
        if use_spda:
            dense_attn_mask = dense_attn_mask[h_start:h_end]
    return sparse_layout, sparse_pattern, dense_attn_mask


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
        assert blocksparse_params is not None
        assert alibi_slopes is None, ValueError(
            "Alibi not support for blocksparse flash attention.")
        assert sliding_window is None, ValueError(
            "sliding_window is invalid for blocksparse attention.")
        assert logits_soft_cap is None, ValueError(
            "logits_soft_cap is invalid for blocksparse attention.")

        if "num_heads" not in blocksparse_params:
            blocksparse_params["num_heads"] = num_heads
        if "num_kv_heads" not in blocksparse_params:
            blocksparse_params["num_kv_heads"] = num_kv_heads or num_heads
        self.blocksparse_params = BlocksparseParams(**blocksparse_params)
        self.kv_cache_dtype = kv_cache_dtype

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.alibi_slopes = alibi_slopes
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.local_blocks = self.blocksparse_params.local_blocks
        self.vert_stride = self.blocksparse_params.vert_stride
        self.sparse_block_size = self.blocksparse_params.block_size
        self.head_sliding_step = self.blocksparse_params.head_sliding_step

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        total_num_heads = num_heads * self.tp_size

        n_heads = total_num_heads
        max_seqlen = self.blocksparse_params.max_seqlen
        local_blocks = self.blocksparse_params.local_blocks
        vert_stride = self.blocksparse_params.vert_stride
        block_size = self.blocksparse_params.block_size
        device = None
        dtype = None
        homo_head = self.blocksparse_params.homo_head
        active_head_range = self.blocksparse_params.active_head_range
        q_block_size = None
        use_spda = None

        if use_spda is None:
            use_spda = current_platform.is_rocm() or \
                        current_platform.is_cpu() or not \
                            IS_COMPUTE_8_OR_ABOVE
        device = device or (torch.cuda.current_device()
                            if current_platform.is_cuda_alike() else "cpu")
        device = torch.device(device)
        # NOTE: vllm CPU backend support BF16 instead of FP16.
        dtype = dtype or (torch.bfloat16 if IS_COMPUTE_8_OR_ABOVE
                          or device.type == "cpu" else torch.half)

        self.n_heads = n_heads
        self.max_seqlen = max_seqlen
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.use_spda = use_spda
        self.dtype = dtype
        self.device = device
        self.block_size = block_size
        self.q_block_size = q_block_size
        self.homo_head = homo_head
        self.active_head_range = active_head_range

        from ..ops.blocksparse_attention.utils import get_head_sliding_step
        self.head_sliding_step = get_head_sliding_step(n_heads, vert_stride,
                                                       homo_head)

        sparse_layout, sparse_pattern, self.dense_attn_mask = get_attn_pattern(
            self.n_heads, self.max_seqlen, self.block_size, self.local_blocks,
            self.vert_stride, self.homo_head, self.use_spda,
            self.active_head_range, dtype, device)

        if q_block_size is not None and q_block_size != block_size:
            if q_block_size > block_size:
                assert q_block_size % block_size == 0
                blocks_to_merge = q_block_size // block_size
                shape = sparse_pattern.shape
                sparse_pattern = sparse_pattern.view(shape[0], -1,
                                                     blocks_to_merge,
                                                     shape[-1])
                sparse_pattern = sparse_pattern.sum(2)
                from ..ops.blocksparse_attention.utils import dense_to_crow_col
                sparse_layout = dense_to_crow_col(sparse_pattern)
            else:
                raise ValueError(
                    "Does not support smaller q_block_size. It will be slower."
                )

        self.sparse_layout = sparse_layout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BlocksparseFlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: int = AttentionType.DECODER,
    ) -> torch.Tensor:

        output = torch.ops.vllm.unified_blocksparse_attention(
            query=query,
            key=key,
            value=value,
            n_heads=self.n_heads,
            num_heads=self.num_heads,
            head_size=self.head_size,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
            q_block_size=self.q_block_size,
            kv_cache=kv_cache,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            max_seqlen=self.blocksparse_params.max_seqlen,
            tp_rank=self.tp_rank,
            blocksparse_local_blocks=self.local_blocks,
            blocksparse_vert_stride=self.vert_stride,
            blocksparse_block_size=self.sparse_block_size,
            blocksparse_head_sliding_step=self.head_sliding_step,
            homo_head=self.homo_head,
            active_head_range=list(self.active_head_range),
            use_spda=self.use_spda,
            dense_attn_mask=self.dense_attn_mask,
            sparse_layout=self.sparse_layout,
            attn_type=attn_type,
        )

        return output


@torch.library.custom_op("vllm::unified_blocksparse_attention",
                         mutates_args=["kv_cache"])
def unified_blocksparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,  # total number of heads
    num_heads: int,  # heads in this rank
    head_size: int,
    num_kv_heads: int,
    block_size: int,
    q_block_size: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen: int,
    tp_rank: int,
    blocksparse_local_blocks: int,
    blocksparse_vert_stride: int,
    blocksparse_block_size: int,
    blocksparse_head_sliding_step: int,
    homo_head: int,
    active_head_range: List[int],
    use_spda: bool,
    dense_attn_mask: torch.Tensor,
    sparse_layout: Tuple[torch.Tensor, torch.Tensor],
    attn_type: int = AttentionType.DECODER,
) -> torch.Tensor:
    """Forward pass with FlashAttention and PagedAttention.

    Args:
        query: shape = [num_tokens, num_heads * head_size]
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads * head_size]
        kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            NOTE: kv_cache will be an empty tensor with shape [0]
            for profiling run.
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    """

    current_metadata = get_forward_context()
    assert current_metadata is not None
    assert isinstance(current_metadata, BlocksparseFlashAttentionMetadata)
    attn_metadata: BlocksparseFlashAttentionMetadata = current_metadata

    if attn_type != AttentionType.DECODER:
        raise NotImplementedError("Encoder self-attention and "
                                  "encoder/decoder cross-attention "
                                  "are not implemented for "
                                  "BlocksparseFlashAttentionImpl")

    num_tokens, hidden_size = query.shape
    # Reshape the query, key, and value tensors.
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    if kv_cache.numel() > 0:
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, num_kv_heads, head_size)

        # Reshape the input keys and values and store them in the cache.
        # If kv_cache is not provided, the new key and value tensors are
        # not cached. This happens during the initial memory profiling run.

        PagedAttention.write_to_paged_cache(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    if prefill_meta := attn_metadata.prefill_metadata:

        # Prompt run.
        # normal attention
        # When block_tables are not filled, it means q and k are the
        # prompt, and they have the same length.

        assert kv_cache.numel() == 0 \
                or prefill_meta.block_tables is None \
                or prefill_meta.block_tables.numel() == 0, \
            "Does not support prefix-enabled attention."
        """Dispatch to `varlen_attn` (Ampere or newer) or 
        `self.spda`(cpu, Volta, Turing or older)based on 
        the type of device used and cuda compute capability.

        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
                Support grouped attention, with `q[:, i*r:(i*r + r)]`
                is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,), indicating segment of samples,
                    e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
                    Default None: same as cu_seqlens_k for prefilling or
                    [0, 1, .., batch_size] for decoding.
                    The only case you need to specify 
                    is when q is a mix of prefilling 
                    and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        """
        q = query
        k = key
        v = value
        cu_seqlens_q = prefill_meta.seq_start_loc
        cu_seqlens_k = prefill_meta.seq_start_loc
        sm_scale = softmax_scale

        assert k.dim() == 3
        if use_spda:
            """For CPU, V100 or other older GPUs.
            NOTE: torch SPDA supports nested tensor, 
            but seems extremely slow. Choose to pad instead.
            """
            assert (cu_seqlens_q is None
                    or (cu_seqlens_q == cu_seqlens_k).all()
                    ), "Can only handle prompt with SPDA."
            assert q.size(0) == k.size(0), "can only handle prompt with SPDA."

            assert q.size(1) % k.size(1) == 0
            q_k_ratio = q.size(1) // k.size(1)
            sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))
            assert cu_seqlens_k is not None
            cu_seqlens = cu_seqlens_k.cpu()
            maxlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

            if (dense_attn_mask.dtype != q.dtype
                    or dense_attn_mask.device != q.device):
                _, _, dense_attn_mask = get_attn_pattern(
                    n_heads, max_seqlen, block_size, blocksparse_local_blocks,
                    blocksparse_vert_stride, homo_head, use_spda,
                    tuple(active_head_range), q.dtype, q.device)
            attn_mask = dense_attn_mask[None, :, :maxlen, :maxlen]

            q2 = transpose_and_pad(q, cu_seqlens, maxlen, 1)
            k2, v2 = [
                transpose_and_pad(x, cu_seqlens, maxlen, q_k_ratio)
                for x in [k, v]
            ]
            spda_output = torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v2, attn_mask=attn_mask, scale=sm_scale)
            return transpose_and_unpad(spda_output, cu_seqlens)
        """
        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
        Support grouped attention, with `q[:, i*r:(i*r + r)]`
        is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,), 
        indicating segment of samples, 
        e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
        Default None: same as cu_seqlens_k for prefilling or
        [0, 1, .., batch_size] for decoding.
        The only case you need to specify is when q is a mix of 
        prefilling and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        """
        assert (
            IS_COMPUTE_8_OR_ABOVE
        ), "Requires compute capability of 8 or above (Ampere or newer) to use \
            Triton kernel."

        sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))

        from ..ops.blocksparse_attention.blocksparse_attention_kernel import (  # noqa
            blocksparse_flash_attn_varlen_fwd)

        return blocksparse_flash_attn_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_k,
            cu_seqlens_q,
            sm_scale,
            sparse_layout,
            block_size=block_size,
            q_block_size=q_block_size,
            max_seqlen=max_seqlen,
        )

    if decode_meta := attn_metadata.decode_metadata:
        # Decoding run.
        output = PagedAttention.forward_decode(
            query,
            key_cache,
            value_cache,
            decode_meta.block_tables,
            decode_meta.seq_lens_tensor,
            max_seqlen,
            kv_cache_dtype,
            num_kv_heads,
            softmax_scale,
            alibi_slopes,
            k_scale,
            v_scale,
            tp_rank=tp_rank,
            blocksparse_local_blocks=blocksparse_local_blocks,
            blocksparse_vert_stride=blocksparse_vert_stride,
            blocksparse_block_size=blocksparse_block_size,
            blocksparse_head_sliding_step=blocksparse_head_sliding_step,
        )

    # Reshape the output tensor.
    return output.view(num_tokens, hidden_size)


@unified_blocksparse_attention.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,  # total number of heads
    num_heads: int,  # heads in this rank
    head_size: int,
    num_kv_heads: int,
    block_size: int,
    q_block_size: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen: int,
    tp_rank: int,
    blocksparse_local_blocks: int,
    blocksparse_vert_stride: int,
    blocksparse_block_size: int,
    blocksparse_head_sliding_step: int,
    homo_head: int,
    active_head_range: List[int],
    use_spda: bool,
    dense_attn_mask: torch.Tensor,
    sparse_layout: Tuple[torch.Tensor, torch.Tensor],
    attn_type: int = AttentionType.DECODER,
) -> torch.Tensor:
    return torch.empty_like(query)
