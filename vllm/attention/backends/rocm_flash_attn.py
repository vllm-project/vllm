"""Attention layer ROCm GPUs."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

_PARTITION_SIZE_ROCM = 512
_GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
_ON_NAVI = "gfx1" in _GPU_ARCH
_ON_MI250_MI300 = any(arch in _GPU_ARCH
                      for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"])


class ROCmFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASH"

    @staticmethod
    def get_impl_cls() -> Type["ROCmFlashAttentionImpl"]:
        return ROCmFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return ROCmFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["ROCmFlashAttentionMetadataBuilder"]:
        return ROCmFlashAttentionMetadataBuilder

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
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class ROCmFlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

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

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    _cached_prefill_metadata: Optional["ROCmFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["ROCmFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["ROCmFlashAttentionMetadata"]:
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

        self._cached_prefill_metadata = ROCmFlashAttentionMetadata(
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
    def decode_metadata(self) -> Optional["ROCmFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = ROCmFlashAttentionMetadata(
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

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """

        assert not turn_prefills_into_decodes, \
            ("Chunked prefill is not supported with rocm_flash_attn yet."
             "turn_prefills_into_decodes is a Multi-Step + Chunked-Prefill "
             "specific parameter.")

        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class ROCmFlashAttentionMetadataBuilder(
        CommonMetadataBuilder[ROCmFlashAttentionMetadata]):

    _metadata_cls = ROCmFlashAttentionMetadata


def _make_alibi_bias(alibi_slopes: torch.Tensor,
                     dtype: torch.dtype,
                     seq_lens: Optional[List[int]],
                     make_attn_mask: bool = True) -> List[torch.Tensor]:
    attn_biases = []
    if seq_lens:
        for seq_len in seq_lens:
            bias = torch.arange(seq_len, dtype=dtype)
            # NOTE(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(seq_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]

            num_heads = alibi_slopes.shape[0]
            bias = bias[None, :].repeat(
                (num_heads, 1, 1)).to(alibi_slopes.device)
            bias.mul_(alibi_slopes[:, None, None])
            if make_attn_mask:
                inf_mask = torch.empty(
                    (1, seq_len, seq_len),
                    dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1).to(
                        alibi_slopes.device)
                attn_biases.append((bias + inf_mask).to(dtype))
            else:
                attn_biases.append(bias.to(dtype))

    return attn_biases


class ROCmFlashAttentionImpl(AttentionImpl):
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

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens ----------->|	
    |<-prompt_0->|...|<-prompt_N-1->|<-generation_0->|...|<-generation_M-1->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
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
        if blocksparse_params is not None:
            raise ValueError(
                "ROCmFlashAttention does not support blocksparse attention.")
        if logits_soft_cap is not None:
            raise ValueError(
                "ROCmFlashAttention does not support attention logits soft "
                "capping.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

        self.use_naive_attn = False
        # NOTE: Allow for switching between Triton and CK. Defaulting to triton.
        self.use_triton_flash_attn = envs.VLLM_USE_TRITON_FLASH_ATTN
        if self.use_triton_flash_attn:
            from vllm.attention.ops.triton_flash_attention import (  # noqa: F401
                triton_attention)
            self.attn_func = triton_attention
            logger.debug("Using Triton FA in ROCmBackend")
            if self.sliding_window != (-1, -1):
                logger.warning("ROCm Triton FA does not currently support "
                               "sliding window attention. If using half "
                               "precision, please try using the ROCm CK "
                               "FA backend instead by setting the env var "
                               "`VLLM_USE_TRITON_FLASH_ATTN=0`")
        else:
            # if not using triton, navi3x/navi21/navi10 do not use flash-attn
            # either
            if not current_platform.has_device_capability(90):
                self.use_naive_attn = True
            else:
                try:
                    from flash_attn import flash_attn_varlen_func  # noqa: F401
                    self.attn_func = flash_attn_varlen_func
                    logger.debug("Using CK FA in ROCmBackend")
                except ModuleNotFoundError:
                    self.use_naive_attn = True

            if self.use_naive_attn:
                self.attn_func = _sdpa_attention
                logger.debug("Using naive attention in ROCmBackend")

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return (x[:, :,
                  None, :].expand(tokens, n_kv_heads, n_rep,
                                  head_dim).reshape(tokens, n_kv_heads * n_rep,
                                                    head_dim))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ROCmFlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
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
        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "ROCmFlashAttentionImpl")

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            assert prefill_meta.seq_lens is not None
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                # triton attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                attn_masks = None
                if self.use_triton_flash_attn:
                    if self.alibi_slopes is not None:
                        attn_masks = _make_alibi_bias(
                            self.alibi_slopes,
                            query.dtype,
                            attn_metadata.seq_lens,
                            make_attn_mask=False)  # type: ignore
                    out, _ = self.attn_func(
                        query,
                        key,
                        value,
                        None,
                        prefill_meta.seq_start_loc,
                        prefill_meta.seq_start_loc,
                        prefill_meta.max_prefill_seq_len,
                        prefill_meta.max_prefill_seq_len,
                        True,
                        self.scale,
                        attn_masks[0][None]
                        if attn_masks is not None else None,
                    )
                elif self.use_naive_attn:
                    if self.num_kv_heads != self.num_heads:
                        # Interleave for MQA workaround.
                        key = self.repeat_kv(key, self.num_queries_per_kv)
                        value = self.repeat_kv(value, self.num_queries_per_kv)
                    if self.alibi_slopes is not None:
                        attn_masks = _make_alibi_bias(
                            self.alibi_slopes,
                            query.dtype,
                            attn_metadata.seq_lens,
                            make_attn_mask=True)  # type: ignore
                    query = query.movedim(0, query.dim() - 2)
                    key = key.movedim(0, key.dim() - 2)
                    value = value.movedim(0, value.dim() - 2)
                    # sdpa math backend attention
                    out = self.attn_func(
                        query,
                        key,
                        value,
                        prefill_meta.seq_lens,
                        num_tokens,
                        self.num_heads,
                        self.head_size,
                        self.scale,
                        attn_masks,
                    )
                else:
                    out = self.attn_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=prefill_meta.seq_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_prefill_seq_len,
                        max_seqlen_k=prefill_meta.max_prefill_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                    )

                # common code for prefill
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window[0],
                    k_scale,
                    v_scale,
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Whether to use rocm custom paged attention or not
            num_seqs, num_heads, head_size = decode_query.shape
            block_size = value_cache.shape[3]
            gqa_ratio = num_heads // self.num_kv_heads
            use_custom = _use_rocm_custom_paged_attention(
                decode_query.dtype, head_size, block_size, gqa_ratio,
                decode_meta.max_decode_seq_len)
            if use_custom:
                max_seq_len = decode_meta.max_decode_seq_len
                max_num_partitions = (
                    (max_seq_len + _PARTITION_SIZE_ROCM - 1) //
                    _PARTITION_SIZE_ROCM)
                assert _PARTITION_SIZE_ROCM % block_size == 0
                tmp_output = torch.empty(
                    size=(num_seqs, num_heads, max_num_partitions, head_size),
                    dtype=output.dtype,
                    device=output.device,
                )
                exp_sums = torch.empty(
                    size=(num_seqs, num_heads, max_num_partitions),
                    dtype=torch.float32,
                    device=output.device,
                )
                max_logits = torch.empty_like(exp_sums)
                ops.paged_attention_rocm(
                    output[num_prefill_tokens:],
                    exp_sums,
                    max_logits,
                    tmp_output,
                    decode_query,
                    key_cache,
                    value_cache,
                    self.num_kv_heads,
                    self.scale,
                    decode_meta.block_tables,
                    decode_meta.seq_lens_tensor,
                    block_size,
                    max_seq_len,
                    self.alibi_slopes,
                    self.kv_cache_dtype,
                    k_scale,
                    v_scale,
                )
            else:
                output[num_prefill_tokens:] = PagedAttention.forward_decode(
                    decode_query,
                    key_cache,
                    value_cache,
                    decode_meta.block_tables,
                    decode_meta.seq_lens_tensor,
                    decode_meta.max_decode_seq_len,
                    self.kv_cache_dtype,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    k_scale,
                    v_scale,
                )

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: List[int],
    num_tokens: int,
    num_heads: int,
    head_size: int,
    scale: float,
    attn_masks: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    start = 0
    output = torch.empty((num_tokens, num_heads, head_size),
                         dtype=query.dtype,
                         device=query.device)

    for i, seq_len in enumerate(seq_lens):
        end = start + seq_len
        with torch.backends.cuda.sdp_kernel(enable_math=True,
                                            enable_flash=False,
                                            enable_mem_efficient=False):
            sub_out = torch.nn.functional.scaled_dot_product_attention(
                query[:, start:end, :],
                key[:, start:end, :],
                value[:, start:end, :],
                dropout_p=0.0,
                is_causal=attn_masks is None,
                attn_mask=attn_masks[i] if attn_masks else None,
                scale=scale).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end

    return output


def _use_rocm_custom_paged_attention(qtype: torch.dtype, head_size: int,
                                     block_size: int, gqa_ratio: int,
                                     max_seq_len: int) -> bool:
    # rocm custom page attention not support on navi (gfx1*)
    return (_ON_MI250_MI300 and not _ON_NAVI
            and (qtype == torch.half or qtype == torch.bfloat16)
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16) and max_seq_len <= 32768)
