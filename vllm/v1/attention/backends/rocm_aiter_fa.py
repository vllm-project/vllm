# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with AiterFlashAttention."""

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
    MultipleOf,
)
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_prefills_and_chunk,
)

from vllm.v1.kv_cache_interface import AttentionSpec

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024

if current_platform.is_rocm():
    import aiter
    from vllm.triton_utils import tl, triton
    from vllm.utils.torch_utils import direct_register_custom_op
    from aiter.ops.triton.utils.device_info import get_num_sms


    def block_size(x, head_dim):
        return min(65536 // x.element_size(), triton.next_power_of_2(head_dim))

    def num_programs(head_dim):
        return min(head_dim, get_num_sms())

    @triton.jit
    def cp_mha_gather_cache_kernel(
            key_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
            value_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
            key_ptr,  # [num_tokens, num_heads, head_size]
            value_ptr,  # [num_tokens, num_heads, head_size]
            block_table_ptr,  # [num_batches, max_block_num]
            cu_seqlens_kv_ptr,  # [num_batches + 1]
            token_to_batch_ptr,  # [max_cum_tokens] 
            seq_start_ptr,  # [num_batches]
            k_scale_ptr,
            v_scale_ptr,
            num_heads,
            head_size,
            x,
            max_block_num,
            num_tokens,
            DEQUANT: tl.constexpr,
            PAGE_SIZE: tl.constexpr,
            CACHE_FORMAT: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            NUM_PRGMS: tl.constexpr):
        bid = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        if DEQUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)

        for token_id in tl.range(bid, num_tokens, NUM_PRGMS):
            key_ptr_offset = key_ptr + token_id * head_size * num_heads
            value_ptr_offset = value_ptr + token_id * head_size * num_heads
            batch_idx = tl.load(token_to_batch_ptr + token_id)
            batch_start = tl.load(seq_start_ptr + batch_idx)
            token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
            batch_offset = token_id - token_start + batch_start
            block_offset = batch_offset // PAGE_SIZE
            block_id = tl.load(block_table_ptr + max_block_num * batch_idx +
                               block_offset)
            slot_id = batch_offset % PAGE_SIZE

            if CACHE_FORMAT == "NHD":
                # for kv cache layout as
                # K: [num_blocks, page_size, num_head, head_dim]
                # V: [num_blocks, page_size, num_head, head_dim]
                key_cache_ptr_offset = key_cache_ptr + block_id * num_heads * \
                    head_size * PAGE_SIZE + slot_id * num_heads * head_size
                value_cache_ptr_offset = value_cache_ptr + block_id * \
                    num_heads * head_size * PAGE_SIZE + slot_id * \
                        num_heads * head_size

                for i in tl.range(0, head_size * num_heads, BLOCK_SIZE):
                    mask = (col_offsets + i) < head_size * num_heads
                    k_reg = tl.load(key_cache_ptr_offset + col_offsets + i,
                                    mask=mask)
                    v_reg = tl.load(value_cache_ptr_offset + col_offsets + i,
                                    mask=mask)
                    if DEQUANT:
                        k_dtype = k_reg.dtype
                        v_dtype = v_reg.dtype
                        k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)
                        v_reg = (v_reg.to(tl.float32) * v_scale).to(v_dtype)
                    tl.store(key_ptr_offset + col_offsets + i,
                             k_reg,
                             mask=mask)
                    tl.store(value_ptr_offset + col_offsets + i,
                             v_reg,
                             mask=mask)

    def cp_mha_gather_cache(key_cache: torch.Tensor, value_cache: torch.Tensor,
                            key: torch.Tensor, value: torch.Tensor,
                            block_tables: torch.Tensor, k_scales: float,
                            v_scales: float, cu_seqlens_kv: torch.Tensor,
                            token_to_batch: torch.Tensor,
                            seq_starts: torch.Tensor, dequant: bool,
                            kv_cache_layout: str, total_tokens: int):
        assert kv_cache_layout in [
            "v0", "NHD", "HND"
        ], "kv_cache_layout only support v0, NHD, HND"
        head_dim = key.shape[2]
        x = 0
        assert dequant is True, "Currently, we only support "\
        "gather cache with dequant"
        # For k cache layout: [num_blocks, num_heads, page_size, head_dim]
        assert kv_cache_layout == "NHD", "ROCM_AITER_FA_BACKEND Only "\
        "support NHD kv cache layout for now"
        assert head_dim == key_cache.shape[
            3], "We assume your kv cache layout is [num_blocks, "\
            "page_size, num_heads, head_dim], but got otherwise"
        page_size = key_cache.shape[1]
        num_heads = key_cache.shape[2]

        NUM_PRGMS = num_programs(total_tokens)
        BLOCK_SIZE = block_size(key_cache, head_dim)
        grid = lambda meta: (NUM_PRGMS, )
        cp_mha_gather_cache_kernel[grid](key_cache,
                                         value_cache,
                                         key,
                                         value,
                                         block_tables,
                                         cu_seqlens_kv,
                                         token_to_batch,
                                         seq_starts,
                                         k_scales,
                                         v_scales,
                                         num_heads,
                                         head_dim,
                                         x,
                                         block_tables.size(1),
                                         total_tokens,
                                         DEQUANT=dequant,
                                         PAGE_SIZE=page_size,
                                         CACHE_FORMAT=kv_cache_layout,
                                         BLOCK_SIZE=BLOCK_SIZE,
                                         NUM_PRGMS=NUM_PRGMS)


logger = init_logger(__name__)


@dataclass
class AiterFlashAttentionDecodeMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


@dataclass
class AiterFlashAttentionPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


@dataclass
class AiterChunkContextMetadata:
    workspace: torch.Tensor
    cu_seq_lens_chunk: torch.Tensor
    chunk_starts: torch.Tensor
    token_to_batch: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    num_chunks: int
    total_token_per_batch: list[int]


@dataclass
class AiterFlashAttentionChunkPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    chunk_context_metadata: AiterChunkContextMetadata


@dataclass
class AiterFlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_actual_kv_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor

    # prefill and deocde split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_chunk_prefills: int
    num_chunk_prefill_tokens: int

    decode_metadata: AiterFlashAttentionDecodeMetadata | None
    pure_prefill_metadata: AiterFlashAttentionPrefillMetadata | None
    chunk_prefill_metadata: AiterFlashAttentionChunkPrefillMetadata | None


    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    total_tokens: int


class AiterFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[AiterFlashAttentionMetadata]
):
    cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None
        self.total_tokens: int = 0

        self.chunk_prefill_workspace_size = \
            _CP_TOKENS_PER_ITER_ROCM * self.num_heads_kv * self.headdim

        self.chunk_prefill_workspace = torch.empty(
            [2, _CP_TOKENS_PER_ITER_ROCM, self.num_heads_kv, self.headdim],
            dtype=self.model_config.dtype,
            device=device)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        self.total_tokens = (
            self.model_config.max_model_len
            * self.vllm_config.scheduler_config.max_num_partial_prefills
        )
        res = self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.total_tokens = 0
        return res

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "AiterFlashAttentionMetadata":

        split_ret = \
            split_decodes_prefills_and_chunk(common_attn_metadata,
                                       decode_threshold=self.reorder_batch_threshold)

        num_decodes, num_chunk_prefills, num_pure_prefills, num_decode_tokens,\
            num_chunk_prefill_tokens, num_pure_prefill_tokens = split_ret

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        seq_lens = common_attn_metadata.seq_lens_cpu

        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AiterFlashAttentionDecodeMetadata(
                max_query_len=query_lens_cpu[:num_decodes].max().item(),
                min_query_len=query_lens_cpu[:num_decodes].min().item(),
                max_seq_len=seq_lens[:num_decodes].max().item(),
                query_start_loc=common_attn_metadata.
                query_start_loc[:num_decodes + 1])

        pure_prefill_metadata = None
        if num_pure_prefills > 0:
            query_lens_for_pure_prefill = query_lens_cpu[num_decodes +
                                                         num_chunk_prefills:]
            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes + num_chunk_prefills:]
            pure_prefill_metadata = AiterFlashAttentionPrefillMetadata(
                max_query_len=query_lens_for_pure_prefill.max().item(),
                min_query_len=query_lens_for_pure_prefill.min().item(),
                max_seq_len=seq_lens[num_decodes +
                                     num_chunk_prefills:].max().item(),
                query_start_loc=query_start_loc_device -
                query_start_loc_device[0])

        chunk_prefill_metadata = None
        if num_chunk_prefills > 0:
            query_lens_for_chunk_prefill = query_lens_cpu[
                num_decodes:num_decodes + num_chunk_prefills]
            seq_lens_for_chunk_prefill = common_attn_metadata.seq_lens_cpu[
                num_decodes:num_decodes + num_chunk_prefills]
            computed_kv_lens = seq_lens_for_chunk_prefill - \
                query_lens_for_chunk_prefill

            # allocate the equal amount of workspace for
            # each chunk prefill request
            max_context_chunk = (_CP_TOKENS_PER_ITER_ROCM //
                                 num_chunk_prefills)
            num_chunks = cdiv(computed_kv_lens.max().item(), max_context_chunk)

            chunk_starts = torch.arange(
                num_chunks, dtype=torch.int32).unsqueeze(1).expand(
                    -1, num_chunk_prefills) * max_context_chunk
            chunk_ends = torch.min(computed_kv_lens.unsqueeze(0),
                                   chunk_starts + max_context_chunk)
            chunk_seq_lens = (chunk_ends - chunk_starts).clamp(
                min=0)  # [num_chunks, num_chunk_prefills]
            cu_seq_lens_cpu = torch.zeros([num_chunks, num_chunk_prefills + 1],
                                          dtype=torch.int32,
                                          pin_memory=True)
            torch.cumsum(chunk_seq_lens,
                         dim=1,
                         out=cu_seq_lens_cpu[:, 1:],
                         dtype=torch.int32)
            max_cum_tokens = cu_seq_lens_cpu[:, -1].max().item()

            range_idx = torch.arange(max_cum_tokens,
                                     dtype=torch.int32)[None, None, :]
            idx_to_batch_tensor = \
                range_idx == cu_seq_lens_cpu[:,1:][:, :,None]
            idx_to_batch_tensor = idx_to_batch_tensor.sum(
                dim=1)  # [num_chunks, max_cum_tokens]
            token_to_batch_tensor = torch.cumsum(idx_to_batch_tensor, dim=1)

            chunk_context_metadata = AiterChunkContextMetadata(
                workspace=self.chunk_prefill_workspace,
                cu_seq_lens_chunk=cu_seq_lens_cpu.to(self.device,
                                                     non_blocking=True),
                chunk_starts=chunk_starts.to(self.device, non_blocking=True),
                seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                seq_lens=chunk_seq_lens,
                token_to_batch=token_to_batch_tensor.to(self.device,
                                                        non_blocking=True),
                num_chunks=num_chunks,
                total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist())

            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes:num_decodes + num_chunk_prefills + 1]
            seq_lens_device = common_attn_metadata.seq_lens[
                num_decodes:num_decodes + num_chunk_prefills]
            cu_seq_lens = torch.zeros(num_chunk_prefills + 1,
                                      dtype=torch.int32,
                                      device=seq_lens_device.device)
            torch.cumsum(seq_lens_device,
                         dim=0,
                         dtype=cu_seq_lens.dtype,
                         out=cu_seq_lens[1:])
            chunk_prefill_metadata = AiterFlashAttentionChunkPrefillMetadata(
                max_query_len=query_lens_for_chunk_prefill.max().item(),
                min_query_len=query_lens_for_chunk_prefill.min().item(),
                max_seq_len=seq_lens[num_decodes:num_decodes +
                                     num_chunk_prefills].max().item(),
                query_start_loc=query_start_loc_device -
                query_start_loc_device[0],
                chunk_context_metadata=chunk_context_metadata)

        num_actual_kv_tokens = torch.sum(seq_lens).item()

        use_cascade = common_prefix_len > 0

        attn_metadata = AiterFlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_pure_prefills,
            num_prefill_tokens=num_pure_prefill_tokens,
            num_chunk_prefills=num_chunk_prefills,
            num_chunk_prefill_tokens=num_chunk_prefill_tokens,
            decode_metadata=decode_metadata,
            pure_prefill_metadata=pure_prefill_metadata,
            chunk_prefill_metadata=chunk_prefill_metadata,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            total_tokens=self.total_tokens,
        )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class AiterFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_supported_kernel_block_size() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes."
            )

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionImpl"]:
        return AiterFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AiterFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionMetadataBuilder"]:
        return AiterFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")

        return (2, num_blocks, block_size, num_kv_heads, head_size)


class AiterFlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = [-1, -1]
        else:
            self.sliding_window = [sliding_window - 1, 0]
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0.0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        AiterFlashAttentionBackend.validate_head_size(head_size)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashAttentionImpl"
            )

    def chunk_prefill_forward(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: float,
        v_scale: float,
    ):
        out, lse = aiter.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            min_seqlen_q=min_seqlen_q,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            return_lse=True)
        assert attn_metadata.chunk_prefill_metadata is not None
        chunk_context_metadata = \
            attn_metadata.chunk_prefill_metadata.chunk_context_metadata
        num_chunks = chunk_context_metadata.num_chunks
        workspace = chunk_context_metadata.workspace
        cu_seqlens_kv = chunk_context_metadata.cu_seq_lens_chunk
        max_seqlens = chunk_context_metadata.max_seq_lens
        chunk_starts = chunk_context_metadata.chunk_starts
        token_to_batch = chunk_context_metadata.token_to_batch
        total_token_per_batch = chunk_context_metadata.total_token_per_batch
        key_fetched, value_fetched = workspace[0], workspace[1]
        chunked_output = None
        chunked_lse = None
        for chunk_idx in range(num_chunks):

            cp_mha_gather_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                key=key_fetched,
                value=value_fetched,
                block_tables=block_table,
                k_scales=k_scale,
                v_scales=v_scale,
                cu_seqlens_kv=cu_seqlens_kv[chunk_idx],
                token_to_batch=token_to_batch[chunk_idx],
                seq_starts=chunk_starts[chunk_idx],
                dequant=True,
                kv_cache_layout="NHD",
                total_tokens=total_token_per_batch[chunk_idx],
            )

            suf_out, suf_lse = aiter.flash_attn_varlen_func(
                q=query,
                k=key_fetched,
                v=value_fetched,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv[chunk_idx],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlens[chunk_idx],
                min_seqlen_q=min_seqlen_q,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                return_lse=True)
            if chunked_output is None:
                chunked_output = suf_out
                chunked_lse = suf_lse
            else:
                tmp_output = torch.empty_like(out)
                tmp_lse = torch.empty_like(lse)
                merge_attn_states(output=tmp_output,
                                  output_lse=tmp_lse,
                                  prefix_output=chunked_output,
                                  prefix_lse=chunked_lse,
                                  suffix_output=suf_out,
                                  suffix_lse=suf_lse)
                chunked_output = tmp_output
                chunked_lse = tmp_lse

            merge_attn_states(
                output=output,
                prefix_output=chunked_output,
                prefix_lse=chunked_lse,
                suffix_output=out,
                suffix_lse=lse,
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with AiterFlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is
        # executed in eager-mode PyTorch. Thus, we need to be careful
        # about any CPU overhead in this method. For example, `view`
        # and `slice` (or `[:n]`) operations are surprisingly slow even
        # in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping
            # is not padded. However, we don't need to do
            # key[:num_actual_tokens] and value[:num_actual_tokens] because
            # the reshape_and_cache_flash op uses the slot_mapping's shape
            # to determine the number of actual tokens.

            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        # decode:chunk_prefill:pure_prefill
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]

        output_actual_tokens = output[:num_actual_tokens]

        num_decodes = attn_metadata.num_decodes
        num_pure_prefills = attn_metadata.num_prefills
        num_chunk_prefills = attn_metadata.num_chunk_prefills

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_chunk_prefill_tokens = attn_metadata.num_chunk_prefill_tokens
        if not attn_metadata.use_cascade:

            # calculate for pure prefills
            if num_pure_prefills > 0:
                assert attn_metadata.pure_prefill_metadata is not None

                prefill_query = query[num_decode_tokens +
                                      num_chunk_prefill_tokens:]
                prefill_key = key[num_decode_tokens +
                                  num_chunk_prefill_tokens:]
                prefill_value = value[num_decode_tokens +
                                      num_chunk_prefill_tokens:]

                aiter.flash_attn_varlen_func(
                    q=prefill_query,
                    k=prefill_key,
                    v=prefill_value,
                    cu_seqlens_q=attn_metadata.pure_prefill_metadata.
                    query_start_loc,
                    cu_seqlens_k=attn_metadata.pure_prefill_metadata.
                    query_start_loc,
                    max_seqlen_q=attn_metadata.pure_prefill_metadata.
                    max_query_len,
                    max_seqlen_k=attn_metadata.pure_prefill_metadata.
                    max_seq_len,
                    min_seqlen_q=attn_metadata.pure_prefill_metadata.
                    min_query_len,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    out=output_actual_tokens[num_decode_tokens +
                                             num_chunk_prefill_tokens:],
                )

            # calculate for chunk prefills
            if num_chunk_prefills > 0:
                assert attn_metadata.chunk_prefill_metadata is not None
                chunk_prefill_querys = query[
                    num_decode_tokens:num_decode_tokens +
                    num_chunk_prefill_tokens]
                chunk_prefill_keys = key[num_decode_tokens:num_decode_tokens +
                                         num_chunk_prefill_tokens]
                chunk_prefill_values = value[
                    num_decode_tokens:num_decode_tokens +
                    num_chunk_prefill_tokens]
                chunk_prefill_outputs = output[
                    num_decode_tokens:num_decode_tokens +
                    num_chunk_prefill_tokens]
                self.chunk_prefill_forward(
                    attn_metadata=attn_metadata,
                    query=chunk_prefill_querys,
                    key=chunk_prefill_keys,
                    value=chunk_prefill_values,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    output=chunk_prefill_outputs,
                    cu_seqlens_q=attn_metadata.chunk_prefill_metadata.
                    query_start_loc,
                    max_seqlen_q=attn_metadata.chunk_prefill_metadata.
                    max_query_len,
                    max_seqlen_k=attn_metadata.chunk_prefill_metadata.
                    max_seq_len,
                    min_seqlen_q=attn_metadata.chunk_prefill_metadata.
                    min_query_len,
                    block_table=attn_metadata.
                    block_table[num_decodes:num_decodes + num_chunk_prefills],
                    slot_mapping=attn_metadata.
                    slot_mapping[num_decodes:num_decodes + num_chunk_prefills],
                    k_scale=layer._k_scale,
                    v_scale=layer._v_scale,
                )

            # calculate for decodes
            if num_decodes > 0:
                assert attn_metadata.decode_metadata is not None
                _, num_heads, head_size = query.shape
                nbytes_per_qo_elem = torch.finfo(query.dtype).bits // 8
                max_num_partitions = (
                    attn_metadata.decode_metadata.max_seq_len +
                    _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM

                workspace_buffer = torch.empty(
                    (num_decode_tokens * num_heads * max_num_partitions *
                     head_size) * nbytes_per_qo_elem + 2 *
                    (num_decode_tokens * num_heads * max_num_partitions) * 4,
                    dtype=torch.uint8,
                    device=output.device,
                )

                torch.ops.aiter.paged_attention_v1(
                    output_actual_tokens[:num_decode_tokens],
                    workspace_buffer,
                    query[:num_decode_tokens],
                    key_cache,
                    value_cache,
                    self.scale,
                    attn_metadata.block_table[:num_decodes],
                    attn_metadata.decode_metadata.query_start_loc,
                    attn_metadata.seq_lens[:num_decodes],
                    attn_metadata.decode_metadata.max_seq_len,
                    self.alibi_slopes,
                    self.kv_cache_dtype,
                    "NHD",
                    self.logits_soft_cap,
                    layer._k_scale,
                    layer._v_scale,
                    None,
                    _PARTITION_SIZE_ROCM,
                )
        else:
            raise NotImplementedError(
                "Cascade attention is not implemented for ROCM AITER"
            )

        return output

