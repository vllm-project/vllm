# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with AiterFlashAttention."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_dtype_size,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    PrequantizedQKV,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_prefills_and_extends,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    is_quantized_kv_cache,
)

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024


def _uses_direct_fp8_context_gather(layers: list[Attention], cache_dtype: str) -> bool:
    """Whether chunked prefill gathers cached K/V into an FP8 workspace.

    Every layer must both support prequantized Q/K/V and send it: with a BF16
    query and no descales, an FP8 workspace would hold unscaled K/V.
    """
    return (
        bool(layers)
        and all(
            layer.impl.supports_prequantized_qkv_input
            and layer.receives_prequantized_qkv
            for layer in layers
        )
        and is_quantized_kv_cache(cache_dtype)
        and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
    )


def _slice_prequantized_qkv(
    prequantized_qkv: PrequantizedQKV,
    token_slice: slice,
    sequence_slice: slice,
) -> PrequantizedQKV:
    """Select matching token and sequence ranges for one attention phase."""
    return PrequantizedQKV(
        query=prequantized_qkv.query[token_slice],
        key=prequantized_qkv.key[token_slice],
        value=prequantized_qkv.value[token_slice],
        query_descale=prequantized_qkv.query_descale[sequence_slice],
        key_descale=prequantized_qkv.key_descale[sequence_slice],
        value_descale=prequantized_qkv.value_descale[sequence_slice],
    )


def _rebased_starts(query_start_loc_cpu: torch.Tensor) -> list[int]:
    """Cumulative token offsets of a phase's sequences, starting at zero."""
    starts = query_start_loc_cpu.tolist()
    return [start - starts[0] for start in starts]


def _expand_descale(
    descale: torch.Tensor, num_sequences: int, num_kv_heads: int
) -> torch.Tensor:
    """View a descale as (sequences, KV heads): per-sequence, per-head or scalar."""
    if descale.dim() == 2 and descale.shape[0] >= num_sequences:
        return descale
    return descale.reshape(1, -1).expand(num_sequences, num_kv_heads)


def _grouped_fp8_varlen_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_starts: Sequence[int],
    k_starts: Sequence[int],
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    out_dtype: torch.dtype,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] | None,
    alibi_slopes: torch.Tensor | None,
    sink_ptr: torch.Tensor | None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
):
    """FP8 varlen attention with one call per (sequence, KV head).

    The AITER FP8 hd256 varlen kernel reads only ``descale[0, 0]`` whatever the
    descale shape, so Q/K/V quantized per (sequence, KV head) are scaled
    correctly only when a call holds a single sequence and a single KV head.
    A batch with one of each is the ordinary call.

    ``q_starts`` and ``k_starts`` are the host-side cumulative token offsets of
    the sequences in ``q`` and ``k``. A sequence without keys (an extend with
    no context in this chunk) gets zero output and a ``-inf`` log-sum-exp,
    which ``merge_attn_states`` gives no weight.
    """
    num_sequences = len(q_starts) - 1
    num_heads, num_kv_heads = q.shape[1], k.shape[1]
    q_descale = _expand_descale(q_descale, num_sequences, num_kv_heads)
    k_descale = _expand_descale(k_descale, num_sequences, num_kv_heads)
    v_descale = _expand_descale(v_descale, num_sequences, num_kv_heads)
    if num_sequences == 1 and num_kv_heads == 1:
        return rocm_aiter_ops.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            return_lse=return_lse,
            out=out,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            sink_ptr=sink_ptr,
        )

    group = num_heads // num_kv_heads
    # One host-to-device copy of every sequence's (query, key) boundaries.
    cu_seqlens = torch.tensor(
        [
            [
                [0, q_starts[i + 1] - q_starts[i]],
                [0, k_starts[i + 1] - k_starts[i]],
            ]
            for i in range(num_sequences)
        ],
        dtype=torch.int32,
    ).to(q.device, non_blocking=True)
    result = (
        out
        if out is not None
        else torch.zeros(q.shape, dtype=out_dtype, device=q.device)
    )
    lse = (
        torch.full((num_heads, q.shape[0]), float("-inf"), device=q.device)
        if return_lse
        else None
    )
    for seq in range(num_sequences):
        q_len = q_starts[seq + 1] - q_starts[seq]
        k_len = k_starts[seq + 1] - k_starts[seq]
        if q_len == 0 or k_len == 0:
            if out is not None:
                out[q_starts[seq] : q_starts[seq + 1]] = 0
            continue
        q_seq = slice(q_starts[seq], q_starts[seq + 1])
        k_seq = slice(k_starts[seq], k_starts[seq + 1])
        for kv_head in range(num_kv_heads):
            heads = slice(kv_head * group, (kv_head + 1) * group)
            attention = rocm_aiter_ops.flash_attn_varlen_func(
                q=q[q_seq, heads],
                k=k[k_seq, kv_head : kv_head + 1],
                v=v[k_seq, kv_head : kv_head + 1],
                cu_seqlens_q=cu_seqlens[seq, 0],
                cu_seqlens_k=cu_seqlens[seq, 1],
                max_seqlen_q=q_len,
                max_seqlen_k=k_len,
                min_seqlen_q=min(min_seqlen_q, q_len),
                return_lse=return_lse,
                q_descale=q_descale[seq, kv_head].reshape(1),
                k_descale=k_descale[seq, kv_head].reshape(1),
                v_descale=v_descale[seq, kv_head].reshape(1),
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                sink_ptr=sink_ptr,
            )
            call_out, call_lse = attention if return_lse else (attention, None)
            result[q_seq, heads] = call_out
            if lse is not None:
                lse[heads, q_seq] = call_lse
    return (result, lse) if return_lse else result


if current_platform.is_rocm():
    from vllm.triton_utils import tl, triton

    def block_size(x, head_dim):
        return min(65536 // x.element_size(), triton.next_power_of_2(head_dim))

    def num_programs(total_tokens):
        return min(total_tokens, num_compute_units())

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
        k_scale_ptr,  # [1] / [num_blocks, num_kv_heads, page_size]
        v_scale_ptr,
        num_heads,
        head_size,
        x,
        max_block_num,
        k_cache_stride0,
        k_cache_stride1,
        k_cache_stride2,
        v_cache_stride0,
        v_cache_stride1,
        v_cache_stride2,
        DEQUANT: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        CACHE_FORMAT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        head_id = tl.program_id(1)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        key_ptr_offset = (
            key_ptr + token_id * head_size * num_heads + head_id * head_size
        )
        value_ptr_offset = (
            value_ptr + token_id * head_size * num_heads + head_id * head_size
        )
        batch_idx = tl.load(token_to_batch_ptr + token_id)
        batch_start = tl.load(seq_start_ptr + batch_idx)
        token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
        batch_offset = token_id - token_start + batch_start
        block_offset = batch_offset // PAGE_SIZE
        block_id = tl.load(
            block_table_ptr + max_block_num * batch_idx + block_offset
        ).to(tl.int64)
        slot_id = batch_offset % PAGE_SIZE

        if CACHE_FORMAT == "NHD":
            # for kv cache layout as
            # K: [num_blocks, page_size, num_head, head_dim]
            # V: [num_blocks, page_size, num_head, head_dim]
            key_cache_ptr_offset = (
                key_cache_ptr
                + block_id * k_cache_stride0
                + slot_id * k_cache_stride1
                + head_id * k_cache_stride2
            )
            value_cache_ptr_offset = (
                value_cache_ptr
                + block_id * v_cache_stride0
                + slot_id * v_cache_stride1
                + head_id * v_cache_stride2
            )
            k_reg = tl.load(key_cache_ptr_offset + col_offsets)
            v_reg = tl.load(value_cache_ptr_offset + col_offsets)
            if DEQUANT:
                k_scale = tl.load(k_scale_ptr)
                v_scale = tl.load(v_scale_ptr)
                k_reg = (k_reg.to(tl.float32) * k_scale).to(
                    key_ptr_offset.dtype.element_ty
                )
                v_reg = (v_reg.to(tl.float32) * v_scale).to(
                    value_ptr_offset.dtype.element_ty
                )
            tl.store(key_ptr_offset + col_offsets, k_reg)
            tl.store(value_ptr_offset + col_offsets, v_reg)

        elif CACHE_FORMAT == "SHUFFLE":
            # for kv cache layout as
            # K: [num_blocks, num_head, head_dim // x, page_size, x]
            # V: [num_blocks, num_head, page_size // x, head_dim, x]
            # Take the block stride from the tensor, as the NHD branch does:
            # the two sides only sit a dense page apart when they are separate
            # planes, and models that pin a block-compact layout interleave
            # them instead.
            key_cache_ptr_offset = (
                key_cache_ptr
                + block_id * k_cache_stride0
                + head_id * head_size * PAGE_SIZE
                + slot_id * x
            )
            value_cache_ptr_offset = (
                value_cache_ptr
                + block_id * v_cache_stride0
                + head_id * head_size * PAGE_SIZE
                + (slot_id // x) * head_size * x
                + slot_id % x
            )
            k_reg_offset = col_offsets // x * PAGE_SIZE * x + col_offsets % x
            v_reg_offset = col_offsets * x
            k_reg = tl.load(key_cache_ptr_offset + k_reg_offset)
            v_reg = tl.load(value_cache_ptr_offset + v_reg_offset)
            if DEQUANT:
                k_scale = tl.load(k_scale_ptr)
                v_scale = tl.load(v_scale_ptr)
                k_reg = k_reg.to(tl.float32) * k_scale
                v_reg = v_reg.to(tl.float32) * v_scale
            tl.store(key_ptr_offset + col_offsets, k_reg)
            tl.store(value_ptr_offset + col_offsets, v_reg)

    def cp_mha_gather_cache(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_tables: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        token_to_batch: torch.Tensor,
        seq_starts: torch.Tensor,
        dequant: bool,
        kv_cache_layout: str,
        total_tokens: int,
    ):
        assert kv_cache_layout in ["NHD", "SHUFFLE"], (
            "kv_cache_layout only supports NHD, SHUFFLE"
        )
        head_dim = key.shape[2]
        x = 16 // key_cache.element_size()
        # assert dequant is True, "Currently, we only support "\
        # "gather cache with dequant"
        # For k cache layout: [num_blocks, page_size, num_heads, head_dim]
        assert head_dim == key_cache.shape[3], (
            "We assume your kv cache layout is [num_blocks, "
            "page_size, num_heads, head_dim], but got otherwise"
        )
        page_size = key_cache.shape[1]
        num_heads = key_cache.shape[2]

        # Pass actual tensor strides so the kernel works correctly
        # even when the cache is non-contiguous (e.g, for hybrid model)
        k_strides = key_cache.stride()
        v_strides = value_cache.stride()

        grid = lambda meta: (total_tokens, num_heads)
        cp_mha_gather_cache_kernel[grid](
            key_cache,
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
            k_strides[0],
            k_strides[1],
            k_strides[2],
            v_strides[0],
            v_strides[1],
            v_strides[2],
            DEQUANT=dequant,
            PAGE_SIZE=page_size,
            CACHE_FORMAT=kv_cache_layout,
            BLOCK_SIZE=head_dim,
        )

    @triton.jit
    def reshape_and_cache_shuffle_kernel(
        key_ptr,  # [num_tokens, num_kv_heads, head_size]
        value_ptr,  # [num_tokens, num_kv_heads, head_size]
        key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
        slot_mapping_ptr,  # [num_tokens]
        k_scale_ptr,  # [num_blocks, num_kv_heads, block_size]
        v_scale_ptr,  # [num_blocks, num_kv_heads, block_size]
        x,
        k_stride0,
        v_stride0,
        k_cache_block_stride,
        v_cache_block_stride,
        block_size,
        head_size,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
        QUANT: tl.constexpr,
    ):
        tid = tl.program_id(0)
        head_id = tl.program_id(1)
        offset = tl.arange(0, BLOCK_SIZE)
        src_offset_k = tid * k_stride0 + head_id * head_size
        src_offset_v = tid * v_stride0 + head_id * head_size
        slot_id = tl.load(slot_mapping_ptr + tid)
        if slot_id < 0:
            return
        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        k_dst_offset = (
            block_id * k_cache_block_stride + head_id * head_size * block_size
        )
        dst_k_shuffle_offset = (
            k_dst_offset + offset // x * block_size * x + block_offset * x + offset % x
        )
        v_dst_offset = (
            block_id * v_cache_block_stride + head_id * head_size * block_size
        )
        dst_v_shuffle_offset = (
            v_dst_offset
            + block_offset // x * head_size * x
            + offset * x
            + block_offset % x
        )
        k_val = tl.load(key_ptr + src_offset_k + offset)
        v_val = tl.load(value_ptr + src_offset_v + offset)
        if QUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)
            k_dtype = key_cache_ptr.type.element_ty
            v_dtype = value_cache_ptr.type.element_ty
            k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
            v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)
        tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
        tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)

    def reshape_and_cache_shuffle_triton(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
    ):
        num_tokens = slot_mapping.shape[0]
        _, num_kv_heads, head_size = key.shape
        num_blocks, block_size, _, _ = key_cache.shape
        x = 16 // key_cache.element_size()
        QUANT = False
        if is_quantized_kv_cache(kv_cache_dtype):
            QUANT = True
        grid = (
            num_tokens,
            num_kv_heads,
        )
        reshape_and_cache_shuffle_kernel[grid](
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            k_scales,
            v_scales,
            x,
            key.stride(0),
            value.stride(0),
            key_cache.stride(0),
            value_cache.stride(0),
            block_size,
            head_size,
            num_kv_heads,
            BLOCK_SIZE=head_size,
            QUANT=QUANT,
        )


logger = init_logger(__name__)


@dataclass
class AiterFlashAttentionDecodeMetadata:
    max_query_len: int
    # Sliding-window decode: window-trimmed block table and rebased seq lens.
    swa_block_table: torch.Tensor | None = None
    swa_seq_lens: torch.Tensor | None = None
    swa_max_seq_len: int = 0


@dataclass
class AiterFlashAttentionPrefillMetadata:
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    # Host copy of query_start_loc, for per-sequence FP8 attention calls.
    query_start_loc_host: list[int] | None = None


@dataclass
class AiterChunkSlidingWindowMetadata:
    swa_cu_seqlens: torch.Tensor
    swa_seq_starts: torch.Tensor
    swa_token_to_batch: torch.Tensor
    swa_max_seqlens: int
    swa_total_tokens: int
    swa_workspace: torch.Tensor


@dataclass
class AiterChunkContextMetadata:
    workspace: torch.Tensor
    cu_seq_lens_chunk: torch.Tensor
    chunk_starts: torch.Tensor
    token_to_batch: torch.Tensor
    max_seq_lens: list[int]
    num_chunks: int
    total_token_per_batch: list[int]
    swa_metadata: AiterChunkSlidingWindowMetadata | None
    # Host copy of cu_seq_lens_chunk, for per-sequence FP8 attention calls.
    cu_seq_lens_chunk_host: list[list[int]] | None = None


@dataclass
class AiterFlashAttentionChunkPrefillMetadata:
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    chunk_context_metadata: AiterChunkContextMetadata
    # Host copy of query_start_loc, for per-sequence FP8 attention calls.
    query_start_loc_host: list[int] | None = None


@dataclass
class AiterKVSharingMetadata:
    workspace: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_batch: torch.Tensor
    seq_starts: torch.Tensor


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
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    causal: bool

    # prefill and decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_extends: int
    num_extend_tokens: int

    decode_metadata: AiterFlashAttentionDecodeMetadata | None
    prefill_metadata: AiterFlashAttentionPrefillMetadata | None
    extend_metadata: AiterFlashAttentionChunkPrefillMetadata | None

    # Only for fp8 shuffle layout kv cache, we allocate kv_scale for each layer
    # since we might integrate per token quant for kv cache in the future.
    k_scale: dict[str, torch.Tensor] | None
    v_scale: dict[str, torch.Tensor] | None
    kv_sharing_metadata: AiterKVSharingMetadata | None = None


@dataclass
class AsmKvScaleTables:
    """Dense per-slot scale tables the asm decode kernel reads for an fp8 cache.

    One table per distinct (device, shape, scale), shared by every layer and
    every metadata builder in the process, so the target and draft models with
    equal scales reuse a single allocation. Tables are constants: never write
    to one in place; a per-token or dynamic scale needs its own buffer.
    """

    tables: dict[tuple[str, int, int, int, float], torch.Tensor] = field(
        default_factory=dict
    )

    def get(
        self,
        device: torch.device,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        key = (str(device), num_blocks, num_kv_heads, block_size, scale)
        table = self.tables.get(key)
        if table is None:
            table = torch.full(
                (num_blocks, num_kv_heads, block_size),
                scale,
                dtype=torch.float32,
                device=device,
            )
            self.tables[key] = table
        return table


class AiterFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[AiterFlashAttentionMetadata]
):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    asm_kv_scale_tables: ClassVar[AsmKvScaleTables] = AsmKvScaleTables()

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
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        sliding_window_configs: set[tuple[int, int] | None] = set()
        kv_sharing_shape = None
        builder_layers: list[Attention] = []
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for name, layer in layers.items():
            if name not in layer_names:
                continue
            builder_layers.append(layer)
            assert isinstance(layer.impl, AiterFlashAttentionImpl), (
                "Aiter Flash Attention Metadata Builder can only be used "
                "with Aiter Flash Attention Impl."
            )
            sliding_window_configs.add(layer.impl.sliding_window)
            if (
                layer.kv_sharing_target_layer_name is not None
                or layer.kv_cache_update_bypass
            ):
                kv_sharing_shape = (layer.impl.num_kv_heads, layer.impl.head_size)

        while len(sliding_window_configs) > 0:
            sliding_window_config = sliding_window_configs.pop()
            if sliding_window_config is not None and sliding_window_config[0] != -1:
                assert self.aot_sliding_window is None, (
                    "Aiter Flash ATTENTION can only support one valid sliding window!"
                )
                self.aot_sliding_window = sliding_window_config

        self.direct_fp8_context_gather = _uses_direct_fp8_context_gather(
            builder_layers, self.cache_config.cache_dtype
        )
        if self.direct_fp8_context_gather:
            logger.info_once(
                "Using an FP8 context workspace for ROCm AITER chunked prefill "
                "(KV cache dtype: %s)",
                kv_cache_spec.dtype,
            )
        workspace_dtype = (
            current_platform.fp8_dtype()
            if self.direct_fp8_context_gather
            else self.model_config.dtype
        )

        # Persistent so full CUDA graphs keep reading the same storage.
        self._swa_block_table: torch.Tensor | None = None
        self._swa_seq_lens: torch.Tensor | None = None
        self._swa_block_offsets: torch.Tensor | None = None
        if self.aot_sliding_window is not None:
            window = self.aot_sliding_window[0] + 1
            num_window_blocks = cdiv(window, self.block_size) + 1
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._swa_block_table = torch.zeros(
                (max_num_reqs, num_window_blocks), dtype=torch.int32, device=device
            )
            self._swa_seq_lens = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )
            self._swa_block_offsets = torch.arange(
                num_window_blocks, dtype=torch.int64, device=device
            )

        self.extend_workspace = torch.empty(
            [2, _CP_TOKENS_PER_ITER_ROCM, self.num_heads_kv, self.headdim],
            dtype=workspace_dtype,
            device=device,
        )
        self.k_scale: dict[str, torch.Tensor] | None = None
        self.v_scale: dict[str, torch.Tensor] | None = None
        self.kv_sharing_workspace: torch.Tensor | None = None
        if kv_sharing_shape is not None:
            self.kv_sharing_workspace = torch.empty(
                (
                    2,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    *kv_sharing_shape,
                ),
                dtype=self.model_config.dtype,
                device=device,
            )

    def _build_swa_decode_tables(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        max_seq_len: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Trim the decode block table to the sliding window.

        Exact: the kernel's window mask drops the leading tokens of the first
        kept block, and RoPE is already applied in the cache.
        """
        if (
            self._swa_block_table is None
            or self.aot_sliding_window is None
            or num_decodes > self._swa_block_table.shape[0]
        ):
            return None, None, 0
        assert self._swa_seq_lens is not None and self._swa_block_offsets is not None
        window = self.aot_sliding_window[0] + 1
        lens = seq_lens[:num_decodes]
        first_block = torch.clamp(lens - window, min=0) // self.block_size
        index = (first_block.to(torch.int64)[:, None] + self._swa_block_offsets).clamp_(
            max=block_table.shape[1] - 1
        )
        swa_block_table = self._swa_block_table[:num_decodes]
        torch.gather(block_table[:num_decodes], 1, index, out=swa_block_table)
        swa_seq_lens = self._swa_seq_lens[:num_decodes]
        swa_seq_lens.copy_(lens - first_block * self.block_size)
        return (
            swa_block_table,
            swa_seq_lens,
            min(max_seq_len, window + self.block_size - 1),
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def _build_asm_kv_scale_tables(self) -> None:
        # Only for fp8 shuffle kv cache: the asm decode kernel reads a dense
        # [num_blocks, num_kv_heads, block_size] scale table per layer. Map each
        # layer to the shared table for its scale once the KV cache is bound.
        if (
            self.k_scale is not None
            or not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
            or not is_quantized_kv_cache(self.vllm_config.cache_config.cache_dtype)
        ):
            return
        attn_layers = self.vllm_config.compilation_config.static_forward_context
        k_scale_tables: dict[str, torch.Tensor] = {}
        v_scale_tables: dict[str, torch.Tensor] = {}
        for layer_name in self.layer_names:
            layer = attn_layers[layer_name]
            num_blocks = layer.kv_cache.shape[0]
            k_scale_tables[layer_name] = self.asm_kv_scale_tables.get(
                self.device,
                num_blocks,
                self.num_heads_kv,
                self.block_size,
                layer._k_scale_float,
            )
            v_scale_tables[layer_name] = self.asm_kv_scale_tables.get(
                self.device,
                num_blocks,
                self.num_heads_kv,
                self.block_size,
                layer._v_scale_float,
            )
        self.k_scale, self.v_scale = k_scale_tables, v_scale_tables

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "AiterFlashAttentionMetadata":
        assert self.reorder_batch_threshold is not None
        split_ret = split_decodes_prefills_and_extends(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
        )
        self._build_asm_kv_scale_tables()
        (
            num_decodes,
            num_extends,
            num_prefills,
            num_decode_tokens,
            num_extend_tokens,
            _,
        ) = split_ret

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        with gpu_sync_allowed():
            # Only copy seq_lens to CPU when prefill or extend is present to avoid a
            # blocking device→host transfer.
            seq_lens = (
                common_attn_metadata.seq_lens.cpu()
                if num_prefills > 0 or num_extends > 0
                else None
            )

        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        kv_sharing_metadata = None
        if self.kv_sharing_workspace is not None and seq_lens is not None:
            query_lens = query_lens_cpu[num_decodes:]
            token_to_batch = torch.repeat_interleave(
                torch.arange(len(query_lens), dtype=torch.int32, device="cpu"),
                query_lens,
            )
            query_start_loc = common_attn_metadata.query_start_loc[num_decodes:]
            kv_sharing_metadata = AiterKVSharingMetadata(
                workspace=self.kv_sharing_workspace,
                query_start_loc=query_start_loc - query_start_loc[0],
                token_to_batch=async_tensor_h2d(token_to_batch, self.device),
                seq_starts=async_tensor_h2d(
                    seq_lens[num_decodes:] - query_lens, self.device
                ),
            )

        decode_metadata = None
        if num_decodes > 0:
            decode_max_query_len = query_lens_cpu[:num_decodes].max().item()
            swa_block_table, swa_seq_lens, swa_max_seq_len = None, None, 0
            if decode_max_query_len == 1:
                swa_block_table, swa_seq_lens, swa_max_seq_len = (
                    self._build_swa_decode_tables(
                        common_attn_metadata.block_table_tensor,
                        common_attn_metadata.seq_lens,
                        num_decodes,
                        common_attn_metadata.max_seq_len,
                    )
                )
            decode_metadata = AiterFlashAttentionDecodeMetadata(
                max_query_len=decode_max_query_len,
                swa_block_table=swa_block_table,
                swa_seq_lens=swa_seq_lens,
                swa_max_seq_len=swa_max_seq_len,
            )

        prefill_metadata = None
        if num_prefills > 0:
            assert seq_lens is not None
            query_lens_for_prefill = query_lens_cpu[num_decodes + num_extends :]
            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes + num_extends :
            ]
            prefill_metadata = AiterFlashAttentionPrefillMetadata(
                max_query_len=query_lens_for_prefill.max().item(),
                max_seq_len=seq_lens[num_decodes + num_extends :].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
                query_start_loc_host=_rebased_starts(
                    query_start_loc_cpu[num_decodes + num_extends :]
                ),
            )

        extend_metadata = None
        if num_extends > 0:
            assert seq_lens is not None
            num_extends_slice = slice(num_decodes, num_decodes + num_extends)
            query_lens_for_extend = query_lens_cpu[num_extends_slice]
            seq_lens_for_extend = seq_lens[num_extends_slice]
            computed_kv_lens = seq_lens_for_extend - query_lens_for_extend
            swa_metadata = None
            if self.aot_sliding_window is not None:
                swa_seqlen_for_extend = torch.minimum(
                    seq_lens_for_extend,
                    query_lens_for_extend + self.aot_sliding_window[0] + 1,
                )
                cu_seq_lens = torch.zeros(
                    num_extends + 1,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                    pin_memory=True,
                )
                torch.cumsum(
                    swa_seqlen_for_extend,
                    dim=0,
                    dtype=cu_seq_lens.dtype,
                    out=cu_seq_lens[1:],
                )
                token_to_seq = torch.arange(
                    0,
                    num_extends,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                )
                token_to_seq = torch.repeat_interleave(
                    token_to_seq, swa_seqlen_for_extend
                )
                fetched_shape = cu_seq_lens[-1].item()
                # TODO(ganyi): Maybe reuse these 2 buffer from extend_workspace
                swa_workspace = torch.empty(
                    (2, fetched_shape, self.num_heads_kv, self.headdim),
                    dtype=self.vllm_config.model_config.dtype,
                    device=self.device,
                )

                seq_starts = seq_lens_for_extend - swa_seqlen_for_extend
                max_seqlen_k = swa_seqlen_for_extend.max().item()
                total_tokens = cu_seq_lens[-1].item()

                swa_metadata = AiterChunkSlidingWindowMetadata(
                    swa_cu_seqlens=cu_seq_lens.to(self.device, non_blocking=True),
                    swa_seq_starts=async_tensor_h2d(seq_starts, self.device),
                    swa_token_to_batch=async_tensor_h2d(token_to_seq, self.device),
                    swa_max_seqlens=max_seqlen_k,
                    swa_total_tokens=total_tokens,
                    swa_workspace=swa_workspace,
                )

            # allocate the equal amount of workspace for
            # each chunk prefill request
            max_context_chunk = _CP_TOKENS_PER_ITER_ROCM // num_extends
            num_chunks = cdiv(computed_kv_lens.max().item(), max_context_chunk)

            chunk_starts = (
                torch.arange(num_chunks, dtype=torch.int32)
                .unsqueeze(1)
                .expand(-1, num_extends)
                * max_context_chunk
            )
            chunk_ends = torch.min(
                computed_kv_lens.unsqueeze(0), chunk_starts + max_context_chunk
            )
            chunk_seq_lens = chunk_ends - chunk_starts
            chunk_seq_lens.clamp_(min=0)  # [num_chunks, num_extends]
            cu_seq_lens_cpu = torch.zeros(
                [num_chunks, num_extends + 1], dtype=torch.int32, pin_memory=True
            )
            torch.cumsum(
                chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
            )
            # Avoid .max() on an empty tensor when there is no context
            # (num_chunks == 0, e.g. Whisper encoder's first pass).
            max_cum_tokens = (
                cu_seq_lens_cpu[:, -1].max().item() if num_chunks > 0 else 0
            )

            range_idx = torch.arange(max_cum_tokens, dtype=torch.int32)[None, None, :]
            idx_to_batch_tensor = range_idx == cu_seq_lens_cpu[:, 1:][:, :, None]
            idx_to_batch_tensor = idx_to_batch_tensor.sum(
                dim=1
            )  # [num_chunks, max_cum_tokens]
            token_to_batch_tensor = torch.cumsum(idx_to_batch_tensor, dim=1)

            chunk_context_metadata = AiterChunkContextMetadata(
                workspace=self.extend_workspace,
                cu_seq_lens_chunk=cu_seq_lens_cpu.to(self.device, non_blocking=True),
                chunk_starts=async_tensor_h2d(chunk_starts, self.device),
                token_to_batch=async_tensor_h2d(token_to_batch_tensor, self.device),
                max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                num_chunks=num_chunks,
                total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist(),
                swa_metadata=swa_metadata,
                cu_seq_lens_chunk_host=cu_seq_lens_cpu.tolist(),
            )

            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes : num_decodes + num_extends + 1
            ]
            seq_lens_device = common_attn_metadata.seq_lens[num_extends_slice]
            cu_seq_lens = torch.zeros(
                num_extends + 1, dtype=torch.int32, device=seq_lens_device.device
            )
            torch.cumsum(
                seq_lens_device, dim=0, dtype=cu_seq_lens.dtype, out=cu_seq_lens[1:]
            )
            extend_metadata = AiterFlashAttentionChunkPrefillMetadata(
                max_query_len=query_lens_for_extend.max().item(),
                max_seq_len=seq_lens[num_extends_slice].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
                chunk_context_metadata=chunk_context_metadata,
                query_start_loc_host=_rebased_starts(
                    query_start_loc_cpu[num_decodes : num_decodes + num_extends + 1]
                ),
            )

        attn_metadata = AiterFlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            causal=common_attn_metadata.causal,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_extends=num_extends,
            num_extend_tokens=num_extend_tokens,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
            extend_metadata=extend_metadata,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            kv_sharing_metadata=kv_sharing_metadata,
        )
        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> AiterFlashAttentionMetadata:
        """Build attention metadata for draft model without CPU-GPU sync.

        During EAGLE drafting all requests are uniform decodes, so we can
        skip split_decodes_prefills_and_extends() and avoid all .cpu() /
        .item() calls that would otherwise break CUDA graph capture.
        """
        self._build_asm_kv_scale_tables()
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        decode_metadata = AiterFlashAttentionDecodeMetadata(
            max_query_len=common_attn_metadata.max_query_len,
        )

        return AiterFlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            causal=common_attn_metadata.causal,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_reqs,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            num_extends=0,
            num_extend_tokens=0,
            decode_metadata=decode_metadata,
            prefill_metadata=None,
            extend_metadata=None,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )


class AiterFlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """ENCODER_DECODER is not supported because the prefill path uses
        flash_attn_varlen_func with cu_seqlens_k set to decoder
        query_start_loc (not encoder seq lens) and causal=True, both of
        which are incorrect for cross-attention layers.
        """
        return attn_type in (AttentionType.DECODER,)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [16, 32]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    forward_includes_kv_cache_update: bool = False
    # External KV writers produce the packed layout, not the shuffle layout.
    supports_kv_cache_update_bypass: bool = (
        not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
    )

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionImpl"]:
        return AiterFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionMetadataBuilder"]:
        return AiterFlashAttentionMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Validate the block size the ROCm gather kernels require, and under
        the shuffle layout publish K and V as two head slots.

        The shuffle read/write kernels rearrange a whole ``head_size x
        block_size`` tile per side, so each side needs its own contiguous run
        in the page. Packed into the content dim they interleave per token
        instead, which no layout can undo.
        """
        # block_size == 1 is the per-token page-size probe (see
        # Platform.get_page_size_bytes); real blocks must be gatherable in
        # 16-token units by the ROCm kernel.
        if spec.block_size != 1 and spec.block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if not rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            return spec
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v, (
            "Separate K/V head slots require symmetric K/V head sizes."
        )
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=spec.num_kv_heads
            * spec.head_size
            * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            # pa_fwd_asm strides between pages by a whole dense page, so the two
            # head slots customize_spec publishes have to be separate planes
            # spanning every block: H outermost. Under LBHNC the slots alternate
            # block by block, doubling the stride the kernel assumes. LBHNC
            # stays listed for models whose mixed HNC shapes need a
            # block-compact layout; those read K/V by stride instead.
            return (KVCacheLayout.LHBNC, KVCacheLayout.LBHNC)
        # K and V come out of the content dim as transposed views rather than
        # copies, so the head dim may sit on either side of the block dim, but
        # the layer must stay outermost.
        return (KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        from vllm.platforms.rocm import get_cdna_version

        # DeviceCapability is currently created using torch.cuda.get_device_capability()
        # which is known to be buggy on rocm systems. on CDNA uses amd-smi which is
        # more reliable.
        return get_cdna_version() > 2

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True


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
        sinks: torch.Tensor | None = None,
    ) -> None:
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
            logits_soft_cap = 0.0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        from vllm.platforms.rocm import get_cdna_version

        self.supports_prequantized_qkv_input = (
            head_size == 256
            and get_cdna_version() == 4
            and self.num_queries_per_kv in (1, 2, 4, 8, 16)
            and self.logits_soft_cap == 0.0
            and self.sliding_window == (-1, -1)
            and self.sinks is None
            and is_quantized_kv_cache(self.kv_cache_dtype)
        )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Only decoder self-attention is supported for "
                "AiterFlashAttentionImpl. ENCODER_DECODER is not supported "
                "because the prefill path uses cu_seqlens_k set to decoder "
                "query_start_loc with causal=True, which is incorrect for "
                "cross-attention."
            )

    def _get_kv_cache_descales(
        self,
        layer: AttentionLayer,
        num_decodes: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # AITER treats non-null descales as an instruction to dequantize.
        if not is_quantized_kv_cache(self.kv_cache_dtype):
            return None, None
        descale_shape = (num_decodes, self.num_kv_heads)
        return (
            layer._k_scale.expand(descale_shape),
            layer._v_scale.expand(descale_shape),
        )

    def extend_for_sliding_window(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key_cache,
        value_cache,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        block_table: torch.Tensor,
        k_scale: float,
        v_scale: float,
    ):
        assert attn_metadata.extend_metadata is not None
        assert attn_metadata.extend_metadata.chunk_context_metadata is not None
        chunked_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        swa_metadata = chunked_metadata.swa_metadata
        assert swa_metadata is not None
        swa_cu_seqlens = swa_metadata.swa_cu_seqlens
        swa_seq_starts = swa_metadata.swa_seq_starts
        swa_token_to_batch = swa_metadata.swa_token_to_batch
        swa_max_seqlens = swa_metadata.swa_max_seqlens
        swa_total_tokens = swa_metadata.swa_total_tokens
        key_fetched, value_fetched = (
            swa_metadata.swa_workspace[0],
            swa_metadata.swa_workspace[1],
        )
        cp_mha_gather_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            key=key_fetched,
            value=value_fetched,
            block_tables=block_table,
            k_scales=k_scale,
            v_scales=v_scale,
            cu_seqlens_kv=swa_cu_seqlens,
            token_to_batch=swa_token_to_batch,
            seq_starts=swa_seq_starts,
            dequant=is_quantized_kv_cache(self.kv_cache_dtype),
            kv_cache_layout="NHD",
            total_tokens=swa_total_tokens,
        )

        rocm_aiter_ops.flash_attn_varlen_func(
            q=query,
            k=key_fetched,
            v=value_fetched,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=swa_cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=swa_max_seqlens,
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            return_lse=False,
            out=output,
            sink_ptr=self.sinks,
        )

    def extend_forward(
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
        min_seqlen_q: int,
        max_seqlen_k: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        prequantized_qkv: PrequantizedQKV | None = None,
    ):
        if self.sliding_window[0] != -1:
            self.extend_for_sliding_window(
                attn_metadata,
                query,
                key_cache,
                value_cache,
                output,
                cu_seqlens_q,
                max_seqlen_q,
                block_table,
                k_scale,
                v_scale,
            )
            return
        attention_query = (
            prequantized_qkv.query if prequantized_qkv is not None else query
        )
        attention_key = prequantized_qkv.key if prequantized_qkv is not None else key
        attention_value = (
            prequantized_qkv.value if prequantized_qkv is not None else value
        )
        extend_metadata = attn_metadata.extend_metadata
        assert extend_metadata is not None
        extend_starts = extend_metadata.query_start_loc_host or cu_seqlens_q.tolist()
        if prequantized_qkv is not None:
            out, lse = _grouped_fp8_varlen_attention(
                q=attention_query,
                k=attention_key,
                v=attention_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                q_starts=extend_starts,
                k_starts=extend_starts,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                min_seqlen_q=min_seqlen_q,
                q_descale=prequantized_qkv.query_descale,
                k_descale=prequantized_qkv.key_descale,
                v_descale=prequantized_qkv.value_descale,
                out_dtype=query.dtype,
                return_lse=True,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                sink_ptr=self.sinks,
            )
        else:
            out, lse = rocm_aiter_ops.flash_attn_varlen_func(
                q=attention_query,
                k=attention_key,
                v=attention_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                min_seqlen_q=min_seqlen_q,
                return_lse=True,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                sink_ptr=self.sinks,
            )
        assert attn_metadata.extend_metadata is not None
        chunk_context_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        num_chunks = chunk_context_metadata.num_chunks
        workspace = chunk_context_metadata.workspace
        cu_seqlens_kv = chunk_context_metadata.cu_seq_lens_chunk
        max_seqlens = chunk_context_metadata.max_seq_lens
        chunk_starts = chunk_context_metadata.chunk_starts
        token_to_batch = chunk_context_metadata.token_to_batch
        total_token_per_batch = chunk_context_metadata.total_token_per_batch
        key_fetched, value_fetched = workspace[0], workspace[1]
        if (
            prequantized_qkv is None
            and key_fetched.dtype == current_platform.fp8_dtype()
        ):
            raise ValueError(
                "The FP8 context workspace needs prequantized Q/K/V and descales"
            )
        fp8_context_gather = (
            prequantized_qkv is not None
            and key_fetched.dtype == current_platform.fp8_dtype()
            and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
        )
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
                dequant=(
                    is_quantized_kv_cache(self.kv_cache_dtype)
                    and not fp8_context_gather
                ),
                kv_cache_layout="SHUFFLE"
                if rocm_aiter_ops.is_shuffle_kv_cache_enabled()
                else "NHD",
                total_tokens=total_token_per_batch[chunk_idx],
            )

            if fp8_context_gather:
                assert prequantized_qkv is not None
                chunk_starts_host = (
                    chunk_context_metadata.cu_seq_lens_chunk_host[chunk_idx]
                    if chunk_context_metadata.cu_seq_lens_chunk_host is not None
                    else cu_seqlens_kv[chunk_idx].tolist()
                )
                suf_out, suf_lse = _grouped_fp8_varlen_attention(
                    q=prequantized_qkv.query,
                    k=key_fetched,
                    v=value_fetched,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_kv[chunk_idx],
                    q_starts=extend_starts,
                    k_starts=chunk_starts_host,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlens[chunk_idx],
                    min_seqlen_q=min_seqlen_q,
                    q_descale=prequantized_qkv.query_descale,
                    k_descale=k_scale,
                    v_descale=v_scale,
                    out_dtype=query.dtype,
                    return_lse=True,
                    softmax_scale=self.scale,
                    causal=False,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    sink_ptr=self.sinks,
                )
            else:
                suf_out, suf_lse = rocm_aiter_ops.flash_attn_varlen_func(
                    q=query,
                    k=key_fetched,
                    v=value_fetched,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_kv[chunk_idx],
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlens[chunk_idx],
                    min_seqlen_q=min_seqlen_q,
                    return_lse=True,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=False,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    sink_ptr=self.sinks,
                )
            if chunked_output is None:
                chunked_output = suf_out
                chunked_lse = suf_lse
            else:
                tmp_output = torch.empty_like(out)
                tmp_lse = torch.empty_like(lse)
                merge_attn_states(
                    output=tmp_output,
                    output_lse=tmp_lse,
                    prefix_output=chunked_output,
                    prefix_lse=chunked_lse,
                    suffix_output=suf_out,
                    suffix_lse=suf_lse,
                )
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
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
            prequantized_qkv=None,
        )

    def forward_with_prequantized_qkv(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor,
        prequantized_qkv: PrequantizedQKV,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.supports_prequantized_qkv_input:
            raise ValueError(
                "Prequantized QKV input requires gfx950, head size 256, "
                "an FP8 KV cache, full causal attention, and no sinks "
                "or soft cap"
            )
        return self._forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
            prequantized_qkv=prequantized_qkv,
        )

    def _forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        prequantized_qkv: PrequantizedQKV | None,
    ) -> torch.Tensor:
        """Forward pass with AiterFlashAttention.

        Args:
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization; not supported
                by this backend.
            output_block_scale: Block scale for fused output quantization;
                not supported by this backend.
            prequantized_qkv: Optional model-provided FP8 Q/K/V tensors and
                descales. Only prefill and extend slices are consumed; decode
                always uses the floating-point query and KV cache.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values

        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported "
                "for AiterFlashAttentionImpl"
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

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        # decode:extend:prefill
        query = query[:num_actual_tokens]
        if key is not None:
            key = key[:num_actual_tokens]
        if value is not None:
            value = value[:num_actual_tokens]
        if prequantized_qkv is not None:
            prequantized_qkv = PrequantizedQKV(
                query=prequantized_qkv.query[:num_actual_tokens],
                key=prequantized_qkv.key[:num_actual_tokens],
                value=prequantized_qkv.value[:num_actual_tokens],
                query_descale=prequantized_qkv.query_descale,
                key_descale=prequantized_qkv.key_descale,
                value_descale=prequantized_qkv.value_descale,
            )

        output_actual_tokens = output[:num_actual_tokens]

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_extends = attn_metadata.num_extends

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_extend_tokens = attn_metadata.num_extend_tokens
        if (self.kv_sharing_target_layer_name is not None or key is None) and (
            num_prefills > 0 or num_extends > 0
        ):
            # Shared layers project only Q. Fetch the new K/V tokens from the
            # target cache for the direct prefill and extend suffix kernels.
            shared = attn_metadata.kv_sharing_metadata
            assert shared is not None
            key, value = shared.workspace[:, :num_actual_tokens].unbind(0)
            cp_mha_gather_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                key=key[num_decode_tokens:],
                value=value[num_decode_tokens:],
                block_tables=attn_metadata.block_table[num_decodes:],
                k_scales=layer._k_scale,
                v_scales=layer._v_scale,
                cu_seqlens_kv=shared.query_start_loc,
                token_to_batch=shared.token_to_batch,
                seq_starts=shared.seq_starts,
                dequant=is_quantized_kv_cache(self.kv_cache_dtype),
                kv_cache_layout="SHUFFLE"
                if rocm_aiter_ops.is_shuffle_kv_cache_enabled()
                else "NHD",
                total_tokens=num_actual_tokens - num_decode_tokens,
            )
        # calculate for pure prefills
        if num_prefills > 0:
            assert attn_metadata.prefill_metadata is not None
            assert key is not None and value is not None

            prefill_token_start = num_decode_tokens + num_extend_tokens
            prefill_query = query[prefill_token_start:]
            prefill_key = key[prefill_token_start:]
            prefill_value = value[prefill_token_start:]
            q_descale = k_descale = v_descale = None
            if prequantized_qkv is not None:
                prefill_sequence_start = num_decodes + num_extends
                prefill_sequence_end = prefill_sequence_start + num_prefills
                prefill_prequantized_qkv = _slice_prequantized_qkv(
                    prequantized_qkv,
                    slice(prefill_token_start, None),
                    slice(prefill_sequence_start, prefill_sequence_end),
                )
                prefill_query = prefill_prequantized_qkv.query
                prefill_key = prefill_prequantized_qkv.key
                prefill_value = prefill_prequantized_qkv.value
                q_descale = prefill_prequantized_qkv.query_descale
                k_descale = prefill_prequantized_qkv.key_descale
                v_descale = prefill_prequantized_qkv.value_descale

            prefill_metadata = attn_metadata.prefill_metadata
            prefill_output = output_actual_tokens[
                num_decode_tokens + num_extend_tokens :
            ]
            if prequantized_qkv is not None:
                assert q_descale is not None
                assert k_descale is not None and v_descale is not None
                starts = (
                    prefill_metadata.query_start_loc_host
                    or prefill_metadata.query_start_loc.tolist()
                )
                _grouped_fp8_varlen_attention(
                    q=prefill_query,
                    k=prefill_key,
                    v=prefill_value,
                    cu_seqlens_q=prefill_metadata.query_start_loc,
                    cu_seqlens_k=prefill_metadata.query_start_loc,
                    q_starts=starts,
                    k_starts=starts,
                    max_seqlen_q=prefill_metadata.max_query_len,
                    max_seqlen_k=prefill_metadata.max_seq_len,
                    min_seqlen_q=1,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    out_dtype=output.dtype,
                    out=prefill_output,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    sink_ptr=self.sinks,
                )
            else:
                rocm_aiter_ops.flash_attn_varlen_func(
                    q=prefill_query,
                    k=prefill_key,
                    v=prefill_value,
                    cu_seqlens_q=prefill_metadata.query_start_loc,
                    cu_seqlens_k=prefill_metadata.query_start_loc,
                    max_seqlen_q=prefill_metadata.max_query_len,
                    max_seqlen_k=prefill_metadata.max_seq_len,
                    min_seqlen_q=1,
                    out=prefill_output,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    sink_ptr=self.sinks,
                )

        # calculate for extends
        if num_extends > 0:
            assert attn_metadata.extend_metadata is not None
            assert key is not None and value is not None
            extend_tokens_slice = slice(
                num_decode_tokens, num_decode_tokens + num_extend_tokens
            )
            extend_queries = query[extend_tokens_slice]
            extend_keys = key[extend_tokens_slice]
            extend_values = value[extend_tokens_slice]
            extend_outputs = output[extend_tokens_slice]
            k_scale = layer._k_scale
            v_scale = layer._v_scale
            extend_prequantized_qkv = None
            if prequantized_qkv is not None:
                extend_sequence_slice = slice(num_decodes, num_decodes + num_extends)
                extend_prequantized_qkv = _slice_prequantized_qkv(
                    prequantized_qkv,
                    extend_tokens_slice,
                    extend_sequence_slice,
                )
            self.extend_forward(
                attn_metadata=attn_metadata,
                query=extend_queries,
                key=extend_keys,
                value=extend_values,
                key_cache=key_cache,
                value_cache=value_cache,
                output=extend_outputs,
                cu_seqlens_q=attn_metadata.extend_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.extend_metadata.max_query_len,
                min_seqlen_q=1,
                max_seqlen_k=attn_metadata.extend_metadata.max_seq_len,
                block_table=attn_metadata.block_table[
                    num_decodes : num_decodes + num_extends
                ],
                slot_mapping=attn_metadata.slot_mapping[
                    num_decodes : num_decodes + num_extends
                ],
                k_scale=k_scale,
                v_scale=v_scale,
                prequantized_qkv=extend_prequantized_qkv,
            )

        # calculate for decodes
        if num_decodes > 0:
            assert attn_metadata.decode_metadata is not None
            decode_max_query_len = attn_metadata.decode_metadata.max_query_len

            # Use unified_attention for the decodes the paged kernels can't
            # take: sliding window, sinks, or a multi-token batch
            # (pa_fwd_asm and paged_attention_v1 don't support sinks).
            # Single-token SWA decode on the trimmed block table is ~2x
            # faster on the paged kernel than unified_attention.
            decode_md = attn_metadata.decode_metadata
            use_swa_paged = (
                self.sliding_window[0] != -1
                and decode_max_query_len == 1
                and self.sinks is None
                and self.alibi_slopes is None
                and decode_md.swa_block_table is not None
                and self.head_size >= 64
                and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
            )
            if (
                (self.sliding_window[0] != -1 and not use_swa_paged)
                or decode_max_query_len > 1
                or self.sinks is not None
            ):
                k_descale, v_descale = self._get_kv_cache_descales(layer, num_decodes)
                assert not rocm_aiter_ops.is_shuffle_kv_cache_enabled(), (
                    "Shuffle KV cache layout is not supported with sliding "
                    "window, sinks, or speculative decoding (multi-token decode)."
                )
                if not attn_metadata.causal:
                    from aiter.ops.triton.attention.mha_v3 import (
                        flash_attn_with_kvcache,
                    )

                    decode_query = query[:num_decode_tokens].reshape(
                        num_decodes,
                        decode_max_query_len,
                        query.shape[1],
                        query.shape[2],
                    )
                    decode_out = flash_attn_with_kvcache(
                        q=decode_query,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        cache_seqlens=attn_metadata.seq_lens[:num_decodes],
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        window_size=self.sliding_window,
                        softcap=self.logits_soft_cap,
                        q_descale=None,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        page_table=attn_metadata.block_table[:num_decodes],
                    )
                    output[:num_decode_tokens].copy_(
                        decode_out.reshape(
                            num_decode_tokens,
                            query.shape[1],
                            query.shape[2],
                        )
                    )
                else:
                    # Non-uniform query lengths can appear in real serving
                    # traffic (e.g. mixed datasets). Fall back to varlen
                    # unified_attention instead of asserting.
                    from aiter.ops.triton.unified_attention import (
                        unified_attention,
                    )

                    unified_attention(
                        q=query[:num_decode_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_decode_tokens],
                        cu_seqlens_q=attn_metadata.query_start_loc[: num_decodes + 1],
                        max_seqlen_q=decode_max_query_len,
                        seqused_k=attn_metadata.seq_lens[:num_decodes],
                        max_seqlen_k=attn_metadata.max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=attn_metadata.block_table[:num_decodes],
                        softcap=self.logits_soft_cap,
                        q_descale=None,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        sinks=self.sinks,
                    )
                return

            # The ll4mi kernel in paged_attention_v1 requires
            # HEAD_SIZE >= 16 * NWARPS (= 64 on ROCm with NWARPS=4).
            # For smaller head sizes or sliding window attention,
            # fall back to the unified_attention triton kernel which
            # handles both correctly.
            _MIN_HEAD_SIZE_FOR_LL4MI = 64
            use_unified_attention = self.head_size < _MIN_HEAD_SIZE_FOR_LL4MI

            if use_unified_attention:
                k_descale, v_descale = self._get_kv_cache_descales(layer, num_decodes)
                assert not rocm_aiter_ops.is_shuffle_kv_cache_enabled(), (
                    "unified_attention fallback with shuffle layout "
                    "is not supported yet."
                )
                from aiter.ops.triton.unified_attention import (
                    unified_attention,
                )

                decode_cu_seqlens_q = attn_metadata.query_start_loc[: num_decodes + 1]
                unified_attention(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=decode_cu_seqlens_q,
                    max_seqlen_q=1,
                    seqused_k=attn_metadata.seq_lens[:num_decodes],
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=attn_metadata.block_table[:num_decodes],
                    softcap=self.logits_soft_cap,
                    q_descale=None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            elif rocm_aiter_ops.is_shuffle_kv_cache_enabled():
                _, num_heads, head_size = query.shape
                num_blocks, block_size, num_kv_heads, _ = key_cache.shape
                x = 16 // key_cache.element_size()
                new_key_cache = key_cache.reshape(
                    num_blocks, num_kv_heads, head_size // x, block_size, x
                )
                new_value_cache = value_cache.reshape(
                    num_blocks, num_kv_heads, block_size // x, head_size, x
                )

                # This kernel derives block addresses arithmetically, so it
                # only reads the right bytes when each side is a dense
                # plane of pages.
                assert new_key_cache.stride(0) == (
                    num_kv_heads * head_size * block_size
                ), (
                    "paged_attention_common needs the K/V sides as dense "
                    "planes; the resolved KV cache layout interleaves them "
                    "within a block."
                )
                num_seqs = attn_metadata.seq_lens.shape[0]
                max_num_partitions = (
                    attn_metadata.max_seq_len + _PARTITION_SIZE_ROCM - 1
                ) // _PARTITION_SIZE_ROCM
                tmp_out = torch.empty(
                    (num_seqs, num_heads, max_num_partitions, head_size),
                    dtype=query.dtype,
                    device=query.device,
                )
                exp_sums = torch.empty(
                    (num_seqs, num_heads, max_num_partitions),
                    dtype=torch.float32,
                    device=query.device,
                )
                max_logits = torch.empty_like(exp_sums)
                # asm indexes this layer's dense scale buffer from build();
                # the HIP kernel reads the per-tensor scale directly.
                k_qscale_asm = v_qscale_asm = None
                if attn_metadata.k_scale is not None:
                    assert attn_metadata.v_scale is not None
                    k_qscale_asm = attn_metadata.k_scale[layer.layer_name]
                    v_qscale_asm = attn_metadata.v_scale[layer.layer_name]
                rocm_aiter_ops.paged_attention_common(
                    Q=query[:num_decode_tokens],
                    K=new_key_cache,
                    V=new_value_cache,
                    tmp_out=tmp_out,
                    max_logits=max_logits,
                    exp_sums=exp_sums,
                    max_seq_len=attn_metadata.max_seq_len,
                    block_tables=attn_metadata.block_table[:num_decodes],
                    context_lens=attn_metadata.seq_lens[:num_decodes],
                    block_tables_stride0=attn_metadata.block_table[:num_decodes].stride(
                        0
                    ),
                    scale=self.scale,
                    K_QScale_hip=layer._k_scale,
                    V_QScale_hip=layer._v_scale,
                    K_QScale_asm=k_qscale_asm,
                    V_QScale_asm=v_qscale_asm,
                    out_=output[:num_decode_tokens],
                    kv_cache_dtype=self.kv_cache_dtype,
                )
            else:
                _, num_heads, head_size = query.shape
                nbytes_per_qo_elem = torch.finfo(query.dtype).bits // 8
                num_seqs = attn_metadata.seq_lens.shape[0]
                if use_swa_paged:
                    decode_block_table = decode_md.swa_block_table
                    decode_seq_lens = decode_md.swa_seq_lens
                    decode_max_seq_len = decode_md.swa_max_seq_len
                else:
                    decode_block_table = attn_metadata.block_table[:num_decodes]
                    decode_seq_lens = attn_metadata.seq_lens[:num_decodes]
                    decode_max_seq_len = attn_metadata.max_seq_len
                max_num_partitions = (
                    decode_max_seq_len + _PARTITION_SIZE_ROCM - 1
                ) // _PARTITION_SIZE_ROCM

                workspace_buffer = torch.empty(
                    (num_seqs * num_heads * max_num_partitions * head_size)
                    * nbytes_per_qo_elem
                    + 2 * (num_seqs * num_heads * max_num_partitions) * 4,
                    dtype=torch.uint8,
                    device=output.device,
                )

                # import so that aiter register the op to the namespace of
                # torch.ops.aiter
                import aiter  # noqa: F401

                torch.ops.aiter.paged_attention_v1(
                    output[:num_decode_tokens],
                    workspace_buffer,
                    query[:num_decode_tokens],
                    key_cache,
                    value_cache,
                    self.scale,
                    decode_block_table,
                    attn_metadata.query_start_loc[:num_decodes],
                    decode_seq_lens,
                    decode_max_seq_len,
                    self.alibi_slopes,
                    self.kv_cache_dtype,
                    "NHD",
                    self.logits_soft_cap,
                    layer._k_scale,
                    layer._v_scale,
                    None,
                    _PARTITION_SIZE_ROCM,
                    1,
                    self.sliding_window[0] + 1,
                )

        return output

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            # (B, 2, N, H*hs) -> one (B, N, H, hs) per side, which is what the
            # shuffle read/write kernels reinterpret in place. The heads are
            # folded into the content dim by customize_spec.
            key_cache, value_cache = kv_cache.unflatten(
                -1, (self.num_kv_heads, self.head_size)
            ).unbind(1)
            return key_cache, value_cache
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(kv_cache)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())
        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping
        # is not padded. However, we don't need to do
        # key[:num_actual_tokens] and value[:num_actual_tokens] because
        # the reshape_and_cache_flash op uses the slot_mapping's shape
        # to determine the number of actual tokens.
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            k_scale = layer._k_scale
            v_scale = layer._v_scale
            assert k_scale is not None and v_scale is not None, (
                "k_scale and v_scale are required for shuffled update"
            )
            reshape_and_cache_shuffle_triton(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )
        else:
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

    def fused_rope_kvcache_supported(self):
        # Only support fusion when shuffle KV cache layout is not used;
        # shuffle layout uses a different cache update path.
        return (
            rocm_aiter_ops.is_enabled()
            and not rocm_aiter_ops.is_shuffle_kv_cache_enabled()
        )

    def fused_qk_norm_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def do_qk_norm_rope_kvcache_update(
        self,
        layer: AttentionLayer,
        qkv: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        rocm_aiter_ops.do_qk_norm_rope_kvcache_update(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_kv_heads,
            head_dim=self.head_size,
            is_neox=is_neox,
            rms_norm_eps=rms_norm_eps,
            q_out=q_out,
            k_out=k_out,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=layer_slot_mapping,
            k_scale=layer._k_scale_cpu,
            v_scale=layer._v_scale_cpu,
            kv_cache_dtype=self.kv_cache_dtype,
            use_shuffle_layout=rocm_aiter_ops.is_shuffle_kv_cache_enabled(),
            return_kv=True,
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
