# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from typing import cast

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
    pack_fp8_and_gather_k_cache,
)
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import _ON_GFX950
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    Q_MXFP8_ROPE_DIM,
    Q_MXFP8_ROW_BYTES,
    build_ragged_indices_from_dense,
    can_stage_fp8_sparse_prefill,
    fp8_sparse_prefill_available,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
    rocm_sparse_attn_prefill_fp8,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Fused q/kv RMSNorm + MXFP8 q quant (ROCm producer for the native CDNA4 GEMM)
# ---------------------------------------------------------------------------
@triton.jit
def _rocm_q_kv_norm_mxfp8_kernel(
    q,
    kv,
    qw,
    kvw,
    qo,
    kvo,
    scales,
    q_stride,
    kv_stride,
    s_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    """One program per (token, half). Half 0 norms+quantizes q, half 1 norms kv.

    The quantization arithmetic is a line-for-line copy of
    ``mxfp8_utils._mxfp8_e4m3_quantize_triton`` (block amax -> ceil(log2(amax /
    448)) + 127, clamped to [0, 254]; rescale by ``exp2(127 - sb)`` rather than
    dividing, so an all-zero block cannot produce 0/0 = NaN). Combined with the
    bf16 round below, the (values, scales) this writes are BITWISE equal to
    ``mxfp8_e4m3_quantize(fused_q_kv_rmsnorm(...)[0])``, which is what makes
    turning this on a pure fusion rather than a numerics change.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if tl.program_id(1) == 0:
        w = tl.load(qw + cols, cols < Q_SIZE, 0.0).to(tl.float32)
        v = tl.load(q + row * q_stride + cols, cols < Q_SIZE, 0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(v * v, 0) / Q_SIZE + eps)
        y = v * rrms * w
        # Preserve the rounding boundary of the bf16 tensor this used to
        # materialize: quantize from the bf16-rounded value, not from fp32.
        y = y.to(q.dtype.element_ty).to(tl.float32)
        grouped = tl.reshape(y, (BLOCK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(grouped), axis=1), TINY)
        sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
        sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
        rescale = tl.exp2(127.0 - sb)
        xq = tl.reshape(grouped * rescale[:, None], (BLOCK,))
        tl.store(qo + row * Q_SIZE + cols, xq.to(qo.dtype.element_ty), cols < Q_SIZE)
        groups = tl.arange(0, BLOCK // 32)
        tl.store(scales + row * s_stride + groups, sb.to(tl.uint8),
                 groups < Q_SIZE // 32)
    else:
        w = tl.load(kvw + cols, cols < KV_SIZE, 0.0).to(tl.float32)
        v = tl.load(kv + row * kv_stride + cols, cols < KV_SIZE, 0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(v * v, 0) / KV_SIZE + eps)
        tl.store(kvo + row * KV_SIZE + cols, v * rrms * w, cols < KV_SIZE)


def _rocm_q_kv_rmsnorm_mxfp8(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
):
    """Normalize q/kv and emit q as MXFP8 with ROW-MAJOR E8M0 scales.

    The ROCm counterpart of ``common/ops/query_quant.fused_q_kv_rmsnorm_quant``.
    That one is CUDA-only by construction -- it writes FlashInfer's F8_128x4
    swizzled scales, and ``can_fuse_query_quant`` opens with
    ``if not current_platform.is_cuda(): return False`` because ``QuantKey``
    cannot express the layout difference. This one writes the ``[M, K/32]``
    row-major layout ``RocmDotScaledMxfp8LinearKernel`` reads, so the gate is
    ``_wq_b_uses_rocm_native_mxfp8`` (a kernel TYPE test) instead.
    """
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    assert qr.ndim == kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == kv.stride(-1) == 1
    assert qr.shape[1] % 32 == 0
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    qo = torch.empty((tokens, q_size), dtype=torch.float8_e4m3fn, device=qr.device)
    kvo = torch.empty_like(kv)
    scales = torch.empty((tokens, q_size // 32), dtype=torch.uint8, device=qr.device)
    block = triton.next_power_of_2(max(q_size, kv_size))
    _rocm_q_kv_norm_mxfp8_kernel[(tokens, 2)](
        qr,
        kv,
        q_weight,
        kv_weight,
        qo,
        kvo,
        scales,
        qr.stride(0),
        kv.stride(0),
        scales.stride(0),
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        BLOCK=block,
        FP8_MAX=float(torch.finfo(torch.float8_e4m3fn).max),
        TINY=float(torch.finfo(torch.float32).tiny),
        num_warps=8 if block >= 2048 else 4,
    )
    return QuantizedActivation(qo, scales, qr.dtype, qr.shape, kMxfp8Dynamic), kvo


def _fp8_prefill_workspace_requests(
    chunk: int, m: int, max_batched_tokens: int, num_heads: int
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    """The staging buffers the FP8 sparse-prefill route asks the arena for.

    Kept in one place because the warmup step has to reserve exactly what the
    real step will request: the arena is locked after warmup, and an under-
    reservation there becomes a hard failure in the serving path.
    """
    return (
        ((chunk, m, Q_MXFP8_ROW_BYTES), torch.uint8),
        ((chunk, m, Q_MXFP8_ROPE_DIM), torch.bfloat16),
        ((max_batched_tokens, num_heads, Q_MXFP8_ROW_BYTES), torch.uint8),
        ((max_batched_tokens, num_heads, Q_MXFP8_ROPE_DIM), torch.bfloat16),
    )


def _trust_dsv4_extra_cache_nan_free(
    kv_cache_dtype: str,
    has_kv_transfer: bool,
    has_extra_cache: bool,
) -> bool:
    return (
        _ON_GFX950
        and kv_cache_dtype == "fp8_ds_mla"
        and not has_kv_transfer
        and has_extra_cache
    )


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


def apply_pre_quantized_block_scaled_mm(
    linear: torch.nn.Module,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled fp8 GEMM on pre-quantized activations.

    The fused q/kv norm kernel writes fp8 qr + per-1x128 scales; this
    drives the linear's block-scaled GEMM directly with them, bypassing
    apply_weights which would re-quantize the fp8 input. Only valid for
    the wq_b-style column/replicated linears: their output is the local
    TP shard, so no all-reduce is needed.
    """
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        FP8BlockParams,
    )

    params = FP8BlockParams.from_layer(linear)
    weight_scale = (
        params.weight_scale
        if params.weight_scale_inv is None
        else params.weight_scale_inv
    )
    kernel = linear.quant_method.fp8_linear
    out = kernel.apply_block_scaled_mm(
        A=x_fp8, B=params.weight, As=x_scale, Bs=weight_scale
    )
    return out.to(dtype=kernel.config.out_dtype)


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    # Lane count for the sliding-window store. ``tl.arange`` needs a
    # power-of-two extent; 0 means "WINDOW_SIZE is already one, use it", which
    # keeps every pre-existing caller of this kernel compiling unchanged.
    PADDED_WINDOW: tl.constexpr = 0,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        # ``swa_len <= WINDOW_SIZE <= PADDED_WINDOW``, so padding the lane count
        # and masking on ``swa_len`` is exact for any window, not just the 128
        # this checkpoint happens to use.
        if PADDED_WINDOW > 0:
            swa_offset = tl.arange(0, PADDED_WINDOW)
        else:
            swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def _combine_topk_swa_indices_reference(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager-Torch reference for :func:`combine_topk_swa_indices`.

    This was the shipped ROCm path.  It is retained, unmodified, purely as the
    correctness oracle that ``eval/eval_prefill_index_combine_fused.py`` holds
    the fused kernel to; nothing in the model calls it.

    It is NOT a fallback.  Two of its steps (``rows[swa_mask]`` and
    ``swa_values[swa_mask]``) are boolean-mask selections, which call
    ``nonzero()`` and therefore synchronize the stream -- once per sparse
    attention layer per prefill chunk, 43 times per prefill step, on the
    critical path.
    """
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    # query_start_loc may have a non-zero base for a narrowed mixed batch.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=seq_lens.device), query_lens
    )
    query_starts = query_start_loc[:-1] - query_start_loc[0]
    token_offsets = torch.arange(num_tokens, device=seq_lens.device) - (
        torch.repeat_interleave(query_starts, query_lens)
    )
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets

    logical_topk_width = min(topk, topk_indices.shape[1])
    topk_lens = torch.minimum(
        (positions + 1) // compress_ratio,
        torch.full_like(positions, logical_topk_width),
    ).clamp_min(0)
    topk_offsets = torch.arange(logical_topk_width, device=seq_lens.device)
    topk_mask = topk_offsets[None, :] < topk_lens[:, None]
    topk_values = topk_indices[:, :logical_topk_width].to(torch.int32)
    topk_valid = topk_mask & (topk_values >= 0) & (topk_values < N)
    combined_indices[:, :logical_topk_width] = torch.where(
        topk_valid,
        topk_values + (M * req_ids).to(torch.int32)[:, None],
        -1,
    )

    swa_lens = torch.minimum(
        positions + 1, torch.full_like(positions, window_size)
    ).clamp_min(0)
    swa_offsets = torch.arange(window_size, device=seq_lens.device)
    swa_mask = swa_offsets[None, :] < swa_lens[:, None]
    swa_columns = topk_lens[:, None] + swa_offsets[None, :]
    gather_starts = seq_lens - gather_lens
    swa_values = (
        M * req_ids[:, None]
        + N
        + swa_offsets[None, :]
        + positions[:, None]
        - swa_lens[:, None]
        + 1
        - gather_starts[req_ids, None]
    ).to(torch.int32)
    rows = torch.arange(num_tokens, device=seq_lens.device)[:, None].expand_as(
        swa_columns
    )
    flat_dst = rows[swa_mask] * combined_topk + swa_columns[swa_mask]
    combined_indices.view(-1).index_copy_(0, flat_dst, swa_values[swa_mask])
    combined_lens.copy_((topk_lens + swa_lens).to(torch.int32))
    return combined_indices, combined_lens


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine compressed-attention and sliding-window indices in one kernel.

    One Triton launch, no host synchronization, bitwise-identical to
    :func:`_combine_topk_swa_indices_reference`.

    ``compress_ratio`` is clamped to 1 only to keep the ``//`` a legal constexpr
    on the five SWA-only layers, where ``compress_ratios[layer] == 0``.  Those
    layers also pass ``topk == 0``, so ``topk_len`` is identically zero and the
    divisor is never observable -- exactly as in the reference, where the same
    division by zero is masked by ``minimum(..., 0)``.
    """
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk_width = topk_indices.shape[1]
    # The reference bounds the live top-k prefix by the tensor it was handed,
    # not by the requested ``topk``; match that or the two disagree whenever a
    # caller passes a wider ``topk`` than the buffer it published.
    logical_topk_width = min(topk, topk_width)

    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    # The kernel writes only each token's live prefix; the pad stays -1, which
    # is the contract every consumer of ``combined_indices`` relies on.
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0:
        return combined_indices, combined_lens

    num_prefills = seq_lens.shape[0]
    _combine_topk_swa_indices_kernel[(num_prefills, 128)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=logical_topk_width,
        COMPRESS_RATIO=max(1, compress_ratio),
        WINDOW_SIZE=window_size,
        TOPK_WIDTH=topk_width,
        PADDED_TOP_K=max(1, triton.next_power_of_2(logical_topk_width)),
        PADDED_WINDOW=max(1, triton.next_power_of_2(window_size)),
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    source_entries = ragged_indices.numel()
    if source_entries > 0:
        ragged_out[:source_entries].copy_(ragged_indices, non_blocking=True)
    if _ON_GFX950:
        # Preserve the graph-stable base pointer while exposing source capacity
        # to the sync-free split selector; indptr still carries the true NNZ.
        ragged_out = ragged_out[: max(source_entries, 1)]
    return ragged_out, indptr_out


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    # Keep fused multi-step decode disabled until update_draft_decode_metadata()
    # also refreshes the ROCm-specific ragged SWA indices and indptrs.
    supports_draft_decode_metadata_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(
                    base.num_decode_tokens, base.decode_swa_width
                ),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                base.decode_swa_width,
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4SparseMLAMetadataBuilder]:
        return DeepseekV4SparseMLAMetadataBuilder


class DeepseekV41ROCMAiterSparseSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterSparseSWAMetadataBuilder"]:
        return DeepseekV4ROCMAiterSparseSWAMetadataBuilder


class DeepseekV41ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4.1."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend
    swa_backend_cls = DeepseekV41ROCMAiterSparseSWABackend

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        # CUDA executes WO_A with a quantized grouped-BMM kernel. ROCm now runs
        # the equivalent AITER FP8 chain (inverse_rope_group_quant +
        # batched_gemm_a8w8_mxscale) instead of dequantizing to BF16, but that
        # chain reads the *plain* MXFP8 linear layout and re-expresses the
        # scales itself, so post-load processing must still leave the weight
        # unshuffled rather than request the CUDA-only BMM kernel.
        self.wo_a.is_bmm = False
        self._has_kv_transfer = vllm_config.kv_transfer_config is not None
        # Activation dtype for the inverse-RoPE cos/sin caches, taken from the
        # model config rather than guessed at first use, so the load-time prime
        # produces the exact dtype the step path will ask for.
        self._woa_act_dtype = vllm_config.model_config.dtype
        # Block scale for the preshuffled weight; None = not preshuffled.
        self._wqa_wkv_scale: torch.Tensor | None = None
        self._wo_b_scale: torch.Tensor | None = None
        self._fused_compressor_weight: torch.Tensor | None
        self.register_buffer("_fused_compressor_weight", None, persistent=False)
        self._fused_compressor_split_sizes: tuple[int, int] | None = None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def prepare_wo_a_fp8(self) -> bool:
        """A1: re-express the WO_A scales for the AITER FP8 chain, at load time.

        Deliberately separate from ``prepare_attn_preshuffle`` below, which this
        model never calls (see ``process_weights_after_loading``): that routine
        also reshuffles ``fused_wqa_wkv`` and ``wo_b``, which are different
        GEMMs with their own evidence requirements. Only the WO_A rearrangement
        is measured and landed here.
        """
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            rocm_prepare_inv_rope_woa_fp8,
        )

        return rocm_prepare_inv_rope_woa_fp8(
            self.rotary_emb,
            self.wo_a,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self._woa_act_dtype,
        )

    def prepare_attn_preshuffle(self) -> None:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )
        from vllm.model_executor.utils import replace_parameter

        def _prep(linear) -> torch.Tensor | None:
            w = getattr(linear, "weight", None)
            if w is None or w.dim() != 2:
                return None
            # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
            if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
                return None
            ws = getattr(linear, "weight_scale_inv", None)  # per-block scale
            if ws is None:
                return None
            if ws.dtype == torch.float8_e8m0fnu:
                ws = _upcast_e8m0_to_fp32(ws).contiguous()
            # Shuffle the weight in place (single weight, no unshuffled copy).
            replace_parameter(
                linear,
                "weight",
                rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
            )
            return ws

        self._wqa_wkv_scale = _prep(self.fused_wqa_wkv)
        self._wo_b_scale = _prep(self.wo_b)

    def prepare_compressor_gemm_fusion(self) -> bool:
        # V4.1 derives index keys from the source compressor's emitted latent
        # and has no nested ``indexer.compressor``.  Keep the projections
        # separate and use the shared linear/PyTorch correctness path.
        return False

    def _bpre_attn_gemm(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        x: torch.Tensor,
        reduce_tp: bool,
    ) -> torch.Tensor:
        from vllm._aiter_ops import rocm_aiter_ops

        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8, weight, x_scale, scale, output_dtype=x.dtype
        )
        if reduce_tp and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._wqa_wkv_scale is not None and hidden_states.dim() == 2:
            return self._bpre_attn_gemm(
                self.fused_wqa_wkv.weight, self._wqa_wkv_scale, hidden_states, False
            )
        return super()._fused_wqa_wkv_gemm(hidden_states)

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return super()._run_parallel_input_projections(hidden_states)

    @functools.cached_property
    def _wq_b_uses_rocm_native_mxfp8(self) -> bool:
        """True when both wq_b GEMMs are the native CDNA4 MXFP8 kernel.

        This is the gate for ``_rocm_q_kv_rmsnorm_mxfp8`` below. It has to be
        a TYPE test, not just an ``input_quant_key`` test: QuantKey does not
        encode scale layout, and ``kMxfp8Dynamic`` is also what FlashInfer's
        kernels advertise for F8_128x4-SWIZZLED scales. Our producer writes
        row-major ``[M, K/32]`` scales, so it may only feed the ROCm kernel.
        Same discipline as ``can_fuse_query_quant`` on the CUDA side.

        Cached: the linear kernels are fixed once the model is built.
        """
        from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
            RocmDotScaledMxfp8LinearKernel,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp8Dynamic,
        )

        linears = [self.wq_b]
        if self.indexer is not None:
            linears.append(self.indexer.wq_b)
        for linear in linears:
            kernel = getattr(getattr(linear, "quant_method", None), "kernel", None)
            if type(kernel) is not RocmDotScaledMxfp8LinearKernel:
                logger.info_once(
                    "DSv4.1 ROCm: fused q/kv norm+MXFP8-quant DECLINED -- "
                    "wq_b kernel is %s, not RocmDotScaledMxfp8LinearKernel",
                    type(kernel).__name__,
                )
                return False
            # The layer must actually be advertising the key, i.e.
            # expose_input_quant_key ran and the consumer will call
            # as_quantized_activation. If it is not advertised, handing it a
            # QuantizedActivation would raise, so decline instead.
            if getattr(linear, "input_quant_key", None) != kMxfp8Dynamic:
                logger.info_once(
                    "DSv4.1 ROCm: fused q/kv norm+MXFP8-quant DECLINED -- "
                    "wq_b does not advertise input_quant_key=%s (got %s)",
                    kMxfp8Dynamic,
                    getattr(linear, "input_quant_key", None),
                )
                return False
        logger.info_once(
            "DSv4.1 ROCm: fused q/kv RMSNorm + MXFP8 q quant ENGAGED; wq_b and "
            "indexer.wq_b share one pre-quantized activation."
        )
        return True

    @functools.cached_property
    def _wq_b_uses_aiter_block_scaled(self) -> bool:
        """True when both wq_b GEMMs run the aiter block-scaled fp8 kernel.

        Cached: the linear kernels and the aiter env gates are fixed once
        the model is built, so this is evaluated at the first forward
        only.

        The fused norm+quant path is only valid if the quant and GEMM it
        replaces are exactly the aiter ones; otherwise fall back to the
        shared path.
        """
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.kernels.linear.scaled_mm import (
            Fp8BlockScaledMMLinearKernel,
        )

        if not rocm_aiter_ops.is_linear_fp8_enabled():
            return False

        linears = [self.wq_b]
        if self.indexer is not None:
            linears.append(self.indexer.wq_b)
        for linear in linears:
            kernel = getattr(getattr(linear, "quant_method", None), "fp8_linear", None)
            if not isinstance(kernel, Fp8BlockScaledMMLinearKernel):
                return False
        return True

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Fuse q/kv RMSNorm + q quant into one kernel, for both wq_b GEMMs.

        The shared path norms q and kv in one triton kernel and the wq_b
        linears then re-read the bf16 qr to quantize it -- TWICE, because
        ``wq_b`` and ``indexer.wq_b`` each quantize the same tensor. Fusing
        norm and quant writes the quantized pair once and both GEMMs consume
        it. kv stays bf16: the fused insert kernel RoPE/quantizes it itself.

        Two producers, picked by what the consumers actually are:

        * MXFP8 1x32 E8M0 (this checkpoint: ``weight_block_size=[32,32]``,
          ``scale_fmt=ue8m0``) -> ``_rocm_q_kv_rmsnorm_mxfp8``, which emits a
          ``QuantizedActivation`` for the native CDNA4 ``tl.dot_scaled``
          kernel. Bitwise-identical to the path it replaces.
        * block-scaled FP8 1x128 -> the aiter kernel below.

        Falls back to the shared path when neither consumer matches.
        """
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        if not (qr.dim() == 2 and qr.shape[0] > 0):
            return super()._split_qkv_and_norm(qr_kv)

        if self.q_lora_rank % 32 == 0 and self._wq_b_uses_rocm_native_mxfp8:
            qr_quant, kv_out = _rocm_q_kv_rmsnorm_mxfp8(
                qr, kv, self.q_norm.weight.data, self.kv_norm.weight.data, self.eps
            )
            return qr_quant, None, kv_out

        if not (self.q_lora_rank % 128 == 0 and self._wq_b_uses_aiter_block_scaled):
            return super()._split_qkv_and_norm(qr_kv)

        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.fused_qk_rmsnorm_group_quant(
            q=qr,
            q_weight=self.q_norm.weight.data,
            q_epsilon=self.eps,
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            kv_epsilon=self.eps,
            group_size=128,
            transpose_scale=False,
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Stage A (inverse RoPE + WO_A) via the AITER FP8 chain, with the BF16
        # einsum as a named fallback; then wo_b.
        # NOTE: the ``_wo_b_scale`` branch below is currently unreachable in
        # this model -- it is only set by ``prepare_attn_preshuffle``, which
        # V4.1's ``process_weights_after_loading`` never calls (V4's does).
        # Left alone deliberately: wiring it up is a separate GEMM change that
        # needs its own evidence, not a rider on this one.
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        zf = z.flatten(1)
        if self._wo_b_scale is not None and zf.dim() == 2:
            result = self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        else:
            result = self.wo_b(zf)
        return result

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio == 0
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            # ...and, if this layer *could* ever take the FP8 route, its
            # buffers too.  Deliberately gated on structural capability
            # alone, not on the step-time profitability bound: warmup has no
            # context lengths to key that bound on, and the arena is locked
            # after warmup, so anything shape-dependent here risks an
            # under-reservation that fails hard in the serving path.
            # Reserving unconditionally costs nothing -- the BF16 request
            # just above is made for every layer and is strictly larger
            # (4160.5 MiB vs 2760.3 MiB at production geometry), and
            # `_ensure_workspace_size` only grows, so the arena is the max
            # over requests, not their sum.  It is not yet locked, so two
            # calls are safe.
            if not swa_only and fp8_sparse_prefill_available(
                attn_sink=self.attn_sink,
                head_dim=q.shape[-1],
                caches_are_ocp=not current_platform.is_fp8_fnuz(),
            ):
                current_workspace_manager().get_simultaneous(
                    *_fp8_prefill_workspace_requests(
                        self.PREFILL_CHUNK_SIZE,
                        M,
                        self.max_num_batched_tokens,
                        q.shape[1],
                    )
                )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            (
                topk_ragged_indices,
                topk_ragged_indptr,
                topk_lens,
            ) = compute_global_topk_ragged_indices_and_indptr(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=None,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
            extra_cache_nan_free=_trust_dsv4_extra_cache_nan_free(
                self.kv_cache_dtype,
                self._has_kv_transfer,
                not swa_only and kv_cache is not None,
            ),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Local indices filled by the index source; SWA-only layers pass
        # top_k=0 and never read them.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[:num_prefill_tokens]
        if not swa_only:
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
            self.PREFILL_CHUNK_SIZE
        )

        # The KV staging format is chosen once per call, not per chunk: the
        # workspace arena is shared, so requesting both a BF16 and an FP8
        # staging buffer would size it to their sum.  The FP8 route's
        # profitability scales with the KV rows staged per request, so take
        # the *smallest* request in the batch -- then every request clears
        # the crossover the gate encodes.
        min_staged_kv_rows = 0
        if not swa_only and num_prefills > 0:
            seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
            query_lens_cpu = swa_metadata.prefill_query_lens_cpu
            assert seq_lens_cpu is not None
            assert query_lens_cpu is not None
            # Mirrors the two gather calls below exactly: the compressed
            # prefix pool written at offset 0, plus the SWA window written
            # at offset N.  Host tensors throughout -- no device sync.
            gather_lens_cpu = query_lens_cpu + torch.clamp(
                seq_lens_cpu - query_lens_cpu,
                min=0,
                max=self.window_size - 1,
            )
            compressed_lens_cpu = torch.div(
                seq_lens_cpu, self.compress_ratio, rounding_mode="floor"
            )
            min_staged_kv_rows = int(
                (compressed_lens_cpu + gather_lens_cpu)[:num_prefills].min()
            )

        # SWA-only layers gather ~window rows per query; the FP8 kernel's
        # win comes from KV bytes moved and inverts at that density, so they
        # stay on BF16.  See LEDGER a1-fp8-prefill-nnz-crossover.
        stage_fp8 = not swa_only and can_stage_fp8_sparse_prefill(
            staged_kv_rows=min_staged_kv_rows,
            attn_sink=self.attn_sink,
            head_dim=q.shape[-1],
            caches_are_ocp=not current_platform.is_fp8_fnuz(),
        )

        workspace_manager = current_workspace_manager()
        if stage_fp8:
            # The cache's own format: 448 FP8 NoPE bytes + 14 E8M0 scale
            # bytes + pad per row, with RoPE kept BF16 alongside.  Smaller
            # than the BF16 staging buffer it replaces, so the arena -- sized
            # to the largest request the process makes -- does not grow.
            kv_nope, kv_rope, q_nope_buf, q_rope_buf = (
                workspace_manager.get_simultaneous(
                    *_fp8_prefill_workspace_requests(
                        self.PREFILL_CHUNK_SIZE,
                        M,
                        self.max_num_batched_tokens,
                        q.shape[1],
                    )
                )
            )
        else:
            kv = workspace_manager.get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            swa_block_table = swa_metadata.block_table[num_decodes:]
            if stage_fp8:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                pack_fp8_and_gather_k_cache(
                    kv_nope[:chunk_size],
                    kv_rope[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
                pack_fp8_and_gather_k_cache(
                    kv_nope[:chunk_size],
                    kv_rope[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                )
            else:
                if not swa_only:
                    assert attn_metadata is not None
                    assert compressed_k_cache is not None
                    block_table = attn_metadata.block_table[num_decodes:]
                    # compressed_k_cache is OCP on every platform.
                    dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=seq_lens[chunk_start:chunk_end]
                        // self.compress_ratio,
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // self.compress_ratio,
                        offset=0,
                        use_fnuz=False,
                    )

                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                    use_fnuz=current_platform.is_fp8_fnuz(),
                )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            if stage_fp8:
                n_q = query_end - query_start
                rocm_sparse_attn_prefill_fp8(
                    q=q[query_start:query_end],
                    q_nope_buf=q_nope_buf[:n_q],
                    q_rope_buf=q_rope_buf[:n_q],
                    kv_nope=kv_nope.view(-1, Q_MXFP8_ROW_BYTES),
                    kv_rope=kv_rope.view(-1, Q_MXFP8_ROPE_DIM),
                    indices=combined_indices,
                    topk_length=combined_lens,
                    scale=self.scale,
                    attn_sink=self.attn_sink,
                    output=output[query_start:query_end],
                )
            else:
                rocm_sparse_attn_prefill(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices,
                    topk_length=combined_lens,
                    scale=self.scale,
                    head_dim=self.head_dim,
                    nope_head_dim=self.nope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                    attn_sink=self.attn_sink,
                    output=output[query_start:query_end],
                )
