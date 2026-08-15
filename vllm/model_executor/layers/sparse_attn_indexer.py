# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.indexer_turboquant import (
    INDEXER_FP8_SLOT_BYTES,
    indexer_tq_store_and_cache,
    is_indexer_tq_4bit_enabled,
    sync_fp8_workspace_for_decode,
    tq4_paged_mqa_logits_triton,
    use_indexer_tq_fused_decode,
)
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)


@triton.jit
def _dcp_scatter_indexer_logits_kernel(
    local_ptr,
    local_row_stride,
    global_ptr,
    global_row_stride,
    seq_lens_local_ptr,  # int32 [num_rows], per-row LOCAL context length
    N: tl.constexpr,  # dcp_world_size
    RANK: tl.constexpr,  # dcp_rank
    S: tl.constexpr,  # cp_kv_cache_interleave_size
    max_local_cols,  # cdiv(global_width, N)
    global_width,
    BLOCK: tl.constexpr,
):
    # Scatter this rank's local-order logits to their global positions.
    # The KV cache is sharded across `N` ranks by an interleaved round-robin at
    # granularity `S` (mirrors block_table._compute_slot_mapping_kernel). The
    # L-th local token on rank RANK lives in interleave-block ``L // S`` at
    # offset ``L % S``, whose global position is
    #     (L // S) * (N * S) + RANK * S + (L % S).
    # With S == 1 this reduces to ``L * N + RANK`` (per-token round-robin).
    row = tl.program_id(0)
    llen = tl.load(seq_lens_local_ptr + row)
    for i in range(0, max_local_cols, BLOCK):
        L = i + tl.arange(0, BLOCK)
        valid = llen > L
        val = tl.load(local_ptr + row * local_row_stride + L, mask=valid, other=0.0)
        gcol = (L // S) * (N * S) + RANK * S + (L % S)
        gvalid = valid & (gcol < global_width)
        tl.store(global_ptr + row * global_row_stride + gcol, val, mask=gvalid)


def _dcp_allgather_indexer_logits(
    local_logits: torch.Tensor,
    local_seq_lens: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    cp_interleave_size: int = 1,
) -> torch.Tensor:
    """Reconstruct full-sequence indexer logits from per-rank shards under DCP.

    Each rank computed ``local_logits`` over the KV tokens it physically holds
    (local order). We scatter them back to their global sequence positions into
    a zero-filled buffer (same shape/layout as the kernel output, so the
    downstream top-k kernels see the exact layout they expect) and SUM-reduce
    across the DCP group. Every global position is owned by exactly one rank
    (the round-robin partition), so the other ranks contribute 0 and the sum
    equals that rank's real logit; positions beyond the sequence are 0 on every
    rank and are masked out by the global seq_lens in the top-k. The reduce uses
    the DCP group coordinator (not a raw torch.distributed call) so it is issued
    on vLLM's collective stream and stays compatible with CUDA graph capture.
    """
    from vllm.distributed import get_dcp_group

    num_rows, width = local_logits.shape
    global_logits = torch.zeros_like(local_logits)
    seq_lens_local_flat = local_seq_lens.reshape(-1).to(torch.int32).contiguous()
    max_local_cols = (width + dcp_world_size - 1) // dcp_world_size
    _dcp_scatter_indexer_logits_kernel[(num_rows,)](
        local_logits,
        local_logits.stride(0),
        global_logits,
        global_logits.stride(0),
        seq_lens_local_flat,
        dcp_world_size,
        dcp_rank,
        cp_interleave_size,
        max_local_cols,
        width,
        BLOCK=1024,
    )
    return get_dcp_group().all_reduce(global_logits)


def indexer_dcp_distributed_topk_enabled() -> bool:
    """Distributed indexer top-k under DCP (decode path only).

    When enabled, replaces the full-width ``all_reduce`` + global top-k
    (``_dcp_allgather_indexer_logits`` + ``persistent_topk`` over the whole
    ``[R, max_model_len]`` buffer) with: per-rank local top-k -> all_gather of
    only the ``k * dcp_world_size`` (global_col, value) candidates -> an N-way
    merge. This turns the DCP indexer glue from O(max_model_len) traffic into
    O(k * dcp_world_size), the dominant win at dcp>=4 / long context.

    Correctness: the global top-k is a subset of the union of the per-rank
    local top-k. Each rank returns k candidates and every globally-selected
    column, on its owning rank, ranks among that rank's local top-k (its rank
    only holds a subset of the sequence), so no winner is ever dropped. All
    ranks all_gather the same candidate set and run an identical merge, so every
    rank writes a bit-identical ``topk_indices_buffer`` -- exactly what the
    downstream per-rank ``triton_convert_req_index_to_global_index`` filtering
    and the LSE combine require.

    Default off; falls back to the ``all_reduce`` path when unset.
    """
    return os.environ.get("VLLM_INDEXER_DCP_DISTRIBUTED_TOPK", "0") == "1"


def _dcp_distributed_topk_indexer(
    local_logits: torch.Tensor,
    local_seq_lens: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    num_padded_tokens: int,
    topk_tokens: int,
    max_seq_len: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_interleave_size: int,
) -> None:
    """Fill ``topk_indices_buffer[:R, :topk_tokens]`` with the GLOBAL top-k
    column indices under DCP, without materializing the full-width all_reduce.

    ``local_logits`` is ``[R, max_model_len]`` in this rank's LOCAL order (the
    paged-MQA kernel wrote only this rank's KV shard, bounded by
    ``local_seq_lens``); tail columns are stale and excluded by the per-row
    seq_lens in the local top-k. See ``indexer_dcp_distributed_topk_enabled``
    for the correctness argument.
    """
    from vllm.distributed import get_dcp_group

    R = num_padded_tokens
    N = dcp_world_size
    S = cp_interleave_size
    logits = local_logits[:R]
    seq_lens_local = local_seq_lens.reshape(-1)[:R].to(torch.int32).contiguous()

    # (1) local top-k -> local column indices [R, k] (int32; -1 pads short rows).
    #     Reuse the same radix persistent_topk, but over this rank's shard only
    #     (bounded by the LOCAL seq_lens), so it scans ~ctx/N per row. Pre-fill
    #     with -1: persistent_topk writes only the valid entries and leaves the
    #     pad slots (rows whose local context < topk_tokens) untouched, so the
    #     -1 must already be there for the `valid` mask below to be correct.
    local_topk = torch.full(
        (R, topk_tokens), -1, dtype=torch.int32, device=logits.device
    )
    workspace_manager = current_workspace_manager()
    (topk_workspace,) = workspace_manager.get_simultaneous(
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )
    torch.ops._C.persistent_topk(
        logits,
        seq_lens_local,
        local_topk,
        topk_workspace,
        topk_tokens,
        max_seq_len,
    )

    # (2) re-gather the local top-k VALUES from the local logits (persistent_topk
    #     emits indices only). A -1 index (short row) -> most-negative so it never
    #     wins the merge.
    valid = local_topk >= 0
    local_vals = torch.gather(logits, 1, local_topk.clamp_min(0).to(torch.int64))
    neg_inf = torch.finfo(local_vals.dtype).min
    local_vals = torch.where(valid, local_vals, neg_inf)

    # (3) local column -> global column (round-robin the scatter kernel uses:
    #     gcol=(L//S)*(N*S)+RANK*S+(L%S)); -1 for pads.
    L = local_topk.to(torch.int64)
    gcol = torch.where(
        valid,
        (L // S) * (N * S) + dcp_rank * S + (L % S),
        torch.full_like(L, -1),
    )

    # (4) Build a DETERMINISTIC total-order key per candidate and exchange it in a
    #     SINGLE all_gather (matches upstream #46076 FLASHINFER_MLA_SPARSE /
    #     #46145). A plain value-only torch.topk breaks ties nondeterministically
    #     per launch; DCP decode runs the fp8 indexer whose scores tie heavily,
    #     so ranks could keep different tied candidates from the (identical)
    #     all_gathered set -> each rank's downstream sparse filter sees a
    #     different global token set -> inconsistent LSE merge -> intermittent
    #     garbled output. The key packs a strict order into one int64: the
    #     score's monotone-uint32 bits in the high 32 and ~global_col in the low
    #     32 (lowest global column wins ties). Global columns are unique across
    #     ranks (disjoint round-robin), so the key is a strict order -> topk has
    #     no ties to break -> identical selection on every rank. Packing (score,
    #     col) into this one key lets us all_gather ONE tensor instead of two
    #     (value + column) and recover the column straight from the winning key.
    sign64 = -9223372036854775808  # 1<<63: flip so signed topk sorts unsigned
    sc_u32 = local_vals.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    # IEEE-754 fp32 bits -> order-preserving uint32: flip all bits if negative
    # (sign set), else flip just the sign bit.
    ordered = torch.where(
        (sc_u32 >> 31) & 1 == 1,
        (~sc_u32) & 0xFFFFFFFF,
        sc_u32 ^ 0x80000000,
    )
    key_local = (ordered << 32) | ((~gcol) & 0xFFFFFFFF)
    key_local = key_local ^ torch.full_like(key_local, sign64)
    all_keys = get_dcp_group().all_gather(key_local.contiguous(), dim=1)  # [R, N*k]

    # (5) merge = global top-k over the exchanged keys; recover the global column
    #     from each winning key's low 32 bits (col = ~low; pads decode to -1).
    top_keys, _ = torch.topk(all_keys, topk_tokens, dim=1)
    recovered_low = (top_keys ^ torch.full_like(top_keys, sign64)) & 0xFFFFFFFF
    topk_indices_buffer[:R, :topk_tokens] = (~recovered_low).to(torch.int32)


def _dcp_reconstruct_full_indexer_k(
    kv_cache: torch.Tensor,
    chunk,
    values_width: int,
    values_dtype: torch.dtype,
    scales_width: int,
    scales_dtype: torch.dtype,
    device: torch.device,
    cp_interleave_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the full prefill context index-K from per-rank DCP shards.

    Mirrors the dense MLA prefill path
    (mla_attention._context_parallel_compute_prefill_context): each rank gathers
    its LOCAL index-K (padded so every rank gathers the same number of tokens),
    we all-gather across the DCP group, then reorg the shards back into global
    token order. The KV cache is sharded by an interleaved round-robin at
    granularity ``S = cp_kv_cache_interleave_size``: global position ``p`` is held
    by rank ``(p // S) % n`` at local index ``(p // (n*S)) * S + (p % S)``. The
    reorg therefore regroups the all-gathered shards ``[n, padded_len, W]`` into
    ``[padded_len // S, n, S, W]`` (interleave-block major) before flattening to
    global order. With S == 1 this is the original transpose-reshape
    (rank-major -> per-token round-robin). Returns (k_quant, k_scale) over the
    full context, in the same layout cp_gather_indexer_k_quant_cache produces
    without DCP.
    """
    from vllm.distributed import get_dcp_group

    n = chunk.dcp_world_size
    s = cp_interleave_size
    sum_padded = int(chunk.local_cu_seq_lens[-1].item())
    local_k = torch.empty((sum_padded, values_width), dtype=values_dtype, device=device)
    local_scale = torch.empty(
        (sum_padded, scales_width), dtype=scales_dtype, device=device
    )
    # Local gather: padded local cu_seq_lens -> this rank's compact local tokens
    # (plus a little block padding) for each request.
    ops.cp_gather_indexer_k_quant_cache(
        kv_cache, local_k, local_scale, chunk.block_table, chunk.local_cu_seq_lens
    )
    # All-gather across the DCP group (rank-major concatenation along dim 0).
    ag_k = (
        get_dcp_group()
        .all_gather(local_k.view(torch.uint8), dim=0)
        .view(values_dtype)
        .view(n, sum_padded, values_width)
    )
    ag_scale = (
        get_dcp_group().all_gather(local_scale, dim=0).view(n, sum_padded, scales_width)
    )

    # Reorg per request back into global token order, trimmed to ctx.
    def _reorg(ag, width, padded_len, ctx_len):
        seg = ag[:, offset : offset + padded_len, :]
        if s == 1:
            return seg.transpose(0, 1).reshape(padded_len * n, width)[:ctx_len]
        assert padded_len % s == 0, (
            f"padded local seq len {padded_len} must be a multiple of "
            f"cp_kv_cache_interleave_size {s}"
        )
        return (
            seg.reshape(n, padded_len // s, s, width)
            .permute(1, 0, 2, 3)
            .reshape(padded_len * n, width)[:ctx_len]
        )

    k_segments = []
    s_segments = []
    offset = 0
    for padded_len, ctx_len in zip(
        chunk.padded_local_seq_lens, chunk.global_seq_lens_lst
    ):
        k_segments.append(_reorg(ag_k, values_width, padded_len, ctx_len))
        s_segments.append(_reorg(ag_scale, scales_width, padded_len, ctx_len))
        offset += padded_len
    k_full = torch.cat(k_segments, dim=0)
    s_full = torch.cat(s_segments, dim=0)
    return k_full, s_full


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        if is_indexer_tq_4bit_enabled() and kv_cache.numel() > 0:
            # Decode paged DeepGEMM reads FP8-shaped workspace synced from TQ cache.
            fp8_ws_shape = tuple(kv_cache.shape[:-1]) + (INDEXER_FP8_SLOT_BYTES,)
            current_workspace_manager().get_simultaneous(
                (fp8_ws_shape, torch.uint8),
            )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
        )
    # torch.compile warmup when DSA disabled (buffer is None).
    if topk_indices_buffer is None:
        return torch.empty(
            0, topk_tokens, dtype=torch.int32, device=hidden_states.device
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        if is_indexer_tq_4bit_enabled():
            indexer_tq_store_and_cache(k, kv_cache, slot_mapping_for_cache)
        else:
            # scale_fmt can be None, but the function expects str.
            assert scale_fmt is not None
            assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping_for_cache,
                quant_block_size,
                scale_fmt,
            )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        use_tq_fused_decode = (
            is_indexer_tq_4bit_enabled()
            and use_indexer_tq_fused_decode()
            and not use_fp4_cache
            and next_n == 1
            and q_scale is None
        )
        if use_tq_fused_decode:
            q_decode = padded_q_quant_decode_tokens[:, 0]
            seq_lens_logits = seq_lens.squeeze(-1) if seq_lens.dim() > 1 else seq_lens
            logits = tq4_paged_mqa_logits_triton(
                q_decode,
                weights[:batch_size],
                kv_cache,
                decode_metadata.block_table,
                seq_lens_logits,
                max_model_len,
            )
        else:
            decode_kv_cache = kv_cache
            if is_indexer_tq_4bit_enabled():
                num_blocks, block_size, _ = kv_cache.shape
                workspace_manager = current_workspace_manager()
                (fp8_workspace,) = workspace_manager.get_simultaneous(
                    ((num_blocks, block_size, INDEXER_FP8_SLOT_BYTES), torch.uint8),
                )
                decode_kv_cache = sync_fp8_workspace_for_decode(
                    kv_cache,
                    fp8_workspace,
                    decode_metadata.block_table,
                    decode_metadata.seq_lens,
                    scale_fmt,
                    max_model_len=max_model_len,
                    schedule_metadata=decode_metadata.schedule_metadata,
                )
            kv_cache_view = kv_cache_as_quant_view(
                decode_kv_cache, head_dim, use_fp4_cache
            )
            padded_q_quant_cast = (
                padded_q_quant_decode_tokens.view(torch.int8)
                if use_fp4_cache
                else padded_q_quant_decode_tokens
            )
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache_view,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )

        # Under DCP the logits kernels above read only this rank's KV shard (with
        # local seq_lens), producing per-rank logits in local order. Scatter them
        # back to global positions and reduce across the DCP group so every rank
        # holds the full logits and selects an identical GLOBAL top-k, using the
        # global seq_lens. DCP forces the FP8 indexer path (TQ4 fused decode is
        # disabled under DCP, matching the prefill assert), so the cooperative TMA
        # top-k -- already dropped by TQ in favor of persistent_topk -- is not a
        # concern here.
        # Distributed top-k (flag-gated): keep the per-rank LOCAL logits, do a
        # local top-k, exchange only the k*N candidates, and merge -- instead of
        # the full-width all_reduce below. Only the CUDA radix persistent_topk
        # path is supported (topk_tokens in {512,1024,2048}); anything else falls
        # back to the all_reduce path.
        dcp_distributed_topk = (
            decode_metadata.dcp_world_size > 1
            and indexer_dcp_distributed_topk_enabled()
            and current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
        )

        if decode_metadata.dcp_world_size > 1 and not dcp_distributed_topk:
            assert decode_metadata.global_seq_lens is not None
            topk_seq_lens = decode_metadata.global_seq_lens[:batch_size]
            logits = _dcp_allgather_indexer_logits(
                logits,
                seq_lens,
                decode_metadata.dcp_world_size,
                decode_metadata.dcp_rank,
                decode_metadata.cp_interleave_size,
            )
        else:
            topk_seq_lens = seq_lens

        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                topk_seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                topk_seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    topk_seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    topk_seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

        if dcp_distributed_topk:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.skip_k_cache_insert, (
            "AMD platform doesn't support skip cache insert yet"
        )
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        from vllm.platforms.rocm import on_gfx11

        if (
            rocm_aiter_ops.is_enabled()
            or rocm_aiter_ops.is_rdna_aiter_enabled()
            or on_gfx11()
        ):
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
