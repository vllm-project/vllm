# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed.parallel_state import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
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
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _use_persistent_topk_decode(topk_tokens: int) -> bool:
    """Return whether decode top-k should use the persistent CUDA kernel.

    The caller must pass persistent_topk a max sequence length in the same
    coordinate system as its seq_lens tensor. Under DCP this is rank-local, not
    the global attention max.
    """
    return current_platform.is_cuda() and topk_tokens in (512, 1024, 2048)


def _local_to_global_position(
    local_idx: torch.Tensor, rank: int, world_size: int, interleave: int
) -> torch.Tensor:
    """Map a per-request LOCAL kv position on this rank to its GLOBAL position.

    g(l, r) = (l // I) * (N * I) + r * I + (l % I)
    with I = interleave, N = world_size. Matches get_dcp_local_seq_lens.
    """
    return (
        (local_idx // interleave) * (world_size * interleave)
        + rank * interleave
        + (local_idx % interleave)
    )


def _global_to_local_position(
    global_idx: torch.Tensor, interleave: int, world_size: int
) -> torch.Tensor:
    """Map a GLOBAL kv position to its LOCAL index on the rank that owns it.

    local(g) = (g // (I * N)) * I + (g % I)
    """
    big = interleave * world_size
    return (global_idx // big) * interleave + (global_idx % interleave)


@triton.jit
def _dcp_pack_topk_candidates_kernel(
    topk_indices_ptr,
    logits_ptr,
    row_starts_ptr,
    packed_ptr,
    topk_stride_0: tl.constexpr,
    topk_stride_1: tl.constexpr,
    logits_stride_0: tl.constexpr,
    logits_stride_1: tl.constexpr,
    packed_stride_0: tl.constexpr,
    packed_stride_1: tl.constexpr,
    packed_stride_2: tl.constexpr,
    logits_width,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    TOPK_TOKENS: tl.constexpr,
    HAS_LOGITS: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOPK_TOKENS

    idx = tl.load(
        topk_indices_ptr + row * topk_stride_0 + offsets * topk_stride_1,
        mask=mask,
        other=-1,
    )
    invalid = idx < 0
    idx_safe = tl.maximum(idx, 0)

    scores = tl.full((BLOCK_K,), -float("inf"), tl.float32)
    if HAS_LOGITS:
        score_idx = idx_safe
        if HAS_ROW_STARTS:
            row_start = tl.load(row_starts_ptr + row)
            score_idx += row_start
        score_idx = tl.minimum(score_idx, logits_width - 1)
        scores = tl.load(
            logits_ptr + row * logits_stride_0 + score_idx * logits_stride_1,
            mask=mask,
            other=-float("inf"),
        )
        scores = tl.where(invalid, -float("inf"), scores)

    global_pos = (
        (idx_safe // INTERLEAVE) * (WORLD_SIZE * INTERLEAVE)
        + RANK * INTERLEAVE
        + (idx_safe % INTERLEAVE)
    )
    global_pos = tl.where(invalid, -1, global_pos)

    packed_base = packed_ptr + row * packed_stride_0 + offsets * packed_stride_2
    tl.store(packed_base, global_pos, mask=mask)
    tl.store(
        packed_base + packed_stride_1,
        scores.to(tl.int32, bitcast=True),
        mask=mask,
    )


@triton.jit
def _dcp_finalize_topk_remap_kernel(
    all_candidates_ptr,
    selected_ptr,
    topk_indices_ptr,
    candidates_stride_0: tl.constexpr,
    candidates_stride_1: tl.constexpr,
    candidates_stride_2: tl.constexpr,
    selected_stride_0: tl.constexpr,
    selected_stride_1: tl.constexpr,
    topk_stride_0: tl.constexpr,
    topk_stride_1: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    TOPK_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOPK_TOKENS

    selected = tl.load(
        selected_ptr + row * selected_stride_0 + offsets * selected_stride_1,
        mask=mask,
        other=0,
    )
    global_pos = tl.load(
        all_candidates_ptr + row * candidates_stride_0 + selected * candidates_stride_2,
        mask=mask,
        other=-1,
    )

    owner = (global_pos // INTERLEAVE) % WORLD_SIZE
    big = INTERLEAVE * WORLD_SIZE
    local_pos = (global_pos // big) * INTERLEAVE + (global_pos % INTERLEAVE)
    mine = (owner == RANK) & (global_pos >= 0)
    final = tl.where(mine, local_pos, -1)

    tl.store(
        topk_indices_ptr + row * topk_stride_0 + offsets * topk_stride_1,
        final,
        mask=mask,
    )


def _use_triton_dcp_remap(topk_indices: torch.Tensor) -> bool:
    return HAS_TRITON and current_platform.is_cuda() and topk_indices.is_cuda


def _dcp_pack_topk_candidates(
    topk_indices: torch.Tensor,
    logits: torch.Tensor | None,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack local candidate global positions and score bits for one all-gather."""
    packed = torch.empty(
        (topk_indices.shape[0], 2, topk_tokens),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if topk_indices.numel() == 0:
        return packed

    if _use_triton_dcp_remap(topk_indices):
        block_k = triton.next_power_of_2(topk_tokens)
        num_warps = 4 if block_k <= 256 else 8
        logits_arg = logits if logits is not None else topk_indices
        row_starts_arg = row_starts if row_starts is not None else topk_indices
        _dcp_pack_topk_candidates_kernel[(topk_indices.shape[0],)](
            topk_indices,
            logits_arg,
            row_starts_arg,
            packed,
            topk_indices.stride(0),
            topk_indices.stride(1),
            logits.stride(0) if logits is not None else 0,
            logits.stride(1) if logits is not None else 0,
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            logits.shape[1] if logits is not None else 1,
            RANK=rank,
            WORLD_SIZE=world_size,
            INTERLEAVE=interleave,
            TOPK_TOKENS=topk_tokens,
            HAS_LOGITS=logits is not None,
            HAS_ROW_STARTS=row_starts is not None,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
        return packed

    invalid = topk_indices < 0
    idx_safe = torch.clamp(topk_indices, min=0)
    if logits is None:
        local_scores = torch.full(
            topk_indices.shape,
            float("-inf"),
            dtype=torch.float32,
            device=topk_indices.device,
        )
    else:
        score_idx = idx_safe.to(torch.int64)
        if row_starts is not None:
            score_idx = score_idx + row_starts.to(
                device=score_idx.device, dtype=score_idx.dtype
            ).view(-1, 1)
        score_idx = torch.clamp(score_idx, min=0, max=logits.shape[1] - 1)
        local_scores = torch.gather(logits, 1, score_idx).to(torch.float32)
    local_scores = local_scores.masked_fill(invalid, float("-inf"))

    global_pos = _local_to_global_position(idx_safe, rank, world_size, interleave)
    global_pos = torch.where(invalid, global_pos.new_full((), -1), global_pos)

    packed[:, 0, :].copy_(global_pos.to(torch.int32))
    packed[:, 1, :].copy_(local_scores.contiguous().view(torch.int32))
    return packed


def _dcp_finalize_topk_remap(
    all_candidates: torch.Tensor,
    selected: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
) -> None:
    """Finalize selected global candidates into rank-local top-k indices."""
    if topk_indices.numel() == 0:
        return

    if _use_triton_dcp_remap(topk_indices):
        block_k = triton.next_power_of_2(topk_tokens)
        num_warps = 4 if block_k <= 256 else 8
        _dcp_finalize_topk_remap_kernel[(topk_indices.shape[0],)](
            all_candidates,
            selected,
            topk_indices,
            all_candidates.stride(0),
            all_candidates.stride(1),
            all_candidates.stride(2),
            selected.stride(0),
            selected.stride(1),
            topk_indices.stride(0),
            topk_indices.stride(1),
            RANK=rank,
            WORLD_SIZE=world_size,
            INTERLEAVE=interleave,
            TOPK_TOKENS=topk_tokens,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
        return

    sel_global = torch.gather(all_candidates[:, 0, :], 1, selected.to(torch.int64))
    owner = (sel_global // interleave) % world_size
    local_of_g = _global_to_local_position(sel_global, interleave, world_size)
    mine = (owner == rank) & (sel_global >= 0)
    final = torch.where(mine, local_of_g, local_of_g.new_full((), -1))
    topk_indices.copy_(final.to(topk_indices.dtype))


def _dcp_global_topk_remap(
    topk_indices: torch.Tensor,
    logits: torch.Tensor | None,
    topk_tokens: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """In place: convert per-rank LOCAL top-k selection into the GLOBAL top-k,
    restricted to the positions this rank owns (positions owned by other ranks
    are written as -1).

    Background: under DCP the KV cache is sharded across the DCP group (interleave
    `I`-way, `N` ranks). Each rank's local top-k is selected from only its shard,
    so naively LSE-merging per-rank sparse attention would reconstruct attention
    over the *union* of per-rank selections (~N * topk positions) instead of the
    true global top-k. To get exact global sparse attention after the DCP LSE
    merge, every rank must select the SAME global set G and then attend only to
    G ∩ (its own shard). This does exactly that:

      1. recover local scores at the selected indices (-1 -> invalid/-inf),
      2. map local positions -> global positions,
      3. all-gather (global_pos, score) candidates across the DCP group,
      4. take the global top-k per row,
      5. map global -> local, keeping only positions owned by this rank (-1 else),
      6. write back in place.

    Rows are aligned across ranks because decode queries are replicated under
    DCP. ``topk_indices`` and ``logits`` must share the same row count.

    Args:
        topk_indices: int32 [num_rows, topk_tokens], per-rank local indices
            (a view into topk_indices_buffer); -1 marks an unused slot.
        logits: float32 [num_rows, seq_pad], the per-row MQA scores the local
            top-k was taken over. None means this rank has no local candidates
            for these rows, but must still participate in the DCP collective.
        topk_tokens: K, the desired global selection size.
        interleave: cp_kv_cache_interleave_size (I).
        row_starts: Optional per-row offset into ``logits``. Prefill top-k
            indices are local to each row's valid [start, end) range.
    """
    dcp_group = get_dcp_group()
    rank = dcp_group.rank_in_group
    world_size = dcp_group.world_size

    # 1-3. Pack (global_pos, score_bits) and all-gather candidate pairs once.
    candidates = _dcp_pack_topk_candidates(
        topk_indices,
        logits,
        topk_tokens,
        rank,
        world_size,
        interleave,
        row_starts=row_starts,
    )
    all_candidates = dcp_group.all_gather(candidates.contiguous(), dim=2)
    all_scores = all_candidates[:, 1, :].view(torch.float32)

    # 4. Global top-k per row using gathered score bits.
    _, sel = torch.topk(all_scores, topk_tokens, dim=1)

    # 5-6. Global -> local; keep only this rank's owned positions, else -1.
    _dcp_finalize_topk_remap(
        all_candidates,
        sel,
        topk_indices,
        topk_tokens,
        rank,
        world_size,
        interleave,
    )


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


@eager_break_during_capture
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
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
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
            use_fp4_cache,
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
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

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
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.total_seq_lens == 0:
                topk_indices.fill_(-1)
                if attn_metadata_narrowed.dcp_world_size > 1:
                    _dcp_global_topk_remap(
                        topk_indices,
                        None,
                        topk_tokens,
                        attn_metadata_narrowed.cp_interleave_size,
                    )
                continue

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
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
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            # Under DCP, convert the per-rank local top-k into the global top-k
            # restricted to this rank's owned positions. See _dcp_global_topk_remap.
            if attn_metadata_narrowed.dcp_world_size > 1:
                _dcp_global_topk_remap(
                    topk_indices,
                    logits,
                    topk_tokens,
                    attn_metadata_narrowed.cp_interleave_size,
                    row_starts=chunk.cu_seqlen_ks,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
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
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if _use_persistent_topk_decode(topk_tokens):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                decode_metadata.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        # Under DCP, convert the per-rank local top-k into the global top-k
        # restricted to this rank's owned positions. Done before the padding
        # unpack so rows align with `logits`. See _dcp_global_topk_remap.
        if attn_metadata_narrowed.dcp_world_size > 1:
            _dcp_global_topk_remap(
                topk_indices,
                logits,
                topk_tokens,
                attn_metadata_narrowed.cp_interleave_size,
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
    use_fp4_cache: bool = False,
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
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
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
            self.use_fp4_cache,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
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
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
