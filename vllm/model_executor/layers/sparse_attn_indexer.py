# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
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

_STABLE_TOPK_SCALE = 1.0e9


def _can_use_stable_topk_radix(scores: torch.Tensor, k: int) -> bool:
    """The fused CUDA radix top-K is exact-stable (fp32 score, then lowest token
    id) and far faster than the fp64-composite torch.topk, so it is always used
    when applicable; the fp64 path is only the fallback (CPU / k > 2048)."""
    return (
        scores.device.type == "cuda" and scores.dtype == torch.float32 and 0 < k <= 2048
    )


def _stable_topk_from_candidates_radix(
    candidate_scores: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Radix CUDA path: returns the selected global token ids (``-1`` padded)."""
    num_rows = candidate_scores.shape[0]
    out = torch.empty((num_rows, k), dtype=torch.int32, device=candidate_scores.device)
    candidate_token_ids = candidate_token_ids.to(dtype=torch.int32, copy=False)
    ops.stable_top_k_from_candidates(
        candidate_scores,
        candidate_token_ids,
        out,
        num_rows,
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_token_ids.stride(0),
        candidate_token_ids.stride(1),
        k,
    )
    return out


def _stable_topk_from_candidates_fp64(
    candidate_scores: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """fp64-composite stable top-K -- the fallback used when the radix kernel is
    unavailable (CPU / k > 2048 / not built). Selects the same set as radix."""
    num_rows, num_candidates = candidate_scores.shape
    device = candidate_scores.device
    select_k = min(k, num_candidates)
    valid = candidate_token_ids >= 0
    composite = candidate_scores.to(
        torch.float64
    ) * _STABLE_TOPK_SCALE - candidate_token_ids.to(torch.float64)
    composite = composite.masked_fill(~valid, float("-inf"))
    _, topk_pos = composite.topk(select_k, dim=-1)

    selected = candidate_token_ids.gather(1, topk_pos).to(torch.int32)
    selected_valid = valid.gather(1, topk_pos)
    selected = torch.where(selected_valid, selected, selected.new_full((), -1))
    if select_k == k:
        return selected
    pad = torch.full((num_rows, k - select_k), -1, dtype=torch.int32, device=device)
    return torch.cat((selected, pad), dim=1)


def _stable_topk_from_candidates(
    candidate_scores: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Deterministic top-K over an already-pruned candidate set.

    Ranks by score descending, then lowest global token id -- a total order, so
    the result is identical on every DCP rank (plain ``torch.topk`` breaks ties
    nondeterministically). Uses the fused CUDA radix kernel when applicable
    (fast, exact); otherwise the fp64 composite fallback. Only the selected SET
    is meaningful -- array order is implementation-defined. ``candidate_token_ids``
    holds global token ids (``-1`` marks padding).
    """
    num_rows, num_candidates = candidate_scores.shape
    device = candidate_scores.device
    if num_candidates == 0:
        return torch.full((num_rows, k), -1, dtype=torch.int32, device=device)
    if _can_use_stable_topk_radix(candidate_scores, k):
        return _stable_topk_from_candidates_radix(
            candidate_scores.float(), candidate_token_ids, k
        )
    return _stable_topk_from_candidates_fp64(candidate_scores, candidate_token_ids, k)


def _dcp_interleave_source_index(
    global_cu_seq_lens: torch.Tensor,
    local_cu_seq_lens_per_rank: torch.Tensor,
    max_local_total: int,
    dcp_size: int,
    global_total: int,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    assert cp_kv_cache_interleave_size == 1, (
        "DCP prefill K-gather currently supports only cp_kv_cache_interleave_size=1."
    )
    device = global_cu_seq_lens.device
    if global_total == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    counts = (global_cu_seq_lens[1:] - global_cu_seq_lens[:-1]).to(torch.int64)
    num_reqs = counts.numel()
    req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=device, dtype=torch.int64), counts
    )
    pos_in_chunk = torch.arange(global_total, device=device, dtype=torch.int64)
    req_starts = global_cu_seq_lens[:-1].to(torch.int64)
    t_in_req = pos_in_chunk - req_starts[req_ids]

    rank_per_pos = t_in_req % dcp_size
    lpos_per_pos = t_in_req // dcp_size
    local_starts = local_cu_seq_lens_per_rank.to(torch.int64)[rank_per_pos, req_ids]
    return rank_per_pos * max_local_total + local_starts + lpos_per_pos


def _local_dcp_indices_to_global(
    local_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    valid = local_indices >= 0
    local = local_indices.to(torch.int64).clamp_min(0)
    global_indices = (
        (local // cp_interleave) * (dcp_world_size * cp_interleave)
        + dcp_rank * cp_interleave
        + local % cp_interleave
    )
    return torch.where(valid, global_indices, -1).to(torch.int32)


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the
    whole score row. Returns global token ids (``-1`` for padding); the
    attention backend localizes them back to physical slots per rank.
    """
    if dcp_world_size <= 1:
        return topk_indices

    valid = topk_indices >= 0
    score_indices = topk_indices.clamp_min(0).to(torch.long)
    if row_starts is not None:
        score_indices = score_indices + row_starts.to(torch.long).view(-1, 1)
    local_scores = logits.gather(1, score_indices)
    local_scores = local_scores.masked_fill(~valid, float("-inf"))
    global_indices = _local_dcp_indices_to_global(
        topk_indices, dcp_rank, dcp_world_size, cp_interleave
    )

    # Pack (score, global_id) so the candidate exchange is a single all-gather.
    # Token ids are < max_model_len (<< 2**24), exactly representable in fp32.
    packed = torch.stack(
        (local_scores.float(), global_indices.to(torch.float32)), dim=-1
    ).contiguous()
    gathered = get_dcp_group().all_gather(packed, dim=1)
    candidate_scores = gathered[..., 0]
    candidate_ids = gathered[..., 1].to(torch.int32)

    return _stable_topk_from_candidates(candidate_scores, candidate_ids, topk_tokens)


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
        chunks_under_dcp = (
            prefill_metadata.chunks
            and prefill_metadata.chunks[0].local_cu_seq_lens is not None
        )
        prefill_dcp_group = get_dcp_group() if chunks_under_dcp else None
        prefill_dcp_world = prefill_dcp_group.world_size if prefill_dcp_group else 1

        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                if prefill_dcp_world > 1:
                    assert prefill_dcp_group is not None
                    assert chunk.local_cu_seq_lens is not None
                    max_local = chunk.max_local_total_seq_lens
                    local_k_quant = torch.empty(
                        (max_local, k_quant.shape[1]),
                        dtype=k_quant.dtype,
                        device=k_quant.device,
                    )
                    local_k_scale = torch.empty(
                        (max_local, k_scale.shape[1]),
                        dtype=k_scale.dtype,
                        device=k_scale.device,
                    )
                    ops.cp_gather_indexer_k_quant_cache(
                        kv_cache,
                        local_k_quant,
                        local_k_scale,
                        chunk.block_table,
                        chunk.local_cu_seq_lens,
                    )
                    gathered_k_quant = prefill_dcp_group.all_gather(
                        local_k_quant, dim=0
                    )
                    gathered_k_scale = prefill_dcp_group.all_gather(
                        local_k_scale, dim=0
                    )
                    counts = chunk.cu_seq_lens[1:] - chunk.cu_seq_lens[:-1]
                    ranks = torch.arange(
                        prefill_dcp_world,
                        device=k_quant.device,
                        dtype=torch.int32,
                    ).unsqueeze(1)
                    local_counts_all = torch.clamp(
                        (
                            counts.to(torch.int32).unsqueeze(0)
                            + (prefill_dcp_world - 1 - ranks)
                        )
                        // prefill_dcp_world,
                        min=0,
                    )
                    local_cu_seq_lens_all = torch.zeros(
                        prefill_dcp_world,
                        chunk.num_reqs + 1,
                        dtype=torch.int32,
                        device=k_quant.device,
                    )
                    torch.cumsum(
                        local_counts_all,
                        dim=1,
                        out=local_cu_seq_lens_all[:, 1:],
                    )
                    src_idx = _dcp_interleave_source_index(
                        chunk.cu_seq_lens,
                        local_cu_seq_lens_all,
                        max_local,
                        prefill_dcp_world,
                        chunk.total_seq_lens,
                        chunk.cp_kv_cache_interleave_size,
                    )
                    k_quant.copy_(gathered_k_quant.view(-1, k_quant.shape[1])[src_idx])
                    k_scale.copy_(gathered_k_scale.view(-1, k_scale.shape[1])[src_idx])
                else:
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

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

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

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
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
                logits.shape[1],
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

        if decode_metadata.global_seq_lens is not None:
            topk_indices.copy_(
                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    decode_metadata.dcp_rank,
                    decode_metadata.dcp_world_size,
                    decode_metadata.cp_kv_cache_interleave_size,
                )
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
