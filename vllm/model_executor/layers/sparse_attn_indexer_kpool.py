# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    append_tail_to_topk,
    expand_pools_to_tokens,
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
)
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

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. We locate every pool completion, gather its
    ``kpool`` tokens, and run the softmax-weighted-sum + Hadamard + fp8 + write
    kernel. Assumes pool-aligned chunk starts (same invariant as sglang).
    """
    valid = slot_mapping >= 0
    valid_pos = torch.nonzero(valid).squeeze(-1)  # positions of pool completions
    n_pools = valid_pos.numel()
    if n_pools == 0:
        return
    pool_starts = valid_pos - (kpool - 1)  # first token of each pool
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    ok = pool_starts >= 0
    if not bool(ok.all()):
        valid_pos = valid_pos[ok]
        pool_starts = pool_starts[ok]
        n_pools = valid_pos.numel()
        if n_pools == 0:
            return
    offs = torch.arange(kpool, device=k.device)
    idx = pool_starts[:, None] + offs[None, :]  # [n_pools, kpool]
    slot_k = k[idx]  # [n_pools, kpool, head_dim]
    slot_score = gate_score[idx]  # [n_pools, kpool, head_dim]
    loc = slot_mapping[valid_pos].to(torch.int64)  # [n_pools] pool slots
    kpool_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size=kpool,
        head_dim=head_dim,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def _scatter_decode_tokens_by_request(
    tokens: torch.Tensor,
    decode_lens: torch.Tensor,
    num_requests: int,
    lmax: int,
    pad_value,
) -> torch.Tensor:
    """Group ``[N, ...]`` decode tokens into a padded ``[num_requests, lmax, ...]``
    layout: request ``r``'s tokens at row ``r`` in order; short requests padded.

    ``N == decode_lens.sum()``. Unlike ``pack_seq_triton`` this is dtype-agnostic
    (needed for the int32 slot/pos tensors) — it builds the request / intra-index
    on device and scatters. Used only for the rare non-uniform
    (``requires_padding``) decode batch; uniform batches use a zero-copy reshape.
    """
    device = tokens.device
    dl = decode_lens.to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int64), dl
    )
    req_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int64), dl[:-1]]),
        dim=0,
    )
    # Broadcast the per-request start offsets to per-token (length N == dl.sum())
    # so each token's intra-request index subtracts its own request's start.
    starts = torch.repeat_interleave(req_starts, dl)
    intra = torch.arange(tokens.shape[0], device=device, dtype=torch.int64) - starts
    out = torch.full(
        (num_requests, lmax, *tokens.shape[1:]),
        pad_value,
        dtype=tokens.dtype,
        device=device,
    )
    out[req_id, intra] = tokens
    return out


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
def sparse_attn_indexer_kpool(
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
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
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

        # Sentinel allocation so the profiler's peak-memory measurement covers
        # the runtime logits tensor. The decode-path fp8_fp4_paged_mqa_logits
        # output is [B*next_n, max_model_len] float32 -- sized by max_model_len,
        # NOT bounded by the prefill chunk cap. This profiling branch returns
        # the fake before ever calling that kernel, so its output tensor is
        # invisible unless we size this sentinel to the real worst-case decode
        # batch; otherwise large max_model_len / max_num_batched_tokens OOMs at
        # warmup (the old fixed VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512MiB was
        # ~10x too small at max_model_len=1M / b8192).
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_model_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_kpool_fake(
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
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # kpool prefill write: pool kpool consecutive prefill tokens via
            # softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots.
            # Decode tokens (the first num_decode_tokens in the batch) cannot be
            # pooled here — their pool's earlier tokens are not in this batch —
            # so they are deferred to the tail-buffer kernel in has_decode.
            # compress_ratio == index_kpool makes slot_mapping pool-granular.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist the prefill tail (trailing incomplete pool's raw K +
                # gate score) into the paged tail cache, so the decode side can
                # compress the boundary pool correctly -- including across PD
                # transfer, where the connector ships this block. Only the last
                # r = n_prefill % kpool tokens matter (the in-progress pool);
                # their tail slots land at offsets 0..r-1 of the request's tail
                # block (pos % kpool), exactly where the decode reconstruction
                # reads them. The op is eager, so the host syncs here are safe.
                if (
                    tail_kv_cache is not None
                    and tail_prefix is not None
                    and os.environ.get("VLLM_KPOOL_SKIP_TAIL_CACHE") != "1"
                ):
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        r = n_prefill % index_kpool
                        if r > 0:
                            t_start = num_tokens - r
                            tslot = tail_meta.slot_mapping[t_start:num_tokens]
                            valid = tslot >= 0
                            if bool(valid.any()):
                                tslot_v = tslot[valid].to(torch.int64)
                                block_ids = tslot_v // index_kpool
                                offsets = tslot_v % index_kpool
                                tail_kv_cache[block_ids, 0, offsets, :] = k[
                                    t_start:num_tokens
                                ][valid]
                                tail_kv_cache[block_ids, 1, offsets, :] = gate_score[
                                    t_start:num_tokens
                                ][valid]
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
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

        # Short-sequence full-attention fast path (mirrors sglang
        # IndexerKPool._full_topk_for_short_sequence). When every prefill
        # request's full context is <= topk_tokens, sparse selection would
        # pick ALL pools anyway (topk_pool = topk_tokens // index_kpool >=
        # num_pools, plus the always-selected tail == every token), so running
        # the MQA-logits is pointless and (in this port) triggers OOBs. Skip
        # it and attend to every token causally instead. The index-K cache was
        # already written above; this only fills the topk buffer. Real
        # sparsity only kicks in for contexts > topk_tokens.
        n_prefill_sf = num_tokens - num_decode_tokens
        short_prefill = (
            n_prefill_sf > 0
            and positions is not None
            and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
            <= topk_tokens
        )
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _arange = torch.arange(
                topk_indices_buffer.shape[1],
                device=topk_indices_buffer.device,
                dtype=torch.int32,
            )
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _buf[:] = _arange[None, :]
            _buf[_arange[None, :] > _pos[:, None]] = -1

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
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

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
            logits = fp8_fp4_mqa_logits(
                (q_slice_cast, q_scale_slice),
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                valid = pool_ids >= 0
                expanded = expand_pools_to_tokens(
                    pool_ids, valid, topk_tokens, index_kpool
                )
                # Append the per-query causal tail: tokens beyond the last
                # complete pool within [0, pos]. seq_len_q = pos + 1; the
                # incomplete remainder (seq_len_q % kpool) is not pooled so it
                # is reachable only by direct inclusion.
                if positions is not None:
                    q_pos = positions[chunk.token_start : chunk.token_end].to(
                        torch.int32
                    )
                    q_seq = q_pos + 1
                    pool_lens = (q_seq // index_kpool).to(torch.int32)
                    expanded = append_tail_to_topk(
                        expanded, q_seq, pool_lens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # kpool decode write (must precede the logits read). Append each decode
        # token's k/gate to its REQUEST's tail ring; when a pool fills
        # (pos % kpool == kpool-1) compress + write at the pool slot that
        # compress_ratio hands us via slot_mapping.
        #
        # Spec verify batches next_n (>1) tokens per request. The per-request
        # tail ring must accumulate a request's tokens IN POSITION ORDER, so we
        # group tokens by request ([num_requests, next_n, ...]) and run the
        # per-request kernel once per token-slot — sequential launches keep each
        # request's tokens ordered (token t stashes before token t+1 reads it
        # for pool completion). Mirrors sglang's _forward_cuda_target_verify
        # (per-request kpool write plan, seqlen_per_q = write_start + k + 1).
        # Plain decode (next_n == 1) collapses to a single launch.
        #
        # NOTE: positions must be TOKEN-granular (per-token position, not the
        # pool-granular decode_metadata.seq_lens which is divided by
        # compress_ratio). The kernel derives the pool phase and tail-ring index
        # from pos % kpool, so a pool-granular pos misaligns every pool; a
        # per-request pos under spec is also too short (B entries for B*next_n
        # tokens) and reads out of bounds.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
            and os.environ.get("VLLM_KPOOL_SKIP_DECODE_WRITE") != "1"
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # The indexer's flatten decode path rewrites decode_lens to all-1s
            # and reports requires_padding=False even for a variable MTP-verify
            # batch (e.g. one request verifies 3 tokens while the rest verify
            # 4). The logits read is fine with that, but the kpool WRITE must
            # group tokens by their original request. Uniformity and the scatter
            # lmax are precomputed on the host in build()
            # (decode_is_uniform / write_max_decode_len), so this branch needs
            # no runtime .item() -- a .item() under cudagraph capture forces a
            # host sync and invalidates the stream.
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. Unreached now (per-request
                # lens is always populated for decode), kept defensive.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            if not use_uniform:
                # Non-uniform decode_lens (mixed plain-decode + spec-verify, or
                # a variable MTP-verify batch): scatter actual tokens into a
                # padded [B, lmax] layout. int32 tensors can't go through
                # pack_seq_triton (float/uint8 only).
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], group_lens, num_requests, lmax, 0
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens],
                    group_lens,
                    num_requests,
                    lmax,
                    0,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens],
                    group_lens,
                    num_requests,
                    lmax,
                    -1,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    group_lens,
                    num_requests,
                    lmax,
                    -1,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            # Paged tail cache replaces the transient _DECODE_TAIL ring. Group
            # the tail group's token-granular slot_mapping per-request, mirroring
            # dec_slot / dec_pos, so the kernel gets each request's current-token
            # tail slot (block * kpool + pos % kpool).
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if tail_meta is None or tail_kv_cache is None:
                dec_tail_slot = None
            elif not use_uniform:
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:num_decode_tokens],
                    group_lens,
                    num_requests,
                    lmax,
                    -1,
                )
            else:
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
            # The compress kernel writes the raw fp8 cache (not the quant view);
            # pass the underlying kv_cache, not kv_cache_quant_view.
            if dec_tail_slot is not None:
                # Single batched launch over [num_requests, next_n] replaces the
                # per-token sequential loop. The kernel iterates each request's
                # tokens in position order internally, preserving the
                # pool-completion read-after-stash dependency that the loop
                # provided. Inputs are already grouped per request (uniform:
                # view; non-uniform: _scatter_decode_tokens_by_request padded to
                # [B, lmax]) — no per-token .contiguous() copies needed.
                kpool_decode_update_and_maybe_write_cache_batched(
                    kv_cache_raw,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
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
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and select_k in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                select_k,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            valid = pool_ids >= 0
            out = expand_pools_to_tokens(pool_ids, valid, topk_tokens, index_kpool)
            # Append the always-selected tail: the request's trailing incomplete
            # pool (seq_len % kpool tokens) is not in the index K cache, so it
            # can only be reached by direct inclusion.
            n = out.shape[0]
            # NOTE: decode_metadata.seq_lens is POOL-granular (divided by
            # compress_ratio in the indexer metadata builder) because it feeds
            # the paged-MQA logits. append_tail_to_topk / pool_lens need
            # TOKEN-granular seq_len, so recover it from the decode tokens'
            # positions (pos == seq_len - 1). Using the compressed seq_lens
            # here yields dec_seq=0 for seq_len<kpool -> empty topk -> the
            # sparse MLA attends to nothing -> decode degradation.
            if positions is not None:
                dec_seq = positions[:n].to(torch.int32) + 1
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            pool_lens = (dec_seq // index_kpool).to(torch.int32)
            out = append_tail_to_topk(out, dec_seq, pool_lens, index_kpool)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


def sparse_attn_indexer_kpool_fake(
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
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool",
    op_func=sparse_attn_indexer_kpool,
    # The indexer writes the index-K cache in place (prefill k-cache insert +
    # kpool decode write), so kv_cache must be declared as mutated — otherwise
    # under full-graph compile dynamo assumes it is unchanged across the
    # indexer→MLA boundary and the MLA reads stale/misaligned KV. The paged tail
    # cache is likewise written in place (prefill tail scatter + decode stash).
    mutates_args=["topk_indices_buffer", "kv_cache", "tail_kv_cache"],
    fake_impl=sparse_attn_indexer_kpool_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
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
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
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
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(
                hidden_states,
                q_quant,
                k,
                weights,
                gate_score=gate_score,
                compress_ape=compress_ape,
                index_kpool=index_kpool,
                positions=positions,
            )
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
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer_kpool(
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
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
        )

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
