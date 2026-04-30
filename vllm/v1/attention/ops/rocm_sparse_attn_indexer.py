# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm path for DeepSeek-V4's sparse attention indexer when the compressor
has already inserted the (compressed) K into the indexer's KV cache, i.e.
``skip_k_cache_insert=True`` and the call site passes ``k=None``.

The CUDA implementation in ``vllm/model_executor/layers/sparse_attn_indexer.py``
relies on DeepGEMM's ``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits``
which are NVIDIA-only. The existing ROCm AITER op
(``rocm_aiter_sparse_attn_indexer`` in ``rocm_aiter_mla_sparse.py``) always
performs its own ``indexer_k_quant_and_cache`` call and dereferences ``k``,
so it can't be reused for the V4 layout where the compressor pre-inserts K
and returns ``None``.

This module fills that gap with:
  * A streaming Triton MQA-logits kernel that runs on gfx9xx, computing
    logits without materializing the (H, M, N) intermediate that the torch
    reference does (which would OOM at long context).
  * A torch fallback (``_mqa_logits_torch_inplace``) used for smoke tests and
    on platforms without a usable Triton runtime.
  * The orchestration (``rocm_sparse_attn_indexer_no_insert``) that mirrors
    the CUDA ``sparse_attn_indexer`` body but skips the K-insert and uses
    only ROCm-available helper ops (``cp_gather_indexer_k_quant_cache``,
    ``top_k_per_row_prefill`` / ``top_k_per_row_decode``).
"""
from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops


# Reuse the gather-workspace helper from the CUDA module so the workspace
# layout (and therefore the size estimate during profile_run) is shared.
def _gather_workspace_shapes_fp8(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
) -> tuple[
    tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]
]:
    """FP8 path layout used by ``cp_gather_indexer_k_quant_cache``: a flat
    ``(T, head_dim)`` FP8 values buffer and a ``(T, 4)`` uint8 buffer that
    aliases ``(T, 1)`` float32 dequant scales (one scale per token block).
    Mirrors the FP8 branch of ``_gather_workspace_shapes`` in
    ``sparse_attn_indexer.py``.
    """
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


# ---------------------------------------------------------------------------
# Triton MQA-logits kernel (prefill / chunked path).
#
# Computes `logits[m, n] = scale[n] * sum_h weights[m, h] * relu(q[m,h,:] . k[n,:])`
# without materializing the (H, M, N) intermediate. Streams over heads so the
# only per-program memory is (BLOCK_N,) accumulator + (D,) Q + (BLOCK_N, D) K.
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _mqa_logits_prefill_kernel(
        q_ptr,  # (M, H, D) fp8
        weights_ptr,  # (M, H) fp32
        k_ptr,  # (N, D) fp8
        k_scale_ptr,  # (N,) fp32
        cu_seqlen_ks_ptr,  # (M,) int32
        cu_seqlen_ke_ptr,  # (M,) int32
        logits_ptr,  # (M, N) fp32 (output)
        stride_qm,
        stride_qh,
        stride_qd,
        stride_wm,
        stride_wh,
        stride_kn,
        stride_kd,
        stride_lm,
        stride_ln,
        M,
        N,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        in_bounds = n_offsets < N

        ks = tl.load(cu_seqlen_ks_ptr + pid_m)
        ke = tl.load(cu_seqlen_ke_ptr + pid_m)
        valid = in_bounds & (n_offsets >= ks) & (n_offsets < ke)

        d_offsets = tl.arange(0, D)

        # Load K block once and reuse across heads: (BLOCK_N, D) fp32.
        k_block = tl.load(
            k_ptr
            + n_offsets[:, None] * stride_kn
            + d_offsets[None, :] * stride_kd,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        accum = tl.zeros([BLOCK_N], dtype=tl.float32)

        for h in range(H):
            q = tl.load(
                q_ptr
                + pid_m * stride_qm
                + h * stride_qh
                + d_offsets * stride_qd,
            ).to(tl.float32)
            w = tl.load(weights_ptr + pid_m * stride_wm + h * stride_wh).to(
                tl.float32
            )

            score = tl.sum(k_block * q[None, :], axis=1)
            accum += w * tl.maximum(score, 0.0)

        k_scale = tl.load(k_scale_ptr + n_offsets, mask=valid, other=0.0)
        logits = accum * k_scale
        logits = tl.where(valid, logits, float("-inf"))

        tl.store(
            logits_ptr + pid_m * stride_lm + n_offsets * stride_ln,
            logits,
            mask=in_bounds,
        )


def _mqa_logits_triton(
    q_fp8: torch.Tensor,  # (M, H, D)
    k_fp8: torch.Tensor,  # (N, D)
    k_scale: torch.Tensor,  # (N,) fp32
    weights: torch.Tensor,  # (M, H) fp32
    cu_seqlen_ks: torch.Tensor,  # (M,) int32
    cu_seqlen_ke: torch.Tensor,  # (M,) int32
) -> torch.Tensor:
    M, H, D = q_fp8.shape
    N = k_fp8.shape[0]
    assert k_fp8.shape[1] == D
    assert weights.shape == (M, H)
    assert k_scale.shape == (N,)

    logits = torch.empty((M, N), dtype=torch.float32, device=q_fp8.device)
    BLOCK_N = 64

    grid = (M, triton.cdiv(N, BLOCK_N))
    _mqa_logits_prefill_kernel[grid](
        q_fp8,
        weights,
        k_fp8,
        k_scale,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        weights.stride(0),
        weights.stride(1),
        k_fp8.stride(0),
        k_fp8.stride(1),
        logits.stride(0),
        logits.stride(1),
        M,
        N,
        H=H,
        D=D,
        BLOCK_N=BLOCK_N,
    )
    return logits


def _mqa_logits_torch(
    q_fp8: torch.Tensor,  # (M, H, D)
    k_fp8: torch.Tensor,  # (N, D)
    k_scale: torch.Tensor,  # (N,) fp32
    weights: torch.Tensor,  # (M, H) fp32
    cu_seqlen_ks: torch.Tensor,  # (M,) int32
    cu_seqlen_ke: torch.Tensor,  # (M,) int32
) -> torch.Tensor:
    """Reference impl mirroring ``fp8_mqa_logits_torch`` (DeepGEMM test). Only
    used for unit tests; production should always go through the Triton path
    because this materializes a (H, M, N) fp32 intermediate.
    """
    N = k_fp8.shape[0]
    q = q_fp8.to(torch.bfloat16)
    k = k_fp8.to(torch.bfloat16)

    arange_n = torch.arange(N, device=q.device)
    mask = (arange_n[None, :] >= cu_seqlen_ks[:, None]) & (
        arange_n[None, :] < cu_seqlen_ke[:, None]
    )

    # (H, M, N) fp32; relu must be applied per-head BEFORE the weighted sum.
    score = torch.einsum("mhd,nd->hmn", q, k).float() * k_scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


def _mqa_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the Triton kernel when available; fall back to torch
    reference for environments without a working Triton runtime."""
    if HAS_TRITON:
        return _mqa_logits_triton(
            q_fp8, k_fp8, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke
        )
    return _mqa_logits_torch(
        q_fp8, k_fp8, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke
    )


def _mqa_logits_paged_torch(
    q_fp8: torch.Tensor,  # (B, next_n, H, D)
    kv_cache_4d: torch.Tensor,  # (num_blocks, block_size, 1, D + scale_pad)
    weights: torch.Tensor,  # (B*next_n, H) fp32
    context_lens: torch.Tensor,  # (B,) int32 (or (B, next_n))
    block_tables: torch.Tensor,  # (B, max_blocks) int32
    max_model_len: int,
    head_dim: int,
) -> torch.Tensor:
    """Per-batch torch implementation of the paged MQA-logits compute. Walks
    each batch element's block_table, dequantizes the FP8 K-cache slot, and
    accumulates per-head relu-weighted logits. Slow but correct, and only
    materializes one block's worth of intermediate at a time.

    Mirrors ``fp8_paged_mqa_logits_torch`` in ``rocm_aiter_mla_sparse.py`` but
    keeps the (H, ...) intermediate scoped to a single block.
    """
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, H, D = q_fp8.shape

    # Cache layout: last dim = D fp8 + 4 byte (1 fp32) scale per token.
    kv_values = kv_cache_4d[..., :head_dim]  # uint8
    kv_scale = kv_cache_4d[..., head_dim:]  # uint8 (4 bytes per slot)

    num_block, block_size, _, _ = kv_values.size()

    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )

    # Normalize context_lens to (B,).
    if context_lens.dim() == 2:
        context_lens_b = context_lens[:, 0]
    else:
        context_lens_b = context_lens
    ctx_lens = context_lens_b.tolist()

    q_bf16 = q_fp8.to(torch.bfloat16)
    weights_f32 = weights.to(torch.float32)

    for i in range(batch_size):
        ctx_len = ctx_lens[i]
        if ctx_len <= 0:
            continue
        # Per-token weight slice for this batch element.
        # weight_slice shape: (H, next_n)
        weight_slice = (
            weights_f32[i * next_n : (i + 1) * next_n, :]
            .transpose(0, 1)
            .contiguous()
        )

        for block_rk in range(cdiv(ctx_len, block_size)):
            phys_block = int(block_tables[i, block_rk].item())
            # K block: (block_size, D) bf16 = fp8 dequant * fp32 scale.
            k_fp8_block = (
                kv_values[phys_block, :, 0, :]
                .view(fp8_dtype)
                .to(torch.bfloat16)
            )
            k_scale_block = (
                kv_scale[phys_block, :, 0, :].contiguous().view(torch.float32)
            )  # (block_size, 1)
            k_block_bf16 = k_fp8_block * k_scale_block.to(torch.bfloat16)

            # Compute (H, next_n, block_size) scores in fp32.
            qx = q_bf16[i]  # (next_n, H, D)
            score = (
                torch.einsum("nhd,sd->hns", qx, k_block_bf16).float()
            )

            # Per-head relu before weighting. weight_slice: (H, next_n)
            score = score.relu() * weight_slice.unsqueeze(-1)
            block_logits = score.sum(dim=0)  # (next_n, block_size)

            # Mask k positions beyond ctx_len within this block.
            n_start = block_rk * block_size
            n_end = min((block_rk + 1) * block_size, ctx_len)
            valid = n_end - n_start
            if valid <= 0:
                continue
            logits[
                i * next_n : (i + 1) * next_n,
                n_start:n_end,
            ] = block_logits[:, :valid]

    return logits


# ---------------------------------------------------------------------------
# Custom op: orchestration that mirrors the CUDA sparse_attn_indexer body
# but assumes ``skip_k_cache_insert=True`` (the V4 layout) and uses only
# ROCm-available helpers.
# ---------------------------------------------------------------------------
def rocm_sparse_attn_indexer_no_insert(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # Profile-run path: no real attn_metadata; just reserve workspace and
    # the dummy logits buffer for the memory profiler (matches the shape /
    # dtype the runtime path will actually use).
    if not isinstance(attn_metadata, dict):
        values_spec, scales_spec = _gather_workspace_shapes_fp8(
            total_seq_lens, head_dim, fp8_dtype
        )
        current_workspace_manager().get_simultaneous(values_spec, scales_spec)
        max_logits_elems = (
            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        )
        _ = torch.empty(
            max_logits_elems,
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        if topk_indices_buffer is None:
            return torch.empty(
                (hidden_states.shape[0], topk_tokens),
                dtype=torch.int32,
                device=hidden_states.device,
            )
        return topk_indices_buffer

    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None

    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens

    # NOTE: K-cache insert is INTENTIONALLY skipped here. DeepSeek-V4's
    # compressor (DeepseekCompressor.forward) writes the compressed K to the
    # indexer's KV cache via its fused triton kernel before this op is called,
    # and the call site passes k=None.

    topk_indices_buffer[: hidden_states.shape[0]] = -1

    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            # Reuse the workspace to gather the FP8 K + scale for this chunk.
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes_fp8(
                total_seq_lens, head_dim, fp8_dtype
            )
            k_quant_full, k_scale_full = (
                workspace_manager.get_simultaneous(values_spec, scales_spec)
            )
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_quant,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
            )

            q_slice = q_fp8[chunk.token_start : chunk.token_end]
            w_slice = weights[chunk.token_start : chunk.token_end]
            k_scale_f32 = k_scale.view(torch.float32).squeeze(-1)

            logits = _mqa_logits(
                q_slice,
                k_quant,
                k_scale_f32,
                w_slice,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )

            num_rows = logits.shape[0]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            torch.ops._C.top_k_per_row_prefill(
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
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None

        # The kv_cache stored shape is (num_blocks, block_size, head_dim+pad);
        # paged-mqa-logits expects an extra "n_head" singleton dim.
        kv_cache_4d = kv_cache.unsqueeze(-2)

        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )

        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        # Slow-but-correct paged compute. Future Triton kernel TODO: walk the
        # block_table on-device to avoid the per-batch python loop and the
        # per-block (H, next_n, block_size) intermediate.
        logits = _mqa_logits_paged_torch(
            padded_q_fp8_decode_tokens,
            kv_cache_4d,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            max_model_len,
            head_dim,
        )

        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[
                : topk_indices.shape[0], : topk_indices.shape[-1]
            ] = topk_indices

    return topk_indices_buffer


def rocm_sparse_attn_indexer_no_insert_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # Mirror rocm_aiter_sparse_attn_indexer_fake's profile-run estimate so
    # vllm's memory profiler accounts for the gather workspace.
    fp8_dtype = current_platform.fp8_dtype()
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4],
        device=q_fp8.device,
        dtype=torch.uint8,
    )
    _ = _flattened_kv[..., :head_dim].view(fp8_dtype).contiguous()
    _ = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    if topk_indices_buffer is None:
        return torch.empty(
            (hidden_states.shape[0], topk_tokens),
            dtype=torch.int32,
            device=q_fp8.device,
        )
    return topk_indices_buffer


# Register as a vllm custom op so vllm's compile / dispatch infrastructure
# treats it the same as the existing sparse_attn_indexer ops.
direct_register_custom_op(
    op_name="rocm_sparse_attn_indexer_no_insert",
    op_func=rocm_sparse_attn_indexer_no_insert,
    mutates_args=["topk_indices_buffer"],
    fake_impl=rocm_sparse_attn_indexer_no_insert_fake,
    dispatch_key=current_platform.dispatch_key,
)
