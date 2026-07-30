# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged gather + non-causal sparse decode attention for ZoomKV."""

from __future__ import annotations

import os
from functools import lru_cache

import torch
import torch.nn.functional as F


def logical_to_physical_slots(
    logical_token_ids: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map logical token indices to physical slots.

    Args:
        logical_token_ids: [...] int64, -1 for invalid
        block_table: [max_blocks] physical block ids for one request
        block_size: tokens per block
    Returns:
        physical_slots: same shape, -1 for invalid
    """
    ids = logical_token_ids.to(torch.int64)
    valid = ids >= 0
    block_idx = torch.div(ids.clamp(min=0), block_size, rounding_mode="floor")
    offset = torch.remainder(ids.clamp(min=0), block_size)
    max_b = block_table.shape[0]
    block_idx_c = block_idx.clamp(0, max_b - 1)
    phys_block = block_table[block_idx_c]
    slots = phys_block.to(torch.int64) * block_size + offset
    return torch.where(valid, slots, torch.full_like(slots, -1))


def gather_kv_by_logical_indices(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_token_ids: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V for logical token indices of one request.

    Args:
        key_cache / value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: [max_blocks]
        logical_token_ids: [num_kv_heads, n_tok] or [n_tok]
    Returns:
        gathered_k/v: [num_kv_heads, n_tok, head_dim]
    """
    if logical_token_ids.dim() == 1:
        logical_token_ids = logical_token_ids.unsqueeze(0).expand(
            key_cache.shape[2], -1
        )
    if key_cache.is_cuda:
        from vllm.v1.attention.ops.zoomkv.paged_triton import paged_gather_kv

        return paged_gather_kv(
            key_cache,
            value_cache,
            block_table,
            logical_token_ids,
            block_size,
        )
    kv_heads, n_tok = logical_token_ids.shape
    head_dim = key_cache.shape[-1]

    # Flatten cache for gather: [num_blocks * block_size, kv, D]
    num_blocks = key_cache.shape[0]
    flat_k = key_cache.reshape(num_blocks * block_size, kv_heads, head_dim)
    flat_v = value_cache.reshape(num_blocks * block_size, kv_heads, head_dim)
    slots = logical_to_physical_slots(logical_token_ids, block_table, block_size)
    valid = slots >= 0
    slots = slots.clamp(min=0, max=flat_k.shape[0] - 1)
    heads = torch.arange(
        kv_heads, device=key_cache.device, dtype=torch.int64
    ).unsqueeze(1)
    out_k = flat_k[slots, heads]
    out_v = flat_v[slots, heads]
    out_k.masked_fill_(~valid.unsqueeze(-1), 0)
    out_v.masked_fill_(~valid.unsqueeze(-1), 0)
    return out_k, out_v


def gather_kv_by_logical_indices_batch(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_token_ids: torch.Tensor,
    block_size: int,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched gather from paged KV cache.

    Args:
        key_cache / value_cache: [num_blocks, block_size, Hkv, D]
        block_table: [B, max_blocks]
        logical_token_ids: [B, Hkv, n_tok]
    Returns:
        gathered_k/v: [B, Hkv, n_tok, D]
    """
    if logical_token_ids.dim() != 3:
        raise ValueError(
            "batched gather expects logical_token_ids [B, Hkv, n_tok], "
            f"got {tuple(logical_token_ids.shape)}"
        )
    if key_cache.is_cuda:
        from vllm.v1.attention.ops.zoomkv.paged_triton import paged_gather_kv_batch

        return paged_gather_kv_batch(
            key_cache,
            value_cache,
            block_table,
            logical_token_ids,
            block_size,
            out_k=out_k,
            out_v=out_v,
        )
    batch, kv_heads, n_tok = logical_token_ids.shape
    outs_k = []
    outs_v = []
    for i in range(batch):
        gk, gv = gather_kv_by_logical_indices(
            key_cache,
            value_cache,
            block_table[i],
            logical_token_ids[i],
            block_size,
        )
        outs_k.append(gk)
        outs_v.append(gv)
    return torch.stack(outs_k, dim=0), torch.stack(outs_v, dim=0)


def gather_kv_from_topk_batch(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_logical: torch.Tensor,
    block_size: int,
    sink_size: int,
    local_size: int,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused sink/local/topk assembly + paged gather for a decode batch.

    Direct fully-valid path uses this to avoid materializing ``ctx_idx`` /
    ``_ctx_valid`` and launching a separate assemble kernel.
    """
    if topk_logical.dim() != 3:
        raise ValueError(
            "gather_kv_from_topk_batch expects topk [B, Hkv, K], "
            f"got {tuple(topk_logical.shape)}"
        )
    if key_cache.is_cuda:
        from vllm.v1.attention.ops.zoomkv.paged_triton import (
            paged_gather_kv_from_topk_batch,
        )

        return paged_gather_kv_from_topk_batch(
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            topk_logical,
            block_size,
            sink_size,
            local_size,
            out_k=out_k,
            out_v=out_v,
        )
    # CPU / reference fallback: assemble then gather.
    ctx_idx, _ = assemble_sparse_context_indices_batch(
        seq_lens,
        topk_logical,
        sink_size,
        local_size,
    )
    return gather_kv_by_logical_indices_batch(
        key_cache,
        value_cache,
        block_table,
        ctx_idx,
        block_size,
        out_k=out_k,
        out_v=out_v,
    )


def build_sink_local_indices(
    seq_len: int,
    sink_size: int,
    local_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Absolute token indices for sink + local windows."""
    sink = torch.arange(min(sink_size, seq_len), device=device, dtype=torch.int64)
    local_start = max(sink_size, seq_len - local_size)
    local = torch.arange(local_start, seq_len, device=device, dtype=torch.int64)
    # Drop local tokens already covered by sink.
    if local.numel() and sink.numel():
        local = local[local >= sink[-1] + 1]
    return torch.cat([sink, local], dim=0)


def gather_kv_hybrid(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_token_ids: torch.Tensor,
    block_size: int,
    cpu_pool,
    layer_name: str,
    retrieval_start_block: int,
    retrieval_end_block: int,
    cpu_slots: torch.Tensor | None = None,
    any_offloaded: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V for sparse decode with K+V CPU offload.

    Key and Value are read from GPU for hot blocks and overwritten from the
    pinned CPU KV pool for cold physical blocks (whose GPU pages have been
    zeroed, per ``cpu_pool``'s offloaded mask).

    ``cpu_slots`` may be passed pre-computed for blocks
    ``block_table[retrieval_start_block:retrieval_end_block]`` along with a
    host-side ``any_offloaded`` flag to avoid re-deriving them (and the
    GPU->CPU synchronization that entails).
    """
    if logical_token_ids.dim() == 1:
        logical_token_ids = logical_token_ids.unsqueeze(0).expand(
            key_cache.shape[2], -1
        )
    out_k, out_v = gather_kv_by_logical_indices(
        key_cache, value_cache, block_table, logical_token_ids, block_size
    )
    if retrieval_end_block <= retrieval_start_block:
        return out_k, out_v

    # Physical ids covering the hybrid zone (logical block order).
    bt = block_table
    max_b = bt.shape[0]
    start = max(0, int(retrieval_start_block))
    end = min(int(retrieval_end_block), max_b)
    if end <= start:
        return out_k, out_v
    if cpu_slots is None:
        phys_ids = bt[start:end]
        cpu_slots = cpu_pool.lookup_slots_for_physical_ids(layer_name, phys_ids)
    if any_offloaded is None:
        any_offloaded = bool((cpu_slots >= 0).any().item())
    if not any_offloaded:
        return out_k, out_v

    offloaded_mask = cpu_pool.ensure_offload_mask(layer_name, key_cache.shape[0])
    from vllm.v1.attention.ops.zoomkv.kernels import h2d_fill_keys_hybrid

    h2d_fill_keys_hybrid(
        cpu_pool.key[layer_name],
        logical_token_ids,
        bt,
        cpu_slots,
        offloaded_mask,
        start,
        out_k,
    )
    # Value is offloaded together with Key; restore it from the same slots.
    value_pool = getattr(cpu_pool, "value", None)
    if value_pool is not None and layer_name in value_pool:
        h2d_fill_keys_hybrid(
            value_pool[layer_name],
            logical_token_ids,
            bt,
            cpu_slots,
            offloaded_mask,
            start,
            out_v,
        )
    # Approximate H2D traffic (upper bound; avoids a GPU sync per call).
    tok_bytes = out_k.element_size() * out_k.shape[-1] * out_k.shape[0]
    cpu_pool.metrics.h2d_bytes += 2 * tok_bytes * out_k.shape[1]
    cpu_pool.metrics.h2d_events += 1
    return out_k, out_v


def sparse_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Non-causal SDPA over retrieved KV for a single decode token.

    Args:
        query: [1, num_q_heads, head_dim]
        key / value: [num_kv_heads, n_ctx, head_dim]
        valid_mask: [num_kv_heads, n_ctx] bool (optional)
    Returns:
        out: [1, num_q_heads, head_dim]
    """
    hq = query.shape[1]
    hkv = key.shape[0]
    assert hq % hkv == 0
    # ZoomKV's retrieval range excludes sink/local, so the production path has
    # a fixed, fully-valid K/V width. Use vLLM's FlashAttention extension
    # directly instead of generic SDPA, which otherwise selects a much slower
    # masked GQA path for q_len=1, head_dim=256.
    if query.is_cuda and valid_mask is None:
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            n_ctx = key.shape[1]
            cu_q, cu_k = _flash_cu_seqlens(query.device, n_ctx)
            # q: [total_q=1, Hq, D], k/v: [total_k, Hkv, D].
            # FlashAttention supports GQA natively and only requires the last
            # dimension to be contiguous.
            k_flash = key.permute(1, 0, 2)
            v_flash = value.permute(1, 0, 2)
            return flash_attn_varlen_func(
                query,
                k_flash,
                v_flash,
                max_seqlen_q=1,
                cu_seqlens_q=cu_q,
                max_seqlen_k=n_ctx,
                cu_seqlens_k=cu_k,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=False,
            )
        except (ImportError, RuntimeError):
            # Unit-test environments and unsupported GPUs retain the reference
            # path. Strict production dispatch is checked by the backend.
            if os.environ.get("VLLM_ZOOMKV_STRICT_KERNELS", "0") == "1":
                raise

    repeats = hq // hkv
    k = key.unsqueeze(0)  # [1, Hkv, T, D]
    v = value.unsqueeze(0)
    q = query.unsqueeze(2)  # [1, Hq, 1, D]
    attn_mask = None
    if valid_mask is not None:
        m = valid_mask.repeat_interleave(repeats, dim=0)  # [Hq, T]
        # Bool SDPA mask avoids constructing and reading an fp32 additive mask.
        attn_mask = m.unsqueeze(0).unsqueeze(2)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=hq != hkv,
    )
    return out.squeeze(2)  # [1, Hq, D]


def sparse_decode_attention_batch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched non-causal sparse decode attention.

    Args:
        query: [B, Hq, D]
        key / value: [B, Hkv, n_ctx, D]
        valid_mask: [B, Hkv, n_ctx] bool (optional). When provided and not
            all-True, falls back to per-request SDPA. The common long-context
            path has a fully-valid context and skips this.
    Returns:
        out: [B, Hq, D]
    """
    batch, hq, d = query.shape
    hkv = key.shape[1]
    n_ctx = key.shape[2]
    assert hq % hkv == 0
    # Avoid GPU sync: if the caller knows the mask is fully valid it passes
    # valid_mask=None (the common long-context path used by the batched
    # backend). Otherwise fall through to masked SDPA.
    if query.is_cuda and valid_mask is None:
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            cu_q, cu_k = _flash_cu_seqlens_batch(query.device, batch, n_ctx)
            # Flash varlen wants packed [total_q, Hq, D] / [total_k, Hkv, D].
            q_flat = query.contiguous()
            k_flat = (
                key.permute(0, 2, 1, 3).reshape(batch * n_ctx, hkv, d).contiguous()
            )
            v_flat = (
                value.permute(0, 2, 1, 3).reshape(batch * n_ctx, hkv, d).contiguous()
            )
            return flash_attn_varlen_func(
                q_flat,
                k_flat,
                v_flat,
                max_seqlen_q=1,
                cu_seqlens_q=cu_q,
                max_seqlen_k=n_ctx,
                cu_seqlens_k=cu_k,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=False,
            )
        except (ImportError, RuntimeError):
            if os.environ.get("VLLM_ZOOMKV_STRICT_KERNELS", "0") == "1":
                raise

    # Reference / masked path: one SDPA call with GQA.
    repeats = hq // hkv
    q = query.unsqueeze(2)  # [B, Hq, 1, D]
    attn_mask = None
    if valid_mask is not None:
        m = valid_mask.repeat_interleave(repeats, dim=1)  # [B, Hq, T]
        attn_mask = m.unsqueeze(2)
    out = F.scaled_dot_product_attention(
        q,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=hq != hkv,
    )
    return out.squeeze(2)


@lru_cache(maxsize=128)
def _flash_cu_seqlens(
    device: torch.device, n_ctx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache tiny varlen descriptors; steady-state decode performs no alloc."""
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, int(n_ctx)], dtype=torch.int32, device=device)
    return cu_q, cu_k


@lru_cache(maxsize=128)
def _flash_cu_seqlens_batch(
    device: torch.device, batch: int, n_ctx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Varlen descriptors for B independent q_len=1 / k_len=n_ctx sequences."""
    cu_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    cu_k = torch.arange(
        0, (batch + 1) * int(n_ctx), step=int(n_ctx), dtype=torch.int32, device=device
    )
    return cu_q, cu_k


def assemble_sparse_context_indices(
    seq_len: int,
    topk_logical: torch.Tensor,
    sink_size: int,
    local_size: int,
    device: torch.device,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine sink + local + per-head topk into fixed-width index + mask.

    Args:
        topk_logical: [kv_heads, final_topk]
        out: optional preallocated [kv_heads, sink+local+final_topk] buffer
    Returns:
        indices: [kv_heads, sink_local + final_topk]
        valid_mask: [kv_heads, sink_local + final_topk] bool
    """
    if topk_logical.is_cuda:
        # Single fused Triton launch. Writes into the preallocated ``out``
        # buffer when provided (allocation-free hot path); otherwise allocates.
        from vllm.v1.attention.ops.zoomkv.paged_triton import assemble_context

        return assemble_context(
            seq_len,
            topk_logical,
            sink_size,
            local_size,
            out=out,
        )
    sink_local = build_sink_local_indices(seq_len, sink_size, local_size, device)
    kv, tk = topk_logical.shape
    n_ctx = sink_local.numel() + tk
    if out is not None and out.shape != (kv, n_ctx):
        # Width may differ slightly if config changed; fall back to alloc.
        out = None
    if out is None:
        indices = torch.full((kv, n_ctx), -1, dtype=torch.int64, device=device)
    else:
        indices = out
        indices.fill_(-1)
    valid = torch.zeros(kv, n_ctx, dtype=torch.bool, device=device)
    if sink_local.numel():
        indices[:, : sink_local.numel()] = sink_local.view(1, -1).expand(kv, -1)
        valid[:, : sink_local.numel()] = True
    # Retriever candidates come exclusively from [sink, local_start), so they
    # cannot overlap either fixed region. Avoid two masks + a where on every
    # full-attention layer and decode token.
    toks = topk_logical
    indices[:, sink_local.numel() :] = toks
    valid[:, sink_local.numel() :] = toks >= 0
    return indices, valid


def assemble_sparse_context_indices_batch(
    seq_lens: torch.Tensor,
    topk_logical: torch.Tensor,
    sink_size: int,
    local_size: int,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched assemble of sink + local + topk indices.

    Args:
        seq_lens: [B]
        topk_logical: [B, Hkv, final_topk]
        out: optional [B, Hkv, sink+local+final_topk]
    Returns:
        indices: [B, Hkv, n_ctx]
        valid_mask: [B, Hkv, n_ctx]
    """
    if topk_logical.is_cuda:
        from vllm.v1.attention.ops.zoomkv.paged_triton import assemble_context_batch

        return assemble_context_batch(
            seq_lens,
            topk_logical,
            sink_size,
            local_size,
            out=out,
        )
    batch, kv, tk = topk_logical.shape
    device = topk_logical.device
    n_ctx = int(sink_size) + int(local_size) + tk
    if out is None or out.shape != (batch, kv, n_ctx):
        indices = torch.full(
            (batch, kv, n_ctx), -1, dtype=torch.int64, device=device
        )
    else:
        indices = out
        indices.fill_(-1)
    valid = torch.zeros(batch, kv, n_ctx, dtype=torch.bool, device=device)
    seq_list = [int(x) for x in seq_lens.tolist()]
    for i, seq_len in enumerate(seq_list):
        sink_local = build_sink_local_indices(
            seq_len, sink_size, local_size, device
        )
        sl = sink_local.numel()
        if sl:
            indices[i, :, :sl] = sink_local.view(1, -1).expand(kv, -1)
            valid[i, :, :sl] = True
        indices[i, :, sl : sl + tk] = topk_logical[i]
        valid[i, :, sl : sl + tk] = topk_logical[i] >= 0
    return indices, valid
