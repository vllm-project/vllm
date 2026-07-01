# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl


@triton.jit
def _dspark_qkv_postprocess_kernel(
    q_ptr,
    q_out_ptr,
    kv_ptr,
    kv_out_ptr,
    positions_ptr,
    cos_sin_ptr,
    eps: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    token_pid = tl.program_id(0)
    head_pid = tl.program_id(1)

    offs = tl.arange(0, block_d)
    mask = offs < head_dim
    rope_half: tl.constexpr = rope_dim // 2
    nope_dim: tl.constexpr = head_dim - rope_dim

    q_base = (token_pid * n_heads + head_pid) * head_dim
    q = tl.load(q_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(q * q, axis=0) / head_dim
    rrms = tl.rsqrt(variance + eps)
    q_norm = (q * rrms).to(tl.bfloat16).to(tl.float32)

    pos = tl.load(positions_ptr + token_pid).to(tl.int64)
    rope_offsets = offs - nope_dim
    pair_idx = rope_offsets // 2
    pair_base = nope_dim + pair_idx * 2
    even = tl.load(q_ptr + q_base + pair_base, mask=mask, other=0.0).to(tl.float32)
    odd = tl.load(q_ptr + q_base + pair_base + 1, mask=mask, other=0.0).to(tl.float32)
    even = (even * rrms).to(tl.bfloat16).to(tl.float32)
    odd = (odd * rrms).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(
        cos_sin_ptr + pos * rope_dim + pair_idx,
        mask=(offs >= nope_dim) & (pair_idx < rope_half),
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * rope_dim + rope_half + pair_idx,
        mask=(offs >= nope_dim) & (pair_idx < rope_half),
        other=0.0,
    ).to(tl.float32)
    q_rope = tl.where(
        rope_offsets % 2 == 0,
        even * cos - odd * sin,
        odd * cos + even * sin,
    )
    q_out = tl.where(offs < nope_dim, q_norm, q_rope)
    tl.store(q_out_ptr + q_base + offs, q_out, mask=mask)

    if head_pid == 0:
        kv = tl.load(kv_ptr + token_pid * head_dim + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        kv_even = tl.load(
            kv_ptr + token_pid * head_dim + pair_base,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        kv_odd = tl.load(
            kv_ptr + token_pid * head_dim + pair_base + 1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        kv_rope = tl.where(
            rope_offsets % 2 == 0,
            kv_even * cos - kv_odd * sin,
            kv_odd * cos + kv_even * sin,
        )
        kv_out = tl.where(offs < nope_dim, kv, kv_rope)
        tl.store(kv_out_ptr + token_pid * head_dim + offs, kv_out, mask=mask)


def dspark_qkv_postprocess(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse DSpark q no-weight RMSNorm+RoPE and KV RoPE.

    This matches the existing PyTorch reference order: q is RMS-normalized,
    rounded back to BF16, then RoPE is applied on the tail dimensions.
    """
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {q.shape}")
    if kv.dim() != 2:
        raise ValueError(f"kv must be [tokens, dim], got {kv.shape}")
    if q.shape[0] != kv.shape[0] or q.shape[2] != kv.shape[1]:
        raise ValueError(f"q/kv shape mismatch: q={q.shape}, kv={kv.shape}")
    if not q.is_contiguous() or not kv.is_contiguous():
        raise ValueError("q and kv must be contiguous")
    if q.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise ValueError("DSpark Triton q/kv postprocess currently requires BF16")

    num_tokens, n_heads, head_dim = q.shape
    if num_tokens == 0:
        return torch.empty_like(q), torch.empty_like(kv)
    rope_dim = cos_sin_cache.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)
    _dspark_qkv_postprocess_kernel[(num_tokens, n_heads)](
        q,
        q_out,
        kv,
        kv_out,
        positions.contiguous(),
        cos_sin_cache,
        eps=eps,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        block_d=block_d,
        num_warps=8,
    )
    return q_out, kv_out


@triton.jit
def _dspark_inv_rope_bf16_layout_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_ptr,
    out_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_token,
    out_stride_group,
    out_stride_hidden,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    token_pid = tl.program_id(0).to(tl.int64)
    head_pid = tl.program_id(1).to(tl.int64)
    group = head_pid // heads_per_group
    head_in_group = head_pid - group * heads_per_group

    offs = tl.arange(0, block_d)
    mask = offs < head_dim
    input_base = o_ptr + token_pid * o_stride_token + head_pid * o_stride_head
    x = tl.load(input_base + offs, mask=mask, other=0.0).to(tl.float32)

    rope_offsets = offs - nope_dim
    is_rope = (offs >= nope_dim) & mask
    pair_offsets = tl.maximum(rope_offsets, 0) ^ 1
    x_partner = tl.load(
        input_base + nope_dim + pair_offsets,
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)

    pos = tl.load(positions_ptr + token_pid).to(tl.int64)
    pair_idx = tl.maximum(rope_offsets // 2, 0)
    rope_half: tl.constexpr = rope_dim // 2
    cos = tl.load(
        cos_sin_ptr + pos * cache_stride_pos + pair_idx,
        mask=is_rope & (pair_idx < rope_half),
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * cache_stride_pos + rope_half + pair_idx,
        mask=is_rope & (pair_idx < rope_half),
        other=0.0,
    ).to(tl.float32)
    x_add = x * cos + x_partner * sin
    x_sub = x * cos - x_partner * sin
    rotated = tl.where((rope_offsets & 1) == 0, x_add, x_sub)
    out = tl.where(is_rope, rotated, x)

    hidden_offsets = head_in_group * head_dim + offs
    out_base = out_ptr + token_pid * out_stride_token + group * out_stride_group
    tl.store(
        out_base + hidden_offsets * out_stride_hidden,
        out,
        mask=mask,
    )


def dspark_inv_rope_bf16_layout(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Inverse-RoPE attention output into grouped BF16 WO_A layout.

    Output shape is [tokens, groups, heads_per_group * head_dim], matching the
    activation layout consumed by DSpark WO_A.
    """
    if o.dim() != 3:
        raise ValueError(f"o must be [tokens, heads, dim], got {o.shape}")
    if o.dtype is not torch.bfloat16:
        raise ValueError("DSpark BF16 inverse-RoPE layout currently requires BF16")
    if not o.is_contiguous():
        raise ValueError("o must be contiguous")
    num_tokens, num_heads, head_dim = o.shape
    if num_tokens == 0:
        return torch.empty(
            (0, n_groups, heads_per_group * head_dim),
            dtype=o.dtype,
            device=o.device,
        )
    if num_heads != n_groups * heads_per_group:
        raise ValueError(
            f"head/group mismatch: heads={num_heads}, "
            f"n_groups={n_groups}, heads_per_group={heads_per_group}"
        )
    if head_dim != nope_dim + rope_dim:
        raise ValueError(
            f"head_dim={head_dim} does not match nope+rope={nope_dim + rope_dim}"
        )
    if cos_sin_cache.shape[-1] != rope_dim:
        raise ValueError(
            f"cos_sin_cache last dim must be {rope_dim}, got {cos_sin_cache.shape}"
        )
    out = torch.empty(
        (num_tokens, n_groups, heads_per_group * head_dim),
        dtype=o.dtype,
        device=o.device,
    )
    block_d = triton.next_power_of_2(head_dim)
    _dspark_inv_rope_bf16_layout_kernel[(num_tokens, num_heads)](
        o,
        positions.contiguous(),
        cos_sin_cache,
        out,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        out_stride_token=out.stride(0),
        out_stride_group=out.stride(1),
        out_stride_hidden=out.stride(2),
        head_dim=head_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        block_d=block_d,
        num_warps=8,
    )
    return out


@triton.jit
def _dspark_context_kv_store_kernel(
    kv_ptr,
    cache_ptr,
    positions_ptr,
    query_start_loc_ptr,
    rejected_ptr,
    kv_weight_ptr,
    cos_sin_ptr,
    eps: tl.constexpr,
    kv_stride: tl.constexpr,
    cache_batch_stride: tl.constexpr,
    cache_window_stride: tl.constexpr,
    batch_size: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    window_size: tl.constexpr,
    block_d: tl.constexpr,
    has_rejected: tl.constexpr,
):
    token_pid = tl.program_id(0)

    req_idx = tl.full((), 0, dtype=tl.int64)
    should_store = tl.full((), False, dtype=tl.int1)
    for batch_idx in tl.static_range(0, batch_size):
        start = tl.load(query_start_loc_ptr + batch_idx).to(tl.int64)
        end = tl.load(query_start_loc_ptr + batch_idx + 1).to(tl.int64)
        if has_rejected:
            end -= tl.load(rejected_ptr + batch_idx).to(tl.int64)
        in_request = (token_pid >= start) & (token_pid < end)
        req_idx = tl.where(in_request, batch_idx, req_idx)
        should_store = should_store | in_request

    offs = tl.arange(0, block_d)
    mask = offs < head_dim
    rope_half: tl.constexpr = rope_dim // 2
    nope_dim: tl.constexpr = head_dim - rope_dim

    row = kv_ptr + token_pid * kv_stride
    x = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / head_dim
    rrms = tl.rsqrt(variance + eps)
    weight = tl.load(kv_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    norm = (x * rrms * weight).to(tl.bfloat16).to(tl.float32)

    pos = tl.load(positions_ptr + token_pid).to(tl.int64)
    slot = pos % window_size
    rope_offsets = offs - nope_dim
    pair_idx = rope_offsets // 2
    pair_base = nope_dim + pair_idx * 2
    rope_mask = (offs >= nope_dim) & (pair_idx < rope_half)

    even_x = tl.load(row + pair_base, mask=rope_mask, other=0.0).to(tl.float32)
    odd_x = tl.load(row + pair_base + 1, mask=rope_mask, other=0.0).to(tl.float32)
    even_w = tl.load(kv_weight_ptr + pair_base, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    odd_w = tl.load(kv_weight_ptr + pair_base + 1, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    even = (even_x * rrms * even_w).to(tl.bfloat16).to(tl.float32)
    odd = (odd_x * rrms * odd_w).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(
        cos_sin_ptr + pos * rope_dim + pair_idx,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * rope_dim + rope_half + pair_idx,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    rope = tl.where(
        rope_offsets % 2 == 0,
        even * cos - odd * sin,
        odd * cos + even * sin,
    )
    out = tl.where(offs < nope_dim, norm, rope)

    cache_row = cache_ptr + req_idx * cache_batch_stride + slot * cache_window_stride
    tl.store(cache_row + offs, out, mask=mask & should_store)


def dspark_context_kv_store(
    kv: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    batch_size: int,
    num_rejected_tokens: torch.Tensor | None,
    kv_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> None:
    """Fuse DSpark context KV RMSNorm+RoPE and circular cache scatter."""
    if kv.dim() != 2:
        raise ValueError(f"kv must be [tokens, dim], got {kv.shape}")
    if cache.dim() != 3:
        raise ValueError(f"cache must be [batch, window, dim], got {cache.shape}")
    if kv.shape[1] != cache.shape[2]:
        raise ValueError(
            f"kv/cache head-dim mismatch: kv={kv.shape}, cache={cache.shape}"
        )
    if kv.stride(-1) != 1 or cache.stride(-1) != 1:
        raise ValueError("kv and cache must have contiguous last dimensions")
    if kv.dtype is not torch.bfloat16 or cache.dtype is not torch.bfloat16:
        raise ValueError("DSpark context KV store currently requires BF16 kv/cache")

    num_tokens, head_dim = kv.shape
    if num_tokens == 0:
        return
    rope_dim = cos_sin_cache.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    rejected = (
        num_rejected_tokens if num_rejected_tokens is not None else query_start_loc
    )
    _dspark_context_kv_store_kernel[(num_tokens,)](
        kv,
        cache,
        positions.contiguous(),
        query_start_loc.contiguous(),
        rejected.contiguous(),
        kv_weight,
        cos_sin_cache,
        eps=eps,
        kv_stride=kv.stride(0),
        cache_batch_stride=cache.stride(0),
        cache_window_stride=cache.stride(1),
        batch_size=batch_size,
        head_dim=head_dim,
        rope_dim=rope_dim,
        window_size=cache.shape[1],
        block_d=block_d,
        has_rejected=num_rejected_tokens is not None,
        num_warps=8,
    )


@triton.jit
def _dspark_attention_kernel(
    q_ptr,
    main_kv_ptr,
    draft_kv_ptr,
    main_pos_ptr,
    sink_ptr,
    out_ptr,
    scale: tl.constexpr,
    block_size: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    window_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """KV-shared tensor-core DSpark draft attention.

    DSpark draft attention is head-less on the KV side: the main-window KV and
    the draft-block KV are shared across all heads, so every (draft-token, head)
    query for a batch element attends the SAME [window + block, head_dim] KV
    under the SAME validity mask. A one-program-per-(token, head) launch would
    therefore re-read that KV block_size*n_heads times per batch element. This
    kernel instead tiles the block_size*n_heads query rows (BLOCK_M at a time),
    streams the shared KV once per tile in BLOCK_N chunks, and runs a flash
    online-softmax with the per-head attention sink folded in as the running-max
    initializer (a keyless logit that contributes to the denominator only).
    QK^T and P@V use tl.dot; KV is bf16 in storage, so P@V runs in bf16.
    """
    batch_idx = tl.program_id(0)
    m_tile = tl.program_id(1)

    rows_per_batch: tl.constexpr = block_size * n_heads
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    m_valid = offs_m < rows_per_batch
    head_of_row = offs_m % n_heads

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # q_flat[batch] is [block*heads, head_dim] contiguous; row = draft*heads + head
    q_base = batch_idx * rows_per_batch * head_dim
    q_ptrs = q_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_valid[:, None] & d_mask[None, :], other=0.0)

    sink = tl.load(sink_ptr + head_of_row, mask=m_valid, other=0.0).to(tl.float32)

    valid_main_end = tl.load(main_pos_ptr + batch_idx)
    valid_main_end = tl.minimum(valid_main_end, window_size - 1)

    # sink folded in as a keyless logit: init running max=sink, denom=1, acc=0
    m_i = sink
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    total_kv: tl.constexpr = window_size + block_size
    for start in range(0, total_kv, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        main_mask = offs_n < window_size
        draft_off = offs_n - window_size
        valid_n = tl.where(main_mask, offs_n <= valid_main_end, draft_off < block_size)

        main_ptrs = (
            main_kv_ptr
            + (batch_idx * window_size + offs_n[:, None]) * head_dim
            + offs_d[None, :]
        )
        draft_ptrs = (
            draft_kv_ptr
            + (batch_idx * block_size + draft_off[:, None]) * head_dim
            + offs_d[None, :]
        )
        kv_ptrs = tl.where(main_mask[:, None], main_ptrs, draft_ptrs)
        kv = tl.load(kv_ptrs, mask=valid_n[:, None] & d_mask[None, :], other=0.0)

        scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale
        scores = tl.where(valid_n[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid_n[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m_i = m_new

    out = acc / l_i[:, None]
    out_ptrs = out_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :]
    tl.store(out_ptrs, out, mask=m_valid[:, None] & d_mask[None, :])


def dspark_triton_attention(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    draft_kv: torch.Tensor,
    main_positions: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Small-shape fused DSpark attention over circular main KV + draft KV."""
    if q.dim() != 4:
        raise ValueError(f"q must be [batch, block, heads, dim], got {q.shape}")
    batch_size, block_size, n_heads, head_dim = q.shape
    window_size = main_kv.shape[1]
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)

    # One program per (batch, query-row tile). BLOCK_M x BLOCK_N x BLOCK_D bf16
    # tiles fit the shared-memory budget with num_stages=1 (the KV loop is only
    # a few iterations, so software pipelining buys nothing here).
    block_m, block_n = 32, 32
    rows_per_batch = block_size * n_heads
    grid = (batch_size, triton.cdiv(rows_per_batch, block_m))
    _dspark_attention_kernel[grid](
        q,
        main_kv,
        draft_kv,
        main_positions,
        attn_sink,
        out,
        scale=scale,
        block_size=block_size,
        n_heads=n_heads,
        head_dim=head_dim,
        window_size=window_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _dspark_markov_argmax_blocks_kernel(
    base_logits_ptr,
    prev_token_ids_ptr,
    w1_ptr,
    w2_ptr,
    block_vals_ptr,
    block_ids_ptr,
    vocab_size: tl.constexpr,
    rank: tl.constexpr,
    base_batch_stride: tl.constexpr,
    w1_vocab_stride: tl.constexpr,
    w2_vocab_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    block_v: tl.constexpr,
    block_r: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    offs_v = block_pid * block_v + tl.arange(0, block_v)
    offs_r = tl.arange(0, block_r)
    v_mask = offs_v < vocab_size
    r_mask = offs_r < rank

    prev_token_id = tl.load(prev_token_ids_ptr + batch_pid).to(tl.int64)
    embed = tl.load(
        w1_ptr + prev_token_id * w1_vocab_stride + offs_r,
        mask=r_mask,
        other=0.0,
    ).to(tl.float32)
    w2 = tl.load(
        w2_ptr + offs_v[:, None] * w2_vocab_stride + offs_r[None, :],
        mask=v_mask[:, None] & r_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    markov_bias = tl.sum(w2 * embed[None, :], axis=1)
    base_logits = tl.load(
        base_logits_ptr + batch_pid * base_batch_stride + offs_v,
        mask=v_mask,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.where(v_mask, base_logits + markov_bias, -float("inf"))
    max_val = tl.max(scores, axis=0)
    max_id = tl.min(tl.where(scores == max_val, offs_v, vocab_size), axis=0)
    tl.store(block_vals_ptr + batch_pid * num_blocks + block_pid, max_val)
    tl.store(block_ids_ptr + batch_pid * num_blocks + block_pid, max_id)


@triton.jit
def _dspark_markov_argmax_reduce_kernel(
    block_vals_ptr,
    block_ids_ptr,
    out_token_ids_ptr,
    num_blocks: tl.constexpr,
    block_nb: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    offs = tl.arange(0, block_nb)
    mask = offs < num_blocks
    vals = tl.load(
        block_vals_ptr + batch_pid * num_blocks + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        block_ids_ptr + batch_pid * num_blocks + offs,
        mask=mask,
        other=2147483647,
    ).to(tl.int64)
    max_val = tl.max(vals, axis=0)
    token_id = tl.min(tl.where(vals == max_val, ids, 2147483647), axis=0)
    tl.store(out_token_ids_ptr + batch_pid, token_id)


def dspark_markov_greedy_argmax(
    base_logits: torch.Tensor,
    prev_token_ids: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    block_vals: torch.Tensor,
    block_ids: torch.Tensor,
    out_token_ids: torch.Tensor,
    *,
    block_v: int = 64,
) -> None:
    """Greedy argmax for base logits plus DSpark Markov bias.

    This is a small-shape greedy-only path. It avoids materializing the full
    Markov-bias vector separately from the base-logit add and argmax.
    """
    if base_logits.dim() != 2:
        raise ValueError(f"base_logits must be [batch, vocab], got {base_logits.shape}")
    if w1_weight.dim() != 2 or w2_weight.dim() != 2:
        raise ValueError("w1_weight and w2_weight must be matrices")
    batch_size, vocab_size = base_logits.shape
    if w1_weight.shape[0] < vocab_size or w2_weight.shape[0] < vocab_size:
        raise ValueError(
            "Markov weights must cover the logits vocabulary: "
            f"logits={base_logits.shape}, w1={w1_weight.shape}, w2={w2_weight.shape}"
        )
    rank = w1_weight.shape[1]
    if w2_weight.shape[1] != rank:
        raise ValueError(
            f"Markov rank mismatch: w1={w1_weight.shape}, w2={w2_weight.shape}"
        )
    if prev_token_ids.shape[0] < batch_size or out_token_ids.shape[0] < batch_size:
        raise ValueError("prev_token_ids and out_token_ids must cover batch_size")

    num_blocks = triton.cdiv(vocab_size, block_v)
    if block_vals.shape[0] < batch_size or block_vals.shape[1] < num_blocks:
        raise ValueError(
            "block_vals too small: "
            f"have {block_vals.shape}, need {(batch_size, num_blocks)}"
        )
    if block_ids.shape[0] < batch_size or block_ids.shape[1] < num_blocks:
        raise ValueError(
            "block_ids too small: "
            f"have {block_ids.shape}, need {(batch_size, num_blocks)}"
        )

    block_r = triton.next_power_of_2(rank)
    _dspark_markov_argmax_blocks_kernel[(batch_size, num_blocks)](
        base_logits,
        prev_token_ids,
        w1_weight,
        w2_weight,
        block_vals,
        block_ids,
        vocab_size=vocab_size,
        rank=rank,
        base_batch_stride=base_logits.stride(0),
        w1_vocab_stride=w1_weight.stride(0),
        w2_vocab_stride=w2_weight.stride(0),
        num_blocks=num_blocks,
        block_v=block_v,
        block_r=block_r,
        num_warps=8,
    )
    _dspark_markov_argmax_reduce_kernel[(batch_size,)](
        block_vals,
        block_ids,
        out_token_ids,
        num_blocks=num_blocks,
        block_nb=triton.next_power_of_2(num_blocks),
        num_warps=8,
    )


@triton.jit
def _dspark_markov_probs_blocks_kernel(
    logits_ptr,
    inv_temp_ptr,
    block_max_ptr,
    block_sumexp_ptr,
    block_maxid_ptr,
    block_gval_ptr,
    block_gid_ptr,
    seed,
    vocab_size: tl.constexpr,
    logits_row_stride: tl.constexpr,
    scratch_stride: tl.constexpr,
    block_v: tl.constexpr,
):
    """Per-vocab-block partial reductions for the fused probabilistic sampler.

    Computes, over one ``block_v`` slice of one request's logits:
    online-softmax block stats (max + relative sum-exp), the plain argmax id
    (greedy fallback), and an in-kernel Gumbel-max (block value + id) using
    ``argmax(z - log(q))`` with ``q ~ Exp(1)`` drawn from Philox. Keeping the
    Gumbel selection in raw-logit space avoids materializing softmax probs or a
    separate exponential-noise tensor.
    """
    batch_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    offs_v = block_pid * block_v + tl.arange(0, block_v)
    v_mask = offs_v < vocab_size

    inv_t = tl.load(inv_temp_ptr + batch_pid).to(tl.float32)
    z = (
        tl.load(
            logits_ptr + batch_pid * logits_row_stride + offs_v,
            mask=v_mask,
            other=-float("inf"),
        ).to(tl.float32)
        * inv_t
    )

    active_mask = v_mask & (z != -float("inf"))
    z_active = tl.where(active_mask, z, -float("inf"))
    bmax = tl.max(z_active, axis=0)
    bmax_for_exp = tl.where(bmax == -float("inf"), 0.0, bmax)
    e = tl.where(active_mask, tl.exp(z_active - bmax_for_exp), 0.0)
    bsum = tl.sum(e, axis=0)
    bmaxid = tl.min(
        tl.where((z_active == bmax) & active_mask, offs_v, vocab_size),
        axis=0,
    )

    # In-kernel Exp(1) noise -> Gumbel-max token selection in raw-logit space.
    rand_offset = batch_pid * vocab_size + offs_v
    u = tl.rand(seed, rand_offset)
    u = tl.maximum(u, 1e-20)
    q = -tl.log(u)
    g = tl.where(active_mask, z_active - tl.log(q), -float("inf"))
    bgval = tl.max(g, axis=0)
    bgid = tl.min(tl.where((g == bgval) & active_mask, offs_v, vocab_size), axis=0)

    out_idx = batch_pid * scratch_stride + block_pid
    tl.store(block_max_ptr + out_idx, bmax)
    tl.store(block_sumexp_ptr + out_idx, bsum)
    tl.store(block_maxid_ptr + out_idx, bmaxid)
    tl.store(block_gval_ptr + out_idx, bgval)
    tl.store(block_gid_ptr + out_idx, bgid)


@triton.jit
def _dspark_markov_probs_reduce_kernel(
    block_max_ptr,
    block_sumexp_ptr,
    block_maxid_ptr,
    block_gval_ptr,
    block_gid_ptr,
    is_greedy_ptr,
    row_max_ptr,
    row_invz_ptr,
    out_tokens_ptr,
    out_tokens_stride,
    scratch_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    block_nb: tl.constexpr,
):
    """Combine per-block stats into the row softmax denominator + sampled token.

    Uses the standard online-softmax block combination for ``Z`` and selects
    the Gumbel-max token (or plain argmax for greedy rows). Ties break to the
    lowest vocab id, matching the greedy Triton kernel's convention.
    """
    batch_pid = tl.program_id(0)
    offs = tl.arange(0, block_nb)
    mask = offs < num_blocks
    base = batch_pid * scratch_stride + offs

    bmax = tl.load(block_max_ptr + base, mask=mask, other=-float("inf")).to(tl.float32)
    bsum = tl.load(block_sumexp_ptr + base, mask=mask, other=0.0).to(tl.float32)
    row_max = tl.max(bmax, axis=0)
    row_z = tl.sum(tl.where(mask, bsum * tl.exp(bmax - row_max), 0.0), axis=0)
    tl.store(row_max_ptr + batch_pid, row_max)
    tl.store(row_invz_ptr + batch_pid, 1.0 / row_z)

    int_max = 2147483647
    greedy = tl.load(is_greedy_ptr + batch_pid)

    bgval = tl.load(block_gval_ptr + base, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    bgid = tl.load(block_gid_ptr + base, mask=mask, other=int_max).to(tl.int64)
    gmax = tl.max(bgval, axis=0)
    gumbel_token = tl.min(tl.where((bgval == gmax) & mask, bgid, int_max), axis=0)

    bmaxid = tl.load(block_maxid_ptr + base, mask=mask, other=int_max).to(tl.int64)
    greedy_token = tl.min(tl.where((bmax == row_max) & mask, bmaxid, int_max), axis=0)

    token = tl.where(greedy != 0, greedy_token, gumbel_token)
    tl.store(out_tokens_ptr + batch_pid * out_tokens_stride, token)


@triton.jit
def _dspark_markov_probs_normalize_kernel(
    logits_ptr,
    inv_temp_ptr,
    row_max_ptr,
    row_invz_ptr,
    out_probs_ptr,
    vocab_size: tl.constexpr,
    logits_row_stride: tl.constexpr,
    probs_row_stride: tl.constexpr,
    block_v: tl.constexpr,
):
    """Write the normalized softmax probabilities for the draft-probs output."""
    batch_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    offs_v = block_pid * block_v + tl.arange(0, block_v)
    v_mask = offs_v < vocab_size

    inv_t = tl.load(inv_temp_ptr + batch_pid).to(tl.float32)
    z = (
        tl.load(
            logits_ptr + batch_pid * logits_row_stride + offs_v,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)
        * inv_t
    )
    row_max = tl.load(row_max_ptr + batch_pid).to(tl.float32)
    inv_z = tl.load(row_invz_ptr + batch_pid).to(tl.float32)
    probs = tl.exp(z - row_max) * inv_z
    tl.store(out_probs_ptr + batch_pid * probs_row_stride + offs_v, probs, mask=v_mask)


def dspark_markov_probs_sample(
    step_logits: torch.Tensor,
    inv_temp: torch.Tensor,
    is_greedy: torch.Tensor,
    out_tokens: torch.Tensor,
    out_probs: torch.Tensor,
    scratch: dict[str, torch.Tensor],
    seed: int,
    *,
    block_v: int = 1024,
) -> None:
    """Fused probabilistic DSpark Markov sampler for one block step.

    Given per-request ``step_logits`` (base LM-head logits already summed with
    the Markov bias, *pre*-temperature), writes the sampled ``out_tokens`` and
    the full softmax ``out_probs`` (draft probabilities the rejection sampler
    consumes). Temperature scaling, softmax, Gumbel sampling, and the
    draft-probs write are fused into three launches instead of the ~10 eager
    ops of the reference sampler. The Markov ``w2`` GEMM is intentionally left
    to the caller so its full-vocab weight is read once, not per pass.
    """
    if step_logits.dim() != 2:
        raise ValueError(f"step_logits must be [batch, vocab], got {step_logits.shape}")
    batch_size, vocab_size = step_logits.shape
    if out_probs.shape != step_logits.shape:
        raise ValueError(
            f"out_probs shape {out_probs.shape} must match step_logits "
            f"{step_logits.shape}"
        )
    if out_probs.stride(-1) != 1:
        raise ValueError("out_probs must have a contiguous vocab dimension")
    if inv_temp.shape[0] < batch_size or is_greedy.shape[0] < batch_size:
        raise ValueError("inv_temp and is_greedy must cover batch_size")
    if out_tokens.shape[0] < batch_size:
        raise ValueError("out_tokens must cover batch_size")

    num_blocks = triton.cdiv(vocab_size, block_v)
    for name in ("block_max", "block_sumexp", "block_gval"):
        buf = scratch.get(name)
        if buf is None or buf.shape[0] < batch_size or buf.shape[1] < num_blocks:
            raise ValueError(
                f"scratch['{name}'] too small: {None if buf is None else buf.shape}"
            )
    for name in ("block_maxid", "block_gid"):
        buf = scratch.get(name)
        if buf is None or buf.shape[0] < batch_size or buf.shape[1] < num_blocks:
            raise ValueError(
                f"scratch['{name}'] too small: {None if buf is None else buf.shape}"
            )
    for name in ("row_max", "row_invz"):
        buf = scratch.get(name)
        if buf is None or buf.shape[0] < batch_size:
            raise ValueError(
                f"scratch['{name}'] too small: {None if buf is None else buf.shape}"
            )

    block_max = scratch["block_max"]
    block_sumexp = scratch["block_sumexp"]
    block_maxid = scratch["block_maxid"]
    block_gval = scratch["block_gval"]
    block_gid = scratch["block_gid"]
    row_max = scratch["row_max"]
    row_invz = scratch["row_invz"]

    grid = (batch_size, num_blocks)
    _dspark_markov_probs_blocks_kernel[grid](
        step_logits,
        inv_temp,
        block_max,
        block_sumexp,
        block_maxid,
        block_gval,
        block_gid,
        int(seed) & 0x7FFFFFFF,
        vocab_size=vocab_size,
        logits_row_stride=step_logits.stride(0),
        scratch_stride=block_max.stride(0),
        block_v=block_v,
        num_warps=8,
    )
    _dspark_markov_probs_reduce_kernel[(batch_size,)](
        block_max,
        block_sumexp,
        block_maxid,
        block_gval,
        block_gid,
        is_greedy,
        row_max,
        row_invz,
        out_tokens,
        out_tokens.stride(0),
        scratch_stride=block_max.stride(0),
        num_blocks=num_blocks,
        block_nb=triton.next_power_of_2(num_blocks),
        num_warps=8,
    )
    _dspark_markov_probs_normalize_kernel[grid](
        step_logits,
        inv_temp,
        row_max,
        row_invz,
        out_probs,
        vocab_size=vocab_size,
        logits_row_stride=step_logits.stride(0),
        probs_row_stride=out_probs.stride(0),
        block_v=block_v,
        num_warps=8,
    )
