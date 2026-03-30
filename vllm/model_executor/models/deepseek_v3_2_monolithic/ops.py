# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _rms_norm_small_dim_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    y_ptr,
    y_stride,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * x_stride
    y_row_ptr = y_ptr + row_idx * y_stride

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < HIDDEN_SIZE
    x = tl.load(x_row_ptr + block, mask=mask, other=0.0)
    w = tl.load(w_ptr + block, mask=mask)
    y = _rms_norm(x, w, eps, HIDDEN_SIZE)
    tl.store(y_row_ptr + block, y, mask=mask)


def rms_norm_small(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert x.ndim == 2
    assert w.ndim == 1
    num_tokens, hidden_size = x.shape
    y = torch.empty_like(x)

    _rms_norm_small_dim_kernel[(num_tokens,)](
        x,
        x.stride(0),
        w,
        y,
        y.stride(0),
        eps,
        hidden_size,
        BLOCK_SIZE=triton.next_power_of_2(hidden_size),
    )
    return y


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    y_ptr,
    y_stride,
    eps,
    NUM_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * x_stride
    y_row_ptr = y_ptr + row_idx * y_stride

    sq_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, NUM_COLS, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < NUM_COLS
        x = tl.load(x_row_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        sq_sum += x * x

    mean_sq = tl.sum(sq_sum, axis=0) / NUM_COLS
    rrms = tl.rsqrt(mean_sq + eps)

    for i in range(0, NUM_COLS, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < NUM_COLS
        x = tl.load(x_row_ptr + offset, mask=mask).to(tl.float32)
        w = tl.load(w_ptr + offset, mask=mask).to(tl.float32)
        y = (x * rrms) * w
        tl.store(y_row_ptr + offset, y, mask=mask)


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert x.ndim == 2
    assert w.ndim == 1
    num_tokens, hidden_size = x.shape

    y = torch.empty_like(x)
    BLOCK_SIZE = 1024  # TODO: Tune this
    _rms_norm_kernel[(num_tokens,)](
        x,
        x.stride(0),
        w,
        y,
        y.stride(0),
        eps,
        hidden_size,
        BLOCK_SIZE,
    )
    return y


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    x_stride,
    residual_ptr,
    residual_stride,
    w_ptr,
    y_ptr,
    y_stride,
    residual_new_ptr,
    residual_new_stride,
    eps,
    W_OFFSET: tl.constexpr,
    NUM_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * x_stride
    r_row_ptr = residual_ptr + row_idx * residual_stride
    y_row_ptr = y_ptr + row_idx * y_stride
    r_new_row_ptr = residual_new_ptr + row_idx * residual_new_stride

    sq_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, NUM_COLS, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < NUM_COLS
        x = tl.load(x_row_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_row_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        sq_sum += x * x

    mean_sq = tl.sum(sq_sum, axis=0) / NUM_COLS
    rrms = tl.rsqrt(mean_sq + eps)

    for i in range(0, NUM_COLS, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < NUM_COLS

        # Recompute x + r
        x = tl.load(x_row_ptr + offset, mask=mask).to(tl.float32)
        r = tl.load(r_row_ptr + offset, mask=mask).to(tl.float32)
        x = x + r

        w = tl.load(w_ptr + offset, mask=mask).to(tl.float32)
        y = (x * rrms) * (w + W_OFFSET)
        tl.store(y_row_ptr + offset, y, mask=mask)

        tl.store(r_new_row_ptr + offset, x, mask=mask)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    w_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2
    assert residual.shape == x.shape
    assert weight.ndim == 1
    num_tokens, hidden_size = x.shape

    y = torch.empty_like(x)
    residual_new = torch.empty_like(x)
    BLOCK_SIZE = 1024  # TODO: Tune this
    _fused_add_rms_norm_kernel[(num_tokens,)](
        x,
        x.stride(0),
        residual,
        residual.stride(0),
        weight,
        y,
        y.stride(0),
        residual_new,
        residual_new.stride(0),
        eps,
        w_offset,
        hidden_size,
        BLOCK_SIZE,
    )
    return y, residual_new


@triton.jit
def _layer_norm(x, w, b, eps, mask, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=0) / HIDDEN_SIZE
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE
    rstd = tl.rsqrt(var + eps)

    w = w.to(tl.float32)
    b = b.to(tl.float32)
    return (x - mean) * rstd * w + b


# Optimized for small hidden size
@triton.jit
def _layer_norm_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    bias_ptr,
    y_ptr,
    y_stride,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * x_stride
    y_row_ptr = y_ptr + row_idx * y_stride

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < HIDDEN_SIZE

    x = tl.load(x_row_ptr + block, mask=mask, other=0.0)
    w = tl.load(w_ptr + block, mask=mask)
    b = tl.load(bias_ptr + block, mask=mask)

    y = _layer_norm(x, w, b, eps, mask, HIDDEN_SIZE)
    tl.store(y_row_ptr + block, y, mask=mask)


def layer_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert x.ndim == 2
    assert w.ndim == 1
    assert bias.ndim == 1
    num_tokens, hidden_size = x.shape
    y = torch.empty_like(x)
    _layer_norm_kernel[(num_tokens,)](
        x,
        x.stride(0),
        w,
        bias,
        y,
        y.stride(0),
        eps,
        hidden_size,
        BLOCK_SIZE=triton.next_power_of_2(hidden_size),
    )
    return y


@triton.jit
def _rope_kernel(
    base_ptr,
    head_stride,
    cos,
    sin,
    NUM_HEADS: tl.constexpr,
    HALF_ROT_DIM: tl.constexpr,
    START_OFFSET: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    head_offset = tl.arange(0, NUM_HEADS)
    dim_offset = tl.arange(0, HALF_ROT_DIM)
    base_ptr = base_ptr + head_offset[:, None] * head_stride + START_OFFSET
    if INTERLEAVED:
        x1 = tl.load(base_ptr + dim_offset * 2).to(tl.float32)
        x2 = tl.load(base_ptr + dim_offset * 2 + 1).to(tl.float32)
        tl.store(base_ptr + dim_offset * 2, x1 * cos - x2 * sin)
        tl.store(base_ptr + dim_offset * 2 + 1, x2 * cos + x1 * sin)
    else:
        x1 = tl.load(base_ptr + dim_offset).to(tl.float32)
        x2 = tl.load(base_ptr + dim_offset + HALF_ROT_DIM).to(tl.float32)
        tl.store(base_ptr + dim_offset, x1 * cos - x2 * sin)
        tl.store(base_ptr + dim_offset + HALF_ROT_DIM, x2 * cos + x1 * sin)


@triton.jit
def _cos_sin_cache_kernel(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _qk_rope_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    NUM_Q_HEADS: tl.constexpr,
    Q_START_OFFSET: tl.constexpr,
    k_ptr,
    k_stride0,
    k_stride1,
    NUM_K_HEADS: tl.constexpr,
    pos_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    HALF_ROT_DIM: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    tok_idx = tl.program_id(1)
    pos = tl.load(pos_ptr + tok_idx)

    cos, sin = _cos_sin_cache_kernel(
        cos_sin_ptr,
        cos_sin_stride,
        pos,
        HALF_ROT_DIM,
    )

    if tl.program_id(0) == 0:
        # Handle Q [NUM_Q_HEADS, ROT_DIM]
        q_base_ptr = q_ptr + tok_idx * q_stride0
        _rope_kernel(
            q_base_ptr,
            q_stride1,
            cos,
            sin,
            NUM_Q_HEADS,
            HALF_ROT_DIM,
            Q_START_OFFSET,
            INTERLEAVED,
        )
    elif tl.program_id(0) == 1:
        # Handle K [NUM_K_HEADS, ROT_DIM]
        k_base_ptr = k_ptr + tok_idx * k_stride0
        _rope_kernel(
            k_base_ptr, k_stride1, cos, sin, NUM_K_HEADS, HALF_ROT_DIM, 0, INTERLEAVED
        )


def qk_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_start_offset: int,
    interleaved: bool,
) -> None:
    assert q.ndim == 3
    assert k.ndim == 3
    assert q.shape[0] == k.shape[0]
    assert cos_sin_cache.ndim == 2
    assert positions.ndim == 1
    assert q_start_offset < q.shape[-1]
    num_tokens, num_q_heads, _ = q.shape
    num_tokens, num_k_heads, _ = k.shape
    rot_dim = cos_sin_cache.shape[-1]
    _qk_rope_kernel[(2, num_tokens)](
        q,
        q.stride(0),
        q.stride(1),
        num_q_heads,
        q_start_offset,
        k,
        k.stride(0),
        k.stride(1),
        num_k_heads,
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        rot_dim // 2,
        interleaved,
    )


@triton.jit
def _fused_norm_rope_kernel(
    pos_ptr,
    # Q RMS norm
    q_c_ptr,
    q_c_stride,
    q_rms_norm_w_ptr,
    q_rms_eps,
    q_c_out_ptr,
    q_c_out_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    # KV RMS norm
    kv_ptr,
    kv_stride,
    kv_rms_norm_w_ptr,
    kv_rms_eps,
    kv_c_out_ptr,
    kv_c_out_stride,
    KV_DIM: tl.constexpr,
    # KV RoPE
    kpe_ptr,
    kpe_stride,
    kpe_rope_cos_sin_cache_ptr,
    kpe_rope_cos_sin_cache_stride,
    KPE_HALF_ROT_DIM: tl.constexpr,
    # Index K layer norm
    index_k_ptr,
    index_k_stride,
    index_k_layer_norm_w_ptr,
    index_k_layer_norm_bias_ptr,
    index_k_layer_norm_eps,
    INDEX_K_DIM: tl.constexpr,
    INDEX_K_BLOCK_SIZE: tl.constexpr,
    # Index K RoPE
    index_k_rope_cos_sin_cache_ptr,
    index_k_rope_cos_sin_cache_stride,
    INDEX_K_HALF_ROT_DIM: tl.constexpr,
):
    tok_idx = tl.program_id(1)
    if tl.program_id(0) == 0:
        # Q RMS norm
        q_block = tl.arange(0, Q_BLOCK_SIZE)
        q_mask = q_block < Q_DIM
        q_c = tl.load(q_c_ptr + tok_idx * q_c_stride + q_block, mask=q_mask, other=0.0)
        q_c_rms_w = tl.load(q_rms_norm_w_ptr + q_block, mask=q_mask)
        q_c = _rms_norm(q_c, q_c_rms_w, q_rms_eps, Q_DIM)
        tl.store(q_c_out_ptr + tok_idx * q_c_out_stride + q_block, q_c, mask=q_mask)
        return
    elif tl.program_id(0) == 1:
        # KV RMS Norm
        kv_block = tl.arange(0, KV_DIM)
        kv_c = tl.load(kv_ptr + tok_idx * kv_stride + kv_block)
        kv_c_rms_w = tl.load(kv_rms_norm_w_ptr + kv_block)
        kv_c = _rms_norm(kv_c, kv_c_rms_w, kv_rms_eps, KV_DIM)
        tl.store(kv_c_out_ptr + tok_idx * kv_c_out_stride + kv_block, kv_c)
        return
    elif tl.program_id(0) == 2:
        # KV RoPE
        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _cos_sin_cache_kernel(
            kpe_rope_cos_sin_cache_ptr,
            kpe_rope_cos_sin_cache_stride,
            pos,
            KPE_HALF_ROT_DIM,
        )
        _rope_kernel(
            kpe_ptr + tok_idx * kpe_stride,
            0,
            cos,
            sin,
            1,
            KPE_HALF_ROT_DIM,
            0,
            True,
        )
        return
    elif tl.program_id(0) == 3:
        # Index K layer norm + RoPE
        index_k_block = tl.arange(0, INDEX_K_BLOCK_SIZE)
        index_k_mask = index_k_block < INDEX_K_DIM

        # Layer Norm
        index_k = tl.load(
            index_k_ptr + tok_idx * index_k_stride + index_k_block,
            mask=index_k_mask,
            other=0.0,
        )
        index_k_w = tl.load(index_k_layer_norm_w_ptr + index_k_block, mask=index_k_mask)
        index_k_b = tl.load(
            index_k_layer_norm_bias_ptr + index_k_block, mask=index_k_mask
        )
        index_k = _layer_norm(
            index_k,
            index_k_w,
            index_k_b,
            index_k_layer_norm_eps,
            index_k_mask,
            INDEX_K_DIM,
        )

        # Save to the original buffer
        tl.store(
            index_k_ptr + tok_idx * index_k_stride + index_k_block,
            index_k,
            mask=index_k_mask,
        )

        # RoPE
        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _cos_sin_cache_kernel(
            index_k_rope_cos_sin_cache_ptr,
            index_k_rope_cos_sin_cache_stride,
            pos,
            INDEX_K_HALF_ROT_DIM,
        )
        _rope_kernel(
            index_k_ptr + tok_idx * index_k_stride,
            0,
            cos,
            sin,
            1,
            INDEX_K_HALF_ROT_DIM,
            0,
            False,
        )
        return


def fused_norm_rope(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    q_rms_norm_w: torch.Tensor,
    q_rms_eps: float,
    kv_c: torch.Tensor,
    kv_rms_norm_w: torch.Tensor,
    kv_rms_eps: float,
    k_pe: torch.Tensor,
    k_rope_cos_sin_cache: torch.Tensor,
    index_k: torch.Tensor,
    index_k_layer_norm_w: torch.Tensor,
    index_k_layer_norm_bias: torch.Tensor,
    index_k_layer_norm_eps: float,
    index_k_rope_cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert positions.ndim == 1
    assert q_c.ndim == 2
    assert kv_c.ndim == 2
    assert k_pe.ndim == 2
    assert index_k.ndim == 2

    num_tokens = positions.shape[0]
    q_dim = q_c.shape[-1]
    kv_dim = kv_c.shape[-1]
    index_k_dim = index_k.shape[-1]

    q_c_out = torch.empty_like(q_c)
    kv_c_out = torch.empty_like(kv_c)
    _fused_norm_rope_kernel[(4, num_tokens)](
        positions,
        # Q RMS norm
        q_c,
        q_c.stride(0),
        q_rms_norm_w,
        q_rms_eps,
        q_c_out,
        q_c_out.stride(0),
        q_dim,
        triton.next_power_of_2(q_dim),
        # KV RMS norm
        kv_c,
        kv_c.stride(0),
        kv_rms_norm_w,
        kv_rms_eps,
        kv_c_out,
        kv_c_out.stride(0),
        kv_dim,
        # KV RoPE
        k_pe,
        k_pe.stride(0),
        k_rope_cos_sin_cache,
        k_rope_cos_sin_cache.stride(0),
        k_rope_cos_sin_cache.shape[-1] // 2,
        # Index K layer norm + RoPE
        index_k,
        index_k.stride(0),
        index_k_layer_norm_w,
        index_k_layer_norm_bias,
        index_k_layer_norm_eps,
        index_k_dim,
        triton.next_power_of_2(index_k_dim),
        index_k_rope_cos_sin_cache,
        index_k_rope_cos_sin_cache.stride(0),
        index_k_rope_cos_sin_cache.shape[-1] // 2,
    )
    return q_c_out, kv_c_out


@triton.jit
def _fused_q_kernel(
    pos_ptr,
    # Q RoPE
    q_ptr,
    q_stride0,
    q_stride1,
    NUM_Q_HEADS: tl.constexpr,
    q_cos_sin_ptr,
    q_cos_sin_stride,
    Q_HALF_ROT_DIM: tl.constexpr,
    Q_START_OFFSET: tl.constexpr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    NUM_INDEX_Q_HEADS: tl.constexpr,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
):
    tok_idx = tl.program_id(1)
    pos = tl.load(pos_ptr + tok_idx)
    head_idx = tl.program_id(2)

    if tl.program_id(0) == 0:
        # Q RoPE
        cos, sin = _cos_sin_cache_kernel(
            q_cos_sin_ptr,
            q_cos_sin_stride,
            pos,
            Q_HALF_ROT_DIM,
        )

        # Each program processes two Q heads.
        # Since grid[2] == NUM_INDEX_Q_HEADS == 2 * TOTAL_NUM_Q_HEADS, this ensures
        # that all local Q heads are handled, even TP=1.
        if 2 * head_idx >= NUM_Q_HEADS:
            return
        _rope_kernel(
            q_ptr + tok_idx * q_stride0 + head_idx * 2 * q_stride1,
            q_stride1,
            cos,
            sin,
            2,
            Q_HALF_ROT_DIM,
            Q_START_OFFSET,
            True,
        )
        return
    elif tl.program_id(0) == 1:
        # Index Q RoPE
        cos, sin = _cos_sin_cache_kernel(
            index_q_cos_sin_ptr,
            index_q_cos_sin_stride,
            pos,
            INDEX_Q_HALF_ROT_DIM,
        )
        _rope_kernel(
            index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1,
            0,
            cos,
            sin,
            1,
            INDEX_Q_HALF_ROT_DIM,
            0,
            False,
        )


def fused_q(
    positions: torch.Tensor,
    q: torch.Tensor,
    q_cos_sin_cache: torch.Tensor,
    q_start_offset: int,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
) -> None:
    assert positions.ndim == 1
    assert q.ndim == 3
    assert q_cos_sin_cache.ndim == 2
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_q_heads = q.shape[1]
    num_index_q_heads = index_q.shape[1]

    _fused_q_kernel[(2, num_tokens, num_index_q_heads)](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        num_q_heads,
        q_cos_sin_cache,
        q_cos_sin_cache.stride(0),
        q_cos_sin_cache.shape[-1] // 2,
        q_start_offset,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        num_index_q_heads,
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
    )
