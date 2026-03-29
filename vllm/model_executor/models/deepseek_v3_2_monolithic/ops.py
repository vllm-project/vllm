# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl


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

    x = tl.load(x_row_ptr + block, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)

    w = tl.load(w_ptr + block, mask=mask).to(tl.float32)
    y = (x * rrms) * w
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

    x = tl.load(x_row_ptr + block, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / HIDDEN_SIZE
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + block, mask=mask).to(tl.float32)
    b = tl.load(bias_ptr + block, mask=mask).to(tl.float32)
    y = (x - mean) * rstd * w + b
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
    NOPE_DIM: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    head_offset = tl.arange(0, NUM_HEADS)
    dim_offset = tl.arange(0, HALF_ROT_DIM)
    base_ptr = base_ptr + head_offset[:, None] * head_stride + NOPE_DIM
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
def _qk_rope_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    NUM_Q_HEADS: tl.constexpr,
    Q_NOPE_DIM: tl.constexpr,
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

    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)

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
            Q_NOPE_DIM,
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
    q_nope_dim: int,
    interleaved: bool,
) -> None:
    assert q.ndim == 3
    assert k.ndim == 3
    assert q.shape[0] == k.shape[0]
    assert cos_sin_cache.ndim == 2
    assert positions.ndim == 1
    assert q_nope_dim < q.shape[-1]
    num_tokens, num_q_heads, _ = q.shape
    num_tokens, num_k_heads, _ = k.shape
    rot_dim = cos_sin_cache.shape[-1]
    _qk_rope_kernel[(2, num_tokens)](
        q,
        q.stride(0),
        q.stride(1),
        num_q_heads,
        q_nope_dim,
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


# if __name__ == "__main__":
#     N = 100
#     NUM_Q_HEADS = 128
#     NUM_K_HEADS = 1
#     ROPE_DIM = 64
#     Q_NOPE_DIM = 128
#     MAX_POS = 10000

#     q = torch.randn(N, NUM_Q_HEADS, Q_NOPE_DIM + ROPE_DIM, device="cuda", dtype=torch.bfloat16)
#     k = torch.randn(N, NUM_K_HEADS, ROPE_DIM, device="cuda", dtype=torch.bfloat16)
#     positions = torch.randint(0, MAX_POS, (N,), device="cuda", dtype=torch.int64)
#     cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM * 2, device="cuda", dtype=torch.bfloat16)
#     q, k = qk_rope(positions, q, k, cos_sin_cache, Q_NOPE_DIM, False)
#     print(q.shape)
#     print(k.shape)
