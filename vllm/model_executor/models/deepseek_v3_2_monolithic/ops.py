# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    y_ptr,
    y_stride,
    eps,
    W_OFFSET: tl.constexpr,
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
        y = (x * rrms) * (w + W_OFFSET)
        tl.store(y_row_ptr + offset, y, mask=mask)


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    w_offset: float = 0.0,
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
        w_offset,
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


# @triton.jit
# def _qk_rope_kernel(
#     q_ptr,
#     q_stride0,
#     q_stride1,
#     k_ptr,
#     k_stride0,
#     k_stride1,
#     pos_ptr,
#     cos_sin_ptr,
#     cos_sin_stride,
#     NOPE_DIM: tl.constexpr,
#     ROT_DIM: tl.constexpr,
#     INTERLEAVED: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     tok_idx = tl.program_id(1)
#     pos = tl.load(pos_ptr + tok_idx)

#     block = tl.arange(0, ROT_DIM // 2)
#     cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + block)
#     cos = cos.to(tl.float32)
#     sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + block + ROT_DIM // 2)
#     sin = sin.to(tl.float32)

#     if INTERLEAVED:

#     if tl.program_id(0) == 0
#         # Handle Q
#         pass
#     elif tl.program_id(0) == 1:
#         # Handle K
#         pass


#     """Apply rotary position embedding to one head of one token in-place.

#     Grid: (num_tokens, num_heads).
#     Each program handles one (token, head) pair, rotating ``2 * HALF_ROPE``
#     elements starting at offset ``nope_dim`` within that head.

#     Supports both NeoX (split-half) and GPT-J (interleaved) styles.
#     cos_sin_cache layout: ``[max_position, rotary_dim]`` where the first
#     ``HALF_ROPE`` columns are cos and the next ``HALF_ROPE`` are sin.
#     """
#     tok_idx = tl.program_id(0)
#     head_idx = tl.program_id(1)

#     # Look up cos / sin for this token's position
#     pos = tl.load(pos_ptr + tok_idx)
#     cs_row = cos_sin_ptr + pos * stride_cs
#     offs = tl.arange(0, HALF_ROPE)
#     cos = tl.load(cs_row + offs).to(tl.float32)
#     sin = tl.load(cs_row + HALF_ROPE + offs).to(tl.float32)

#     # Base pointer for the rope portion of x[tok_idx, head_idx]
#     x_base = x_ptr + tok_idx * stride_tok + head_idx * stride_head + nope_dim

#     if IS_NEOX:
#         # NeoX style: first half paired with second half
#         x1 = tl.load(x_base + offs).to(tl.float32)
#         x2 = tl.load(x_base + HALF_ROPE + offs).to(tl.float32)
#         tl.store(x_base + offs, x1 * cos - x2 * sin)
#         tl.store(x_base + HALF_ROPE + offs, x2 * cos + x1 * sin)
#     else:
#         # GPT-J style: interleaved even/odd pairs
#         even = offs * 2
#         odd = even + 1
#         x1 = tl.load(x_base + even).to(tl.float32)
#         x2 = tl.load(x_base + odd).to(tl.float32)
#         tl.store(x_base + even, x1 * cos - x2 * sin)
#         tl.store(x_base + odd, x2 * cos + x1 * sin)


# def deepseek_rope(
#     positions: torch.Tensor,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     cos_sin_cache: torch.Tensor,
#     qk_nope_head_dim: int,
#     is_neox_style: bool = False,
# ) -> None:
#     num_tokens = positions.shape[0]
#     num_heads = q.shape[1]
#     rope_dim = cos_sin_cache.shape[-1]
#     half_rope = rope_dim // 2

#     # Shared kernel config – rope_dim is small (typically 64), one warp is
#     # enough per program instance.
#     num_waprps = 1 if half_rope <= 32 else (2 if half_rope <= 64 else 4)

#     # Q: grid over (tokens, heads), rope starts at nope_dim within each head
#     _rope_kernel[(num_tokens, num_heads)](
#         q,
#         cos_sin_cache,
#         positions,
#         q.stride(0),
#         q.stride(1),
#         cos_sin_cache.stride(0),
#         qk_nope_head_dim,
#         half_rope,
#         is_neox_style,
#         num_warps=num_warps,
#     )

#     # K: grid over (tokens, 1), rope starts at offset 0
#     _rope_kernel[(num_tokens, 1)](
#         k,
#         cos_sin_cache,
#         positions,
#         k.stride(0),
#         k.stride(1),
#         cos_sin_cache.stride(0),
#         0,  # nope_dim = 0 for K
#         half_rope,
#         is_neox_style,
#         num_warps=num_warps,
#     )
