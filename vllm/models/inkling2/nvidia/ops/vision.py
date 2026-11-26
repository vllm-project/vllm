# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Inkling2 vision encoder.

Rows are tokens of the packed images (a CLS row, then the native patches in
row-major order). Row metadata is int32 `[T, 4]`: native `(y, x)` and the
native grid `(height, width)`, with `y = x = -1` on CLS rows. Outputs are
rounded to bf16 at the same points as the eager PyTorch ops they replace.
"""

import functools

import torch

from vllm.triton_utils import tl, tldevice, triton


@functools.cache
def _rope_inv_freq(head_dim: int, device: torch.device) -> torch.Tensor:
    # Rounded from float64 to float32, matching the reference constants.
    values = [10000.0 ** (-dim / head_dim) for dim in range(0, head_dim, 4)]
    return torch.tensor(values, dtype=torch.float32, device=device)


@triton.jit
def _bf16(x):
    # Round to bf16 and back; inline asm keeps Triton from folding the casts.
    rounded = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;",
        "=h,f",
        [x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )
    return rounded.to(tl.float32)


@triton.jit
def _layer_norm(x, w_ptr, b_ptr, cols, mask, DIM: tl.constexpr, eps):
    # x is zero outside mask; fp32 statistics as in torch's LayerNorm.
    mean = tl.sum(x) / DIM
    diff = tl.where(mask, x - mean, 0.0)
    rstd = tl.math.rsqrt(tl.sum(diff * diff) / DIM + eps)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
    return tl.fma(w, diff * rstd, b)


@triton.jit
def _gather_patches_kernel(
    pixels_ptr,
    offsets_ptr,
    out_ptr,
    mean0,
    mean1,
    mean2,
    clip_std0,
    clip_std1,
    clip_std2,
    std0,
    std1,
    std2,
    SIDE: tl.constexpr,
    PATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Raw RGB -> CLIP (rounded to bf16) -> PE [-1, 1] (rounded to bf16), each
    # op in fp32 without FMA; features ordered (py, px, channel).
    row = tl.program_id(0).to(tl.int64)
    feature = tl.arange(0, BLOCK)
    mask = feature < PATCH * PATCH * 3
    channel = feature % 3
    src = (
        tl.load(offsets_ptr + row).to(tl.int64)
        + (feature // (PATCH * 3)) * (SIDE * 3)
        + feature % (PATCH * 3)
    )
    mean = tl.where(channel == 0, mean0, tl.where(channel == 1, mean1, mean2))
    clip_std = tl.where(
        channel == 0, clip_std0, tl.where(channel == 1, clip_std1, clip_std2)
    )
    std = tl.where(channel == 0, std0, tl.where(channel == 1, std1, std2))
    value = tl.load(pixels_ptr + src, mask=mask).to(tl.float32)
    value = _bf16(tl.div_rn(tl.div_rn(value, 255.0) - mean, clip_std))
    value = 2.0 * (value * std + mean) - 1.0
    tl.store(out_ptr + row * (PATCH * PATCH * 3) + feature, value, mask=mask)


def gather_patches(
    pixels: torch.Tensor,
    offsets: torch.Tensor,
    patch_size: int,
    mean: tuple[float, float, float],
    clip_std: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """Native patches of raw `[N, side, side, 3]` pixels as bf16 rows of
    `(py, px, c)` values in [-1, 1].

    Each value is `2 * (bf16((v / 255 - mean) / clip_std) * std + mean) - 1`.
    `offsets[i]` is the element offset of row i's top-left pixel.
    """
    _, side, _, channels = pixels.shape
    assert channels == 3 and pixels.is_contiguous()
    features = patch_size * patch_size * 3
    out = pixels.new_empty((offsets.numel(), features))
    _gather_patches_kernel[(offsets.numel(),)](
        pixels,
        offsets,
        out,
        *mean,
        *clip_std,
        *std,
        SIDE=side,
        PATCH=patch_size,
        BLOCK=triton.next_power_of_2(features),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _source_coords(index, target, SOURCE: tl.constexpr):
    # align_corners=False bilinear coordinates, clamped to the source grid.
    coord = tl.div_rn(float(SOURCE) * (index + 0.5), target) - 0.5
    coord = tl.maximum(0.0, tl.minimum(coord, SOURCE - 1.0))
    src0 = tl.floor(coord).to(tl.int64)
    src1 = tl.minimum(src0 + 1, SOURCE - 1)
    w1 = coord - src0.to(tl.float32)
    return src0, src1, 1.0 - w1, w1


@triton.jit
def _lerp(a, b, wa, wb):
    return tl.where(wa >= wb, tl.fma(a, wa, b * wb), tl.fma(b, wb, a * wa))


@triton.jit(do_not_specialize=["eps"])
def _embed_kernel(
    patch_ptr,
    patch_rows_ptr,
    grid_ptr,
    meta_ptr,
    class_ptr,
    cls_pos_ptr,
    pre_w_ptr,
    pre_b_ptr,
    ln_w_ptr,
    ln_b_ptr,
    hidden_ptr,
    normed_ptr,
    eps,
    SOURCE_HEIGHT: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    # Patch embedding + resized position embedding (CLS rows: class + CLS
    # position embedding), then ln_pre (the residual) and layer 0's ln_1.
    row = tl.program_id(0).to(tl.int64)
    dim = tl.arange(0, BLOCK_DIM)
    mask = dim < DIM
    meta = meta_ptr + row * 4
    y = tl.load(meta)
    if y < 0:
        hidden = tl.load(class_ptr + dim, mask=mask).to(tl.float32)
        hidden += tl.load(cls_pos_ptr + dim, mask=mask).to(tl.float32)
    else:
        x = tl.load(meta + 1).to(tl.float32)
        height = tl.load(meta + 2).to(tl.float32)
        width = tl.load(meta + 3).to(tl.float32)
        y0, y1, wy0, wy1 = _source_coords(y.to(tl.float32), height, SOURCE_HEIGHT)
        x0, x1, wx0, wx1 = _source_coords(x, width, SOURCE_WIDTH)
        v00 = tl.load(grid_ptr + (y0 * SOURCE_WIDTH + x0) * DIM + dim, mask=mask)
        v01 = tl.load(grid_ptr + (y0 * SOURCE_WIDTH + x1) * DIM + dim, mask=mask)
        v10 = tl.load(grid_ptr + (y1 * SOURCE_WIDTH + x0) * DIM + dim, mask=mask)
        v11 = tl.load(grid_ptr + (y1 * SOURCE_WIDTH + x1) * DIM + dim, mask=mask)
        top = _lerp(v00.to(tl.float32), v01.to(tl.float32), wx0, wx1)
        bottom = _lerp(v10.to(tl.float32), v11.to(tl.float32), wx0, wx1)
        patch_row = tl.load(patch_rows_ptr + row).to(tl.int64)
        hidden = tl.load(patch_ptr + patch_row * DIM + dim, mask=mask)
        hidden = hidden.to(tl.float32) + _bf16(_lerp(top, bottom, wy0, wy1))
    hidden = tl.where(mask, _bf16(hidden), 0.0)
    hidden = _layer_norm(hidden, pre_w_ptr, pre_b_ptr, dim, mask, DIM, eps)
    hidden = tl.where(mask, _bf16(hidden), 0.0)
    tl.store(hidden_ptr + row * DIM + dim, hidden, mask=mask)
    normed = _layer_norm(hidden, ln_w_ptr, ln_b_ptr, dim, mask, DIM, eps)
    tl.store(normed_ptr + row * DIM + dim, normed, mask=mask)


def embed_tokens(
    patch_embeds: torch.Tensor,
    patch_rows: torch.Tensor,
    position_grid: torch.Tensor,
    meta: torch.Tensor,
    class_embedding: torch.Tensor,
    cls_position_embedding: torch.Tensor,
    ln_pre: torch.nn.LayerNorm,
    ln_1: torch.nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(hidden, ln_1(hidden))` with `hidden = ln_pre(embeddings)`.

    Token row i takes patch embedding `patch_embeds[patch_rows[i]]`; CLS rows
    (`patch_rows` unused) take the class embedding.
    """
    num_rows, dim = meta.shape[0], patch_embeds.shape[1]
    assert ln_pre.eps == ln_1.eps
    hidden = patch_embeds.new_empty((num_rows, dim))
    normed = torch.empty_like(hidden)
    _embed_kernel[(num_rows,)](
        patch_embeds,
        patch_rows,
        position_grid,
        meta,
        class_embedding,
        cls_position_embedding,
        ln_pre.weight,
        ln_pre.bias,
        ln_1.weight,
        ln_1.bias,
        hidden,
        normed,
        ln_pre.eps,
        SOURCE_HEIGHT=position_grid.shape[0],
        SOURCE_WIDTH=position_grid.shape[1],
        DIM=dim,
        BLOCK_DIM=triton.next_power_of_2(dim),
        num_warps=4,
    )
    return hidden, normed


@triton.jit
def _rotate_pair(even, odd, cos, sin):
    return tl.fma(even, cos, -odd * sin), tl.fma(odd, cos, even * sin)


@triton.jit
def _rotary_qk_kernel(
    qkv_ptr,
    meta_ptr,
    inv_freq_ptr,
    stride_token,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)[:, None]
    pair = tl.arange(0, BLOCK_PAIRS)[None, :]
    pair_mask = pair < HEAD_DIM // 2
    mask = (head < NUM_HEADS) & pair_mask

    # The first half of the pairs rotate by x + 1, the rest by y + 1 (CLS: 0).
    inv_freq = tl.load(inv_freq_ptr + pair % (HEAD_DIM // 4), mask=pair_mask)
    axis = tl.where(pair < HEAD_DIM // 4, 1, 0)
    position = tl.load(meta_ptr + 4 * token + axis, mask=pair_mask, other=0) + 1
    angle = position.to(tl.float32) * inv_freq
    cos = tldevice.cos(angle)
    sin = tldevice.sin(angle)

    q_ptr = qkv_ptr + token * stride_token + head * HEAD_DIM + 2 * pair
    k_ptr = q_ptr + NUM_HEADS * HEAD_DIM
    q_even = tl.load(q_ptr, mask=mask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_ptr + 1, mask=mask, other=0.0).to(tl.float32)
    k_even = tl.load(k_ptr, mask=mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_ptr + 1, mask=mask, other=0.0).to(tl.float32)
    q_even, q_odd = _rotate_pair(q_even, q_odd, cos, sin)
    k_even, k_odd = _rotate_pair(k_even, k_odd, cos, sin)
    tl.store(q_ptr, q_even, mask=mask)
    tl.store(q_ptr + 1, q_odd, mask=mask)
    tl.store(k_ptr, k_even, mask=mask)
    tl.store(k_ptr + 1, k_odd, mask=mask)


def apply_rotary_qk_(qkv: torch.Tensor, meta: torch.Tensor, num_heads: int) -> None:
    """In-place interleaved 2D RoPE on the q and k thirds of `qkv [T, 3D]`."""
    num_tokens, width = qkv.shape
    head_dim = width // (3 * num_heads)
    block_heads = min(4, triton.next_power_of_2(num_heads))
    grid = (num_tokens, triton.cdiv(num_heads, block_heads))
    _rotary_qk_kernel[grid](
        qkv,
        meta,
        _rope_inv_freq(head_dim, qkv.device),
        qkv.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEADS=block_heads,
        BLOCK_PAIRS=triton.next_power_of_2(head_dim // 2),
        num_warps=4,
    )


@triton.jit(do_not_specialize=["eps"])
def _scaled_residual_norm_kernel(
    hidden_ptr,
    update_ptr,
    gamma_ptr,
    ln_w_ptr,
    ln_b_ptr,
    normed_ptr,
    eps,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dim = tl.arange(0, BLOCK_DIM)
    mask = dim < DIM
    update = tl.load(update_ptr + row * DIM + dim, mask=mask).to(tl.float32)
    gamma = tl.load(gamma_ptr + dim, mask=mask).to(tl.float32)
    hidden = tl.load(hidden_ptr + row * DIM + dim, mask=mask).to(tl.float32)
    hidden = tl.where(mask, _bf16(hidden + _bf16(update * gamma)), 0.0)
    tl.store(hidden_ptr + row * DIM + dim, hidden, mask=mask)
    if HAS_NORM:
        normed = _layer_norm(hidden, ln_w_ptr, ln_b_ptr, dim, mask, DIM, eps)
        tl.store(normed_ptr + row * DIM + dim, normed, mask=mask)


def scaled_residual_norm_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.nn.LayerNorm | None,
) -> torch.Tensor | None:
    """`hidden += update * gamma` in place; returns `norm(hidden)`."""
    num_rows, dim = hidden.shape
    assert hidden.is_contiguous() and update.is_contiguous()
    normed = None if norm is None else torch.empty_like(hidden)
    _scaled_residual_norm_kernel[(num_rows,)](
        hidden,
        update,
        gamma,
        None if norm is None else norm.weight,
        None if norm is None else norm.bias,
        normed,
        0.0 if norm is None else norm.eps,
        DIM=dim,
        BLOCK_DIM=triton.next_power_of_2(dim),
        HAS_NORM=norm is not None,
        num_warps=4,
    )
    return normed
