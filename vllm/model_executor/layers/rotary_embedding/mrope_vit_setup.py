# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d


# num_tokens/num_grids vary per request (incl. the ==1 class) and grids_ptr
# is an offset slice of the packed H2D buffer, so its 16B alignment varies
# with grid count; without these the kernel recompiles mid-serving.
@triton.jit(
    do_not_specialize=["num_tokens", "num_grids"],
    do_not_specialize_on_alignment=["grids_ptr"],
)
def _vit_mrope_setup_kernel(
    inv_freq_t_ptr,
    inv_freq_h_ptr,
    inv_freq_w_ptr,
    cos_out_ptr,
    sin_out_ptr,
    cu_seqlens_ptr,  # (G + 1,) int32 token boundaries per grid
    grids_ptr,  # (G, 3) int32: t, h, w per grid
    num_tokens,
    num_grids,
    half_t: tl.constexpr,
    half_h: tl.constexpr,
    half_w: tl.constexpr,
    spatial_merge: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Build triton_mrope's (3, N, half_rot) cos/sin t/h/w planes in one
    launch: compute each token's (t, h, w) position arithmetically and the
    rotation in fp32 from the per-axis inv_freq. Only each axis's mrope
    section columns are written; the rest is left uninitialized (the
    consumer's per-axis masked loads read only the section columns)."""
    pid = tl.program_id(0)
    tok = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    tok_mask = tok < num_tokens

    # Grid (segment) index per token; cu_seqlens is nondecreasing.
    seg = tl.zeros([BLOCK_N], dtype=tl.int32)
    for g in range(num_grids):
        end = tl.load(cu_seqlens_ptr + g + 1)
        seg += (tok >= end).to(tl.int32)

    seg_start = tl.load(cu_seqlens_ptr + seg, mask=tok_mask, other=0)
    grid_h = tl.load(grids_ptr + seg * 3 + 1, mask=tok_mask, other=1)
    grid_w = tl.load(grids_ptr + seg * 3 + 2, mask=tok_mask, other=1)

    local = tok - seg_start
    rem = local % (grid_h * grid_w)
    tpos = local // (grid_h * grid_w)
    # Within a frame tokens are ordered by spatial-merge block (bh, bw, mh,
    # mw), matching the reference ViT position-id permute.
    bh = rem // (spatial_merge * grid_w)
    r = rem % (spatial_merge * grid_w)
    bw = r // (spatial_merge * spatial_merge)
    r2 = r % (spatial_merge * spatial_merge)
    hpos = bh * spatial_merge + r2 // spatial_merge
    wpos = bw * spatial_merge + r2 % spatial_merge

    d = tl.arange(0, BLOCK_D)
    half_rot = half_t + half_h + half_w
    row = (tok * half_rot)[:, None] + d[None, :]
    plane = num_tokens * half_rot

    # Per-axis compute+store to cap live registers; the fp32 trig sequence
    # wants few elements per thread (hence small BLOCK_N, many warps).
    in_t = d < half_t
    inv_t = tl.load(inv_freq_t_ptr + d, mask=in_t, other=0.0)
    freq = tpos.to(tl.float32)[:, None] * inv_t[None, :]
    mask = tok_mask[:, None] & in_t[None, :]
    tl.store(cos_out_ptr + row, tl.cos(freq), mask=mask)
    tl.store(sin_out_ptr + row, tl.sin(freq), mask=mask)

    in_h = (d >= half_t) & (d < half_t + half_h)
    inv_h = tl.load(inv_freq_h_ptr + d - half_t, mask=in_h, other=0.0)
    freq = hpos.to(tl.float32)[:, None] * inv_h[None, :]
    mask = tok_mask[:, None] & in_h[None, :]
    tl.store(cos_out_ptr + plane + row, tl.cos(freq), mask=mask)
    tl.store(sin_out_ptr + plane + row, tl.sin(freq), mask=mask)

    in_w = (d >= half_t + half_h) & (d < half_rot)
    inv_w = tl.load(inv_freq_w_ptr + d - half_t - half_h, mask=in_w, other=0.0)
    freq = wpos.to(tl.float32)[:, None] * inv_w[None, :]
    mask = tok_mask[:, None] & in_w[None, :]
    tl.store(cos_out_ptr + 2 * plane + row, tl.cos(freq), mask=mask)
    tl.store(sin_out_ptr + 2 * plane + row, tl.sin(freq), mask=mask)


def vit_mrope_setup(
    inv_freq_t: torch.Tensor,
    inv_freq_h: torch.Tensor,
    inv_freq_w: torch.Tensor,
    grid_thw: list[list[int]],
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(3, total_N, half_rot_dim) fp32 cos/sin t/h/w planes for triton_mrope.

    One fused kernel: positions are computed arithmetically from the grids
    and the rotation comes from the per-axis inv_freq, so there is no
    cos/sin cache to size or grow.

    Args:
        inv_freq_t: t-axis inverse frequencies, (t_dim // 2,).
        inv_freq_h: h-axis inverse frequencies, (h_dim // 2,).
        inv_freq_w: w-axis inverse frequencies, (w_dim // 2,).
        grid_thw: per-grid [t, h, w] patch counts (host metadata).
        spatial_merge_size: spatial merge factor of the ViT.

    """
    device = inv_freq_t.device
    cu = [0]
    for t, h, w in grid_thw:
        cu.append(cu[-1] + t * h * w)
    n = cu[-1]
    g = len(grid_thw)
    # grid_thw is per-request host metadata, so one H2D is unavoidable;
    # pack cu_seqlens and grids into a single pinned async copy.
    buf = async_tensor_h2d(
        cu + [d for grid in grid_thw for d in grid], device=device, dtype=torch.int32
    )
    cu_seqlens = buf[: g + 1]
    grids = buf[g + 1 :].view(g, 3)

    half_t = inv_freq_t.numel()
    half_h = inv_freq_h.numel()
    half_w = inv_freq_w.numel()
    half_rot = half_t + half_h + half_w
    cos = torch.empty(3, n, half_rot, device=device, dtype=torch.float32)
    sin = torch.empty(3, n, half_rot, device=device, dtype=torch.float32)
    _vit_mrope_setup_kernel[(triton.cdiv(n, 32),)](
        inv_freq_t,
        inv_freq_h,
        inv_freq_w,
        cos,
        sin,
        cu_seqlens,
        grids,
        n,
        g,
        half_t=half_t,
        half_h=half_h,
        half_w=half_w,
        spatial_merge=spatial_merge_size,
        BLOCK_N=32,
        BLOCK_D=triton.next_power_of_2(half_rot),
        num_warps=8,
    )
    return cos, sin
