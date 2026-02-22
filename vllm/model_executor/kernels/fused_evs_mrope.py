# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused EVS dissimilarity + selective mRoPE position generation kernel.

Replaces the two-step pipeline in _postprocess_video_embeds_evs():
  1. compute_retention_mask()  -- cosine dissimilarity + argsort
  2. compute_mrope_for_media() -- full mRoPE grid, then mask

With a fused pipeline:
  A. Triton kernel: compute cosine dissimilarity between consecutive frames
     (one program per (frame, spatial_position) pair)
  B. torch.topk on dissimilarity scores to get retention mask
  C. Selective mRoPE: compute positions ONLY for retained token indices

This avoids computing the full (T*H'*W', 4) mRoPE grid and then
discarding most of it.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _fused_evs_dissimilarity_kernel(
    # Input: video embeddings, contiguous (T * HW, D), row-major
    embed_ptr,
    # Output: dissimilarity scores, contiguous (T * HW,)
    dissim_ptr,
    # Scalar params
    T: tl.constexpr,       # number of frames
    HW: tl.constexpr,      # tokens per frame (H' * W')
    D: tl.constexpr,       # hidden dimension
    BLOCK_D: tl.constexpr,  # tile size for hidden dim
):
    """
    Compute cosine dissimilarity between consecutive video frames.

    Grid: (T, HW) -- one program per (frame t, spatial position hw).

    For t == 0: output sentinel 255.0 (first frame always retained).
    For t > 0:  output 1 - cos_sim(frame[t, hw], frame[t-1, hw]).

    The cosine similarity is computed as:
        dot(a, b) / (||a|| * ||b||)
    where a = embed[t, hw, :] and b = embed[t-1, hw, :].
    """
    pid_t = tl.program_id(0)   # frame index
    pid_hw = tl.program_id(1)  # spatial position index

    # Output index: flat position in (T * HW,)
    out_idx = pid_t * HW + pid_hw

    # First frame: sentinel value
    if pid_t == 0:
        tl.store(dissim_ptr + out_idx, 255.0)
        return

    # Pointers to current and previous frame embeddings at this spatial pos
    curr_offset = (pid_t * HW + pid_hw) * D
    prev_offset = ((pid_t - 1) * HW + pid_hw) * D

    # Accumulate dot product, norm_a^2, norm_b^2 over hidden dim tiles
    dot_acc = tl.zeros((), dtype=tl.float32)
    norm_a_acc = tl.zeros((), dtype=tl.float32)
    norm_b_acc = tl.zeros((), dtype=tl.float32)

    for d_start in tl.range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D

        a = tl.load(embed_ptr + curr_offset + d_offsets,
                     mask=mask, other=0.0).to(tl.float32)
        b = tl.load(embed_ptr + prev_offset + d_offsets,
                     mask=mask, other=0.0).to(tl.float32)

        dot_acc += tl.sum(a * b)
        norm_a_acc += tl.sum(a * a)
        norm_b_acc += tl.sum(b * b)

    # Cosine similarity: dot / (||a|| * ||b||)
    # Guard against zero norms (degenerate embeddings)
    eps = 1e-8
    norm_product = tl.sqrt(norm_a_acc) * tl.sqrt(norm_b_acc)
    norm_product = tl.where(norm_product > eps, norm_product, eps)
    cos_sim = dot_acc / norm_product

    # Clamp to [-1, 1] for numerical safety
    cos_sim = tl.minimum(tl.maximum(cos_sim, -1.0), 1.0)

    dissimilarity = 1.0 - cos_sim
    tl.store(dissim_ptr + out_idx, dissimilarity)


def _compute_dissimilarity_triton(
    video_embeds: torch.Tensor,
    T: int,
    HW: int,
) -> torch.Tensor:
    """
    Compute cosine dissimilarity scores using the Triton kernel.

    Args:
        video_embeds: (T * HW, D) contiguous tensor.
        T: number of temporal frames.
        HW: spatial tokens per frame (H' * W').

    Returns:
        dissimilarity: (T * HW,) float32 tensor.
    """
    D = video_embeds.shape[-1]
    dissim = torch.empty(T * HW, dtype=torch.float32,
                         device=video_embeds.device)

    grid = (T, HW)
    _fused_evs_dissimilarity_kernel[grid](
        video_embeds,
        dissim,
        T=T,
        HW=HW,
        D=D,
    )
    return dissim


def _selective_mrope_positions(
    retained_indices: torch.Tensor,
    HW: int,
    llm_grid_h: int,
    llm_grid_w: int,
    tokens_per_second: float,
    video_second_per_grid: float | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute mRoPE positions ONLY for retained token indices.

    Given flat indices into the (T * H' * W') grid, recover (t, h, w)
    coordinates and produce the 4-channel mRoPE position vector for
    each retained token.

    Args:
        retained_indices: (K,) long tensor of retained flat indices.
        HW: H' * W' -- spatial tokens per frame.
        llm_grid_h: H // spatial_merge_size.
        llm_grid_w: W // spatial_merge_size.
        tokens_per_second: temporal position scaling factor.
        video_second_per_grid: seconds per grid (float or scalar tensor).
        device: target device.

    Returns:
        positions: (K, 4) long tensor with [t_pos, h_pos, w_pos, grid_w].
    """
    # Recover (t, h, w) from flat index = t * HW + h * llm_grid_w + w
    t_idx = retained_indices // HW
    spatial_idx = retained_indices % HW
    h_idx = spatial_idx // llm_grid_w
    w_idx = spatial_idx % llm_grid_w

    # Temporal position: t * tokens_per_second * video_second_per_grid
    # Determine device for computation (avoid GPU-CPU sync with tensor scalars)
    if isinstance(video_second_per_grid, torch.Tensor):
        t_pos = (t_idx.float() * tokens_per_second
                 * video_second_per_grid.float()).long()
    else:
        t_pos = (t_idx.float()
                 * (tokens_per_second * video_second_per_grid)).long()

    # Grid width broadcast
    grid_w_col = torch.full_like(h_idx, llm_grid_w)

    positions = torch.stack([t_pos, h_idx, w_idx, grid_w_col], dim=1)
    return positions


def fused_evs_prune_with_mrope(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
    tokens_per_second: float = 1.0,
    video_second_per_grid: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused EVS pruning + selective mRoPE position generation.

    Replaces the separate calls to compute_retention_mask() and
    compute_mrope_for_media() with a single pipeline that:
      1. Computes cosine dissimilarity via Triton kernel
      2. Selects top-k tokens via torch.topk
      3. Generates mRoPE positions only for retained tokens

    Args:
        video_embeds: (N, D) video embeddings where N = T * H' * W'.
        video_size_thw: (T, H, W) original grid dimensions.
        spatial_merge_size: merge factor applied to H and W.
        q: pruning rate in [0, 1). Higher means more aggressive pruning.
        tokens_per_second: temporal scaling for mRoPE.
        video_second_per_grid: seconds per video grid position.

    Returns:
        retention_mask: (N,) boolean mask of retained tokens.
        retained_positions: (K, 4) long tensor of mRoPE positions
            for retained tokens, where K = number of retained tokens.
    """
    T, H, W = map(int, video_size_thw)
    llm_grid_h = H // spatial_merge_size
    llm_grid_w = W // spatial_merge_size
    HW = llm_grid_h * llm_grid_w
    N = T * HW

    assert video_embeds.shape[0] == N, (
        f"Expected {N} tokens (T={T}, H'={llm_grid_h}, W'={llm_grid_w}), "
        f"got {video_embeds.shape[0]}"
    )

    # Step 1: Compute dissimilarity via Triton kernel
    # Ensure embeddings are contiguous for the kernel
    embeds_contig = video_embeds.contiguous()
    dissim = _compute_dissimilarity_triton(embeds_contig, T, HW)

    # Step 2: Determine how many tokens to retain
    total_tokens = HW * T
    evs_num_tokens = int(total_tokens * (1 - q))
    min_num_tokens = HW  # at least one full frame
    retain_num_tokens = max(min_num_tokens, evs_num_tokens)

    # Step 3: top-k selection (replaces argsort + slice)
    _, topk_indices = torch.topk(dissim, retain_num_tokens, sorted=False)

    # Build boolean retention mask
    retention_mask = torch.zeros(N, dtype=torch.bool,
                                 device=video_embeds.device)
    retention_mask[topk_indices] = True

    # Step 4: Selective mRoPE -- only compute positions for retained tokens
    # Get sorted retained indices for consistent ordering
    retained_flat = retention_mask.nonzero(as_tuple=True)[0]

    positions = _selective_mrope_positions(
        retained_flat,
        HW=HW,
        llm_grid_h=llm_grid_h,
        llm_grid_w=llm_grid_w,
        tokens_per_second=tokens_per_second,
        video_second_per_grid=video_second_per_grid,
        device=video_embeds.device,
    )

    return retention_mask, positions
