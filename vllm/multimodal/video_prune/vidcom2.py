# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# VidCom2 (Video Compression Commander) video token pruning.
# Liu et al., EMNLP 2025 — https://arxiv.org/abs/2505.14454
# Adapted from the reference implementation:
# https://github.com/xuyang-liu16/VidCom2 (Apache-2.0,
# Copyright (c) 2025 the VidCom2 authors).

import torch
import torch.nn.functional as F

# Multi-scale Gaussian bandwidths from the reference implementation.
_ALPHAS: tuple[float, ...] = tuple(2.0**k for k in range(-3, 2))
_LOW_VAR_CHANNEL_RATIO: float = 0.5
_SOFTMAX_TEMPERATURE: float = 0.01


def compute_retained_tokens_count(
    tokens_per_frame: int, num_frames: int, q: float
) -> int:
    """Number of video tokens retained after VidCom2 pruning.

    The target is `(1 - q) * total_tokens`, i.e. a retention ratio of
    `1 - q` averaged across frames. Because the per-frame budget is floored
    at one token, the global minimum is `num_frames` (one token per frame).
    """
    total_tokens = tokens_per_frame * num_frames
    base_num = int(total_tokens * (1.0 - q))
    return max(num_frames, min(base_num, total_tokens))


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """Compute the VidCom2 retention mask for a single video.

    Args:
        video_embeds: `(T*H*W/merge^2, hidden_size)` post-ViT token features.
        video_size_thw: `(T, H, W)` grid dimensions.
        spatial_merge_size: ViT spatial merge factor (e.g. 2).
        q: Pruning fraction in `[0, 1)`; retention ratio is `1 - q`.

    Returns:
        Flat bool tensor of shape `(T*H*W/merge^2,)`, True for retained
        tokens. The True count equals `compute_retained_tokens_count` so
        placeholders sized at prompt-processing time match exactly.
    """
    T, H, W = map(int, video_size_thw)
    rows = H // spatial_merge_size
    cols = W // spatial_merge_size
    tokens_per_frame = rows * cols
    total_tokens = T * tokens_per_frame

    device = video_embeds.device
    if tokens_per_frame == 0 or total_tokens == 0:
        return torch.ones(0, dtype=torch.bool, device=device)

    target_retained = compute_retained_tokens_count(
        tokens_per_frame=tokens_per_frame, num_frames=T, q=q
    )
    target_retained = min(target_retained, total_tokens)

    # 1. Score in the lowest-variance half of channels.
    variances = video_embeds.var(dim=0, unbiased=False)
    k_channels = max(1, int(video_embeds.size(-1) * _LOW_VAR_CHANNEL_RATIO))
    _, low_var_idx = torch.topk(variances, k=k_channels, largest=False)
    sel = video_embeds.index_select(-1, low_var_idx)

    # 2. Multi-scale Gaussian similarity to video and per-frame centers.
    frames = sel.view(T, tokens_per_frame, sel.size(-1))
    frames = F.normalize(frames, dim=-1)
    vid_center = frames.mean(dim=(0, 1), keepdim=True)  # (1, 1, C)
    frame_center = frames.mean(dim=1, keepdim=True)  # (T, 1, C)
    v_score = _multi_scale_gaussian(frames, vid_center)
    f_score = _multi_scale_gaussian(frames, frame_center)
    # Higher similarity = more redundant; lowest-similarity tokens are kept.
    similarity = v_score + f_score  # (T, tpf)

    # 3. Per-frame dynamic budget: distinctive frames get a larger share.
    base = 1.0 - q
    frame_scores = -v_score.mean(dim=-1)  # (T,)
    probs = F.softmax((frame_scores - frame_scores.max()) / _SOFTMAX_TEMPERATURE, dim=0)
    scales = (base * (1.0 + probs - probs.mean())).clamp(max=1.0)
    ks = (scales * tokens_per_frame).round().long().clamp(min=1, max=tokens_per_frame)

    # 4. Retain the smallest-similarity tokens per frame.
    mask_2d = torch.zeros(T, tokens_per_frame, dtype=torch.bool, device=device)
    for i in range(T):
        k_i = int(ks[i].item())
        if k_i <= 0:
            continue
        _, idx = torch.topk(similarity[i], k=k_i, largest=False, sorted=False)
        mask_2d[i].scatter_(0, idx, True)

    # 5. Reconcile rounding/clamp drift to the exact target count by score.
    flat_mask = mask_2d.view(-1)
    flat_sim = similarity.view(-1)
    current = int(flat_mask.sum().item())
    if current > target_retained:
        drop_n = current - target_retained
        retained_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        retained_sim = flat_sim[retained_idx]
        _, worst = torch.topk(retained_sim, k=drop_n, largest=True, sorted=False)
        flat_mask[retained_idx[worst]] = False
    elif current < target_retained:
        add_n = target_retained - current
        available_idx = (~flat_mask).nonzero(as_tuple=False).squeeze(-1)
        if available_idx.numel() > 0:
            available_sim = flat_sim[available_idx]
            add_n = min(add_n, available_idx.numel())
            _, best = torch.topk(available_sim, k=add_n, largest=False, sorted=False)
            flat_mask[available_idx[best]] = True

    return flat_mask


def _multi_scale_gaussian(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Sum Gaussian kernels over `_ALPHAS`; `(T, N, C) -> (T, N)` scores."""
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2.0 * a)) for a in _ALPHAS)
