# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA helpers shared by the paged kernels.

Region selection depends on rounding to the OCP E4M3 grid, so it is done
explicitly, also on gfx942 whose native FP8 is E4M3FNUZ.
"""

import torch

_OCP_E4M3 = torch.float8_e4m3fn


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to the OCP E4M3 grid, saturating like Triton's ``float8e4nv`` cast."""
    x = x.to(torch.bfloat16).clamp(-448, 448)
    return x.to(_OCP_E4M3).to(torch.bfloat16)


def indexer_norm(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-centred RMSNorm on ``index_q``, LayerNorm on ``index_k``, in FP32."""
    dtype = index_q.dtype
    q = index_q.float()
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    q = q * (1.0 + q_norm_weight.float())
    k = torch.nn.functional.layer_norm(
        index_k.float(),
        (index_k.shape[-1],),
        k_norm_weight.float(),
        k_norm_bias.float(),
        eps,
    )
    return q.to(dtype), k.to(dtype)


def select_regions(
    scores: torch.Tensor,
    positions: torch.Tensor,
    region_size: int,
    topk: int,
) -> torch.Tensor:
    """Top-``topk`` earlier regions plus the query's own, as ascending int32 ids
    ``[T, G, topk + 1]`` padded with ``-1``. Ties pick the lower id."""
    tokens, groups, regions = scores.shape
    device = scores.device
    own = torch.div(positions.long(), region_size, rounding_mode="floor")
    visible = torch.arange(regions, device=device)[None, :] < own[:, None]
    masked = scores.masked_fill(~visible[:, None, :], float("-inf"))

    keep = min(topk, regions)
    picked = torch.sort(masked, dim=-1, descending=True, stable=True).indices
    picked = picked[..., :keep]
    count = torch.clamp(own, max=topk).view(tokens, 1, 1)
    slot = torch.arange(keep, device=device).view(1, 1, keep)
    # Invisible picks sort last (sentinel = regions) and are then dropped.
    picked = torch.where(slot < count, picked, regions)
    picked = torch.sort(picked, dim=-1).values

    out = torch.full((tokens, groups, topk + 1), -1, device=device, dtype=torch.int64)
    out[..., :keep] = picked.masked_fill(picked == regions, -1)
    out.scatter_(
        -1,
        count.expand(tokens, groups, 1),
        own.view(tokens, 1, 1).expand(tokens, groups, 1),
    )
    return out.to(torch.int32)
