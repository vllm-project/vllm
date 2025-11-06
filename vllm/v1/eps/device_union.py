# ABOUTME: Device-side helpers for EPS union gating.
# ABOUTME: Selects block groups using JL scores directly on torch tensors.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.state import EpsJLState


@dataclass
class UnionDecision:
    kept_block_ids: torch.Tensor
    block_keep_mask: torch.Tensor
    group_keep_mask: torch.Tensor
    new_seq_len: int
    groups_total: int


class UnionSelectionError(RuntimeError):
    """Raised when EPS device union selection cannot proceed."""


def _kv_head_index_map(num_attn_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_kv_heads <= 0:
        raise UnionSelectionError("num_kv_heads must be positive")
    if num_attn_heads % num_kv_heads != 0:
        raise UnionSelectionError("num_attn_heads must be divisible by num_kv_heads")
    queries_per_kv = num_attn_heads // num_kv_heads
    return torch.arange(num_attn_heads) // queries_per_kv


def _tokens_per_block(seq_len: int, block_size: int, num_blocks: int) -> torch.Tensor:
    if block_size <= 0:
        raise UnionSelectionError("block_size must be positive")
    if num_blocks < 0:
        raise UnionSelectionError("num_blocks must be non-negative")
    if num_blocks == 0:
        return torch.zeros(0, dtype=torch.int64)
    tokens = torch.full((num_blocks,), block_size, dtype=torch.int64)
    remaining = seq_len - (num_blocks - 1) * block_size
    if remaining <= 0:
        remaining = block_size
    tokens[-1] = remaining
    return tokens


def union_select_for_request(
    *,
    cfg: EpsRuntimeConfig,
    layer_index: int,
    q_attn: torch.Tensor,
    state: EpsJLState,
    block_ids: torch.Tensor,
    block_mapping: Dict[int, int],
    seq_len: int,
    block_size: int,
    num_attn_heads: int,
    num_kv_heads: int,
    alpha_coarse: float | None = None,
) -> UnionDecision:
    if block_ids.numel() == 0 or seq_len <= 0:
        return UnionDecision(
            kept_block_ids=block_ids.clone(),
            block_keep_mask=torch.ones_like(block_ids, dtype=torch.bool),
            group_keep_mask=torch.ones(0, dtype=torch.bool),
            new_seq_len=seq_len,
            groups_total=0,
        )

    group_blocks = cfg.group_blocks
    if group_blocks <= 0:
        raise UnionSelectionError("cfg.group_blocks must be positive")
    if num_attn_heads <= 0 or num_kv_heads <= 0:
        raise UnionSelectionError("Number of heads must be positive")
    if state is None:
        keep_mask = torch.ones(block_ids.numel(), dtype=torch.bool)
        return UnionDecision(
            kept_block_ids=block_ids.clone(),
            block_keep_mask=keep_mask,
            group_keep_mask=torch.ones(0, dtype=torch.bool),
            new_seq_len=seq_len,
            groups_total=0,
        )

    device = q_attn.device
    dtype = state.Phi.dtype

    num_blocks = block_ids.numel()
    tokens_per_block = _tokens_per_block(seq_len, block_size, num_blocks).to(device)

    groups_for_blocks = torch.full((num_blocks,), -1, dtype=torch.long, device=device)
    highest_group = -1
    for idx in range(num_blocks):
        block_id = int(block_ids[idx].item())
        logical_idx = block_mapping.get(block_id)
        if logical_idx is None:
            continue
        group_id = logical_idx // group_blocks
        groups_for_blocks[idx] = group_id
        highest_group = max(highest_group, group_id)

    if highest_group < 0:
        keep_mask = torch.ones(num_blocks, dtype=torch.bool, device=device)
        return UnionDecision(
            kept_block_ids=block_ids.clone(),
            block_keep_mask=keep_mask,
            group_keep_mask=torch.ones(0, dtype=torch.bool, device=device),
            new_seq_len=seq_len,
            groups_total=0,
        )

    groups_total = highest_group + 1
    state.ensure_group_capacity(groups_total)

    kv_head_map = _kv_head_index_map(num_attn_heads, num_kv_heads).to(device)
    attn_indices = torch.arange(num_attn_heads, device=device, dtype=torch.long)
    kv_indices = kv_head_map[attn_indices]

    head_scope = cfg.head_scope
    if head_scope == "retrieval":
        pass
    head_mask = torch.ones_like(attn_indices, dtype=torch.bool)
    head_indices = attn_indices[head_mask]
    kv_indices = kv_indices[head_mask]

    q_heads = q_attn.to(device=device, dtype=dtype)[head_indices]
    head_norms = torch.linalg.norm(q_heads, dim=-1)

    q_kv_norm = torch.zeros(state.Phi.shape[0], dtype=dtype, device=device)
    for kv in kv_indices.unique(sorted=False):
        mask = kv_indices == kv
        if mask.any():
            q_kv_norm[kv] = head_norms[mask].max()

    p = groups_total
    frob2 = state.frob2[layer_index, :, :p].to(device=device)
    coarse_union = (q_kv_norm[:, None] * torch.sqrt(torch.clamp(frob2, min=0.0))).amax(dim=0)

    keep_mask_groups = torch.zeros(p, dtype=torch.bool, device=device)
    last_n = min(cfg.last_n, p)
    if last_n > 0:
        keep_mask_groups[p - last_n :] = True
        Tmin = coarse_union[p - last_n :].max()
    else:
        Tmin = torch.tensor(0.0, dtype=dtype, device=device)

    Phi_sel = state.Phi.index_select(0, kv_indices).to(device=device)
    Z = torch.einsum('hd,hdm->hm', q_heads, Phi_sel)
    G = state.G[layer_index].index_select(0, kv_indices)[:, :p].to(device=device)
    tmp = torch.einsum('hpmn,hm->hpm', G, Z)
    jl_sq = torch.einsum('hpm,hm->hp', tmp, Z).clamp_min_(0.0)
    jl_scores = torch.sqrt(jl_sq).amax(dim=0)

    alpha_final = torch.tensor(cfg.alpha, dtype=dtype, device=device)
    alpha_coarse_val = (
        torch.tensor(alpha_coarse, dtype=dtype, device=device)
        if alpha_coarse is not None
        else alpha_final
    )

    for group_id in range(p - 1, -1, -1):
        if keep_mask_groups[group_id]:
            Tmin = torch.maximum(Tmin, jl_scores[group_id])
            continue
        if coarse_union[group_id] < (Tmin / alpha_coarse_val):
            continue
        if jl_scores[group_id] >= (Tmin / alpha_final):
            keep_mask_groups[group_id] = True
            Tmin = torch.maximum(Tmin, jl_scores[group_id])

    frob_sum = frob2.sum(dim=0)
    if cfg.strict:
        keep_mask_groups = keep_mask_groups | (frob_sum == 0)

    if cfg.top_pages is not None and int(keep_mask_groups.sum().item()) > cfg.top_pages:
        scores = jl_scores.clone()
        if last_n > 0:
            scores[p - last_n :] = torch.finfo(scores.dtype).max
        topk = torch.topk(scores, k=cfg.top_pages).indices
        mask = torch.zeros_like(keep_mask_groups)
        mask[topk] = True
        keep_mask_groups = mask

    block_keep_mask = torch.zeros(num_blocks, dtype=torch.bool, device=device)
    for idx in range(num_blocks):
        group_id = int(groups_for_blocks[idx].item())
        if group_id < 0:
            block_keep_mask[idx] = True
        else:
            block_keep_mask[idx] = keep_mask_groups[group_id]

    if not block_keep_mask.any():
        block_keep_mask[-1] = True
        group_last = int(groups_for_blocks[-1].item())
        if group_last >= 0:
            keep_mask_groups[group_last] = True

    kept_block_ids = block_ids[block_keep_mask]
    new_seq_len = int(tokens_per_block[block_keep_mask].sum().item())
    if groups_total <= 0:
        groups_total = (num_blocks + group_blocks - 1) // group_blocks
    if new_seq_len <= 0:
        new_seq_len = seq_len

    return UnionDecision(
        kept_block_ids=kept_block_ids,
        block_keep_mask=block_keep_mask,
        group_keep_mask=keep_mask_groups,
        new_seq_len=new_seq_len,
        groups_total=groups_total,
    )


__all__ = [
    "UnionDecision",
    "UnionSelectionError",
    "union_select_for_request",
]
