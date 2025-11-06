# ABOUTME: Device-side helpers for EPS union gating.
# ABOUTME: Selects block groups using JL scores directly on torch tensors.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.context import get_eps_context
from vllm.v1.eps.state import EpsJLState
from vllm.v1.eps.telemetry import EpsStepCounters


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


def apply_device_union(
    *,
    layer: torch.nn.Module,
    query_tokens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    attn_metadata,
) -> bool:
    ctx = get_eps_context()
    if ctx is None or not ctx.enabled or ctx.cudagraph_capture:
        return False

    layer_name = getattr(layer, "_eps_layer_name", None)
    if layer_name is None or layer_name not in ctx.layer_map:
        return False

    layer_info = ctx.layer_map[layer_name]
    device_groups = ctx.device_union_groups or set()
    if device_groups and layer_info.group_id not in device_groups:
        return False

    group_runtime = ctx.group_runtimes[layer_info.group_id]
    request_runtimes = group_runtime.request_runtimes

    seq_lens = attn_metadata.seq_lens
    if seq_lens is None or seq_lens.numel() == 0:
        return False

    num_reqs = min(len(request_runtimes), seq_lens.shape[0])
    if num_reqs == 0:
        return False

    block_table = attn_metadata.block_table
    if block_table is None or block_table.shape[0] < num_reqs:
        return False

    query_start_loc = attn_metadata.query_start_loc.to(
        device=query_tokens.device, dtype=torch.long
    )
    if query_start_loc.numel() < num_reqs + 1:
        return False

    last_token_indices = (query_start_loc[1 : num_reqs + 1] - 1).clamp_min_(0)
    query_heads = query_tokens.index_select(0, last_token_indices)

    cfg = ctx.cfg
    counters = ctx.device_counters or EpsStepCounters()
    ctx.device_counters = counters

    block_size = group_runtime.block_size
    device = block_table.device
    sentinel_value = block_table.new_full((), cfg.sentinel, dtype=block_table.dtype)

    before_unique: set[int] = set()
    after_unique: set[int] = set()
    total_groups = 0
    total_groups_kept = 0
    total_blocks = 0
    total_blocks_kept = 0

    seq_lens_device = seq_lens.to(device)

    for req_idx in range(num_reqs):
        runtime = request_runtimes[req_idx]
        state = runtime.state
        seq_len = int(seq_lens_device[req_idx].item())
        if seq_len <= 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        if num_blocks <= 0:
            continue

        block_ids = block_table[req_idx, :num_blocks].to(device=device, dtype=torch.int64)
        before_unique.update(int(b) for b in block_ids.detach().cpu().tolist())

        if state is None:
            keep_groups = (num_blocks + cfg.group_blocks - 1) // cfg.group_blocks
            total_groups += keep_groups
            total_groups_kept += keep_groups
            total_blocks += num_blocks
            total_blocks_kept += num_blocks
            after_unique.update(int(b) for b in block_ids.detach().cpu().tolist())
            continue

        decision = union_select_for_request(
            cfg=cfg,
            layer_index=layer_info.layer_index,
            q_attn=query_heads[req_idx],
            state=state,
            block_ids=block_ids,
            block_mapping=runtime.block_mapping,
            seq_len=seq_len,
            block_size=block_size,
            num_attn_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        kept_ids = decision.kept_block_ids.to(device=device, dtype=block_table.dtype)
        kept_count = kept_ids.numel()
        if kept_count < num_blocks:
            block_table[req_idx, kept_count:num_blocks] = sentinel_value
        block_table[req_idx, :kept_count] = kept_ids

        seq_lens_device[req_idx] = torch.tensor(
            decision.new_seq_len, device=device, dtype=seq_lens_device.dtype
        )

        total_blocks += num_blocks
        total_blocks_kept += kept_count if kept_count > 0 else num_blocks

        groups_total = decision.groups_total or (
            (num_blocks + cfg.group_blocks - 1) // cfg.group_blocks
        )
        if decision.group_keep_mask.numel() > 0:
            groups_kept = int(decision.group_keep_mask.sum().item())
        else:
            groups_kept = groups_total

        total_groups += groups_total
        total_groups_kept += groups_kept
        after_unique.update(int(b) for b in kept_ids.detach().cpu().tolist())

    seq_lens.copy_(seq_lens_device)

    counters.layers += 1
    if total_blocks:
        counters.blocks_total += total_blocks
        counters.blocks_kept += total_blocks_kept
    if total_groups:
        counters.groups_total += total_groups
        counters.groups_kept += total_groups_kept

    counters.unique_blocks_total += len(before_unique)
    counters.unique_blocks_kept += len(after_unique)

    counters.pages_total = counters.groups_total
    counters.pages_visited = counters.groups_kept
    counters.pages_skipped = counters.groups_total - counters.groups_kept
    counters.pages_unique_total = counters.groups_total
    counters.pages_unique_kept = counters.groups_kept

    if seq_lens_device.numel() > 0:
        attn_metadata.max_seq_len = int(seq_lens_device.max().item())

    return True


__all__ = [
    "UnionDecision",
    "UnionSelectionError",
    "union_select_for_request",
    "apply_device_union",
]
