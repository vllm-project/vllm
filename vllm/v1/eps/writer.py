# ABOUTME: EPS helpers for KV cache write interception.
# ABOUTME: Consumes forward context to push JL updates pre-write.

from __future__ import annotations

import torch

from vllm.v1.eps.context import EpsForwardContext, get_eps_context
from vllm.v1.eps.summarizer import jl_update_block


def _get_target_device(ctx: EpsForwardContext) -> torch.device:
    for group in ctx.group_runtimes:
        for runtime in group.request_runtimes:
            if runtime.state is not None:
                return runtime.state.G.device
    return torch.device("cpu")


def apply_eps_prefill_updates(
    *,
    layer_name: str,
    key: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Update JL sketches for a prefill write prior to cache commit."""
    ctx = get_eps_context()
    if ctx is None or not ctx.enabled or ctx.cudagraph_capture:
        return

    layer_info = ctx.layer_map.get(layer_name)
    if layer_info is None:
        return

    group_runtime = ctx.group_runtimes[layer_info.group_id]
    block_size = group_runtime.block_size
    token_req = ctx.token_request_indices

    if token_req.numel() != key.shape[0]:
        raise ValueError("token_request_indices and key length mismatch")

    slots = slot_mapping.to(torch.int64).cpu()
    if slots.numel() != key.shape[0]:
        raise ValueError("slot_mapping length must match key tokens")

    target_device = _get_target_device(ctx)
    if key.device != target_device:
        key = key.to(target_device)

    total_tokens = slots.numel()
    pos = 0
    while pos < total_tokens:
        req_idx = int(token_req[pos])
        req_runtime = group_runtime.request_runtimes[req_idx]
        state = req_runtime.state
        block_id = int(slots[pos] // block_size)
        logical_idx = req_runtime.block_mapping.get(block_id)

        start = pos
        pos += 1
        while pos < total_tokens:
            if int(token_req[pos]) != req_idx:
                break
            if int(slots[pos] // block_size) != block_id:
                break
            pos += 1
        end = pos

        if state is None or logical_idx is None:
            continue

        group = logical_idx // ctx.cfg.group_blocks
        state.ensure_group_capacity(group + 1)

        token_slice = key[start:end]
        if token_slice.numel() == 0:
            continue
        if token_slice.dim() != 3:
            raise ValueError("Expected key slice shaped [T, H, D]")

        for head in range(token_slice.shape[1]):
            jl_update_block(
                state,
                layer=layer_info.layer_index,
                head=head,
                group=group,
                K_block=token_slice[:, head, :],
            )


__all__ = ["apply_eps_prefill_updates"]
