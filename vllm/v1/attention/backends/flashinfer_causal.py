# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request partitioning and masks for FlashInfer mixed causal attention."""

import torch


def causal_group_indices(
    causal: torch.Tensor, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor
) -> list[tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Partition CPU request, query-token and KV-page indices by causality."""
    if causal.device.type != "cpu" or causal.dtype not in (torch.bool, torch.int32):
        raise ValueError("causal must be a CPU boolean or int32 tensor")
    if causal.ndim != 1 or causal.numel() != qo_indptr.numel() - 1:
        raise ValueError("causal must contain one flag per request")
    if kv_indptr.numel() != qo_indptr.numel():
        raise ValueError("query and KV indptr must describe the same requests")
    causal = causal.bool()
    groups = []
    for mode in (True, False):
        requests = (causal == mode).nonzero(as_tuple=True)[0]
        if requests.numel() == 0:
            continue
        tokens = torch.cat(
            [torch.arange(qo_indptr[i], qo_indptr[i + 1]) for i in requests]
        )
        pages = torch.cat(
            [torch.arange(kv_indptr[i], kv_indptr[i + 1]) for i in requests]
        )
        groups.append((mode, requests, tokens, pages))
    return groups


def symmetric_window_mask(
    query_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    window_left: int,
    device: torch.device,
) -> torch.Tensor:
    """Flatten bottom-right-aligned symmetric local masks in request order.

    FlashInfer's window_left only bounds past tokens. Bidirectional sliding
    attention also needs the same bound on future tokens, supplied as a mask.
    """
    masks = []
    for query_len, kv_len in zip(query_lens.tolist(), kv_lens.tolist()):
        query_positions = torch.arange(kv_len - query_len, kv_len, device=device)
        key_positions = torch.arange(kv_len, device=device)
        mask = (query_positions[:, None] - key_positions[None, :]).abs()
        masks.append((mask <= window_left).reshape(-1))
    return torch.cat(masks)
