# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic prefill top-k for the DSA sparse attention indexer."""

import torch


def stable_prefill_topk_from_valid_range(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Select per-row top-k indices over columns in ``[ks, ke)``.

    Invalid columns are dropped after a stable descending argsort, so a
    valid ``-inf`` score cannot lose to a masked column. Equal finite
    scores keep the smaller column index. Writes into ``topk_indices``;
    unused slots are ``-1``.

    Args:
        logits: Prefill indexer scores, shape ``(num_rows, cols)``.
        cu_seqlen_ks: Inclusive start column per row.
        cu_seqlen_ke: Exclusive end column per row.
        topk_indices: Output index buffer, shape
            ``(num_rows, >= topk_tokens)``.
        topk_tokens: Number of indices to write per row.

    Raises:
        ValueError: If shapes are inconsistent, or CPU ``ks``/``ke``
            fall outside the logit columns.
    """
    if logits.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("stable prefill topk expects 2D logits and indices")
    num_rows, cols = logits.shape
    if topk_indices.shape[0] != num_rows:
        raise ValueError("topk rows differ from logits")
    if topk_tokens <= 0 or topk_indices.shape[1] < topk_tokens:
        raise ValueError("topk_tokens wider than index buffer")
    ks = cu_seqlen_ks.to(device=logits.device, dtype=torch.int64).reshape(-1)
    ke = cu_seqlen_ke.to(device=logits.device, dtype=torch.int64).reshape(-1)
    if ks.numel() != num_rows or ke.numel() != num_rows:
        raise ValueError("ks/ke rows differ from logits")
    # CPU contract checks stay strict. On CUDA, skip .any()/.item() host
    # syncs; production ks/ke come from scheduler metadata and the mask
    # below only keeps columns in [ks, ke).
    if ks.device.type == "cpu" and bool(
        (ks < 0).any() or (ke < ks).any() or (ke > cols).any()
    ):
        raise ValueError("ks/ke outside logits columns")
    k = min(int(topk_tokens), cols)
    if cols >= (1 << 30):
        raise ValueError("logits wider than packed column field")
    if num_rows == 0 or cols == 0 or k == 0:
        if k:
            topk_indices[:, :k] = -1
        return
    col = torch.arange(cols, device=logits.device, dtype=torch.int64)
    ok = (col[None, :] >= ks[:, None]) & (col[None, :] < ke[:, None])
    # Invalid columns are filled with -inf for the argsort, then dropped
    # so a valid -inf cannot lose to an invalid column.
    masked = logits.masked_fill(~ok, float("-inf"))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    is_valid = ok.gather(1, order)
    rank = is_valid.to(torch.int64).cumsum(dim=1)
    slot = torch.where(is_valid, rank - 1, torch.full_like(rank, k))
    chosen = torch.full(
        (num_rows, k),
        -1,
        dtype=topk_indices.dtype,
        device=logits.device,
    )
    rows_i = torch.arange(num_rows, device=logits.device)[:, None].expand_as(order)
    sel = slot < k
    chosen[rows_i[sel], slot[sel]] = order[sel].to(topk_indices.dtype)
    width = (ke - ks).clamp(min=0)
    keep = torch.arange(k, device=logits.device)[None, :] < width[:, None]
    chosen = torch.where(keep, chosen, torch.full_like(chosen, -1))
    topk_indices[:, :k] = chosen
