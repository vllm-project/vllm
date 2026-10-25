# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-agnostic pieces of the MLA and sparse-MLA backend suites.

MLA attention runs in the absorbed ("MQA") form: every head attends to one
shared latent row per token, ``[kv_c (kv_lora_rank) ; k_pe (rope)]``, and the
value is ``kv_c``. Dense MLA is sparse MLA whose index set is every visible
position, so one reference serves both.

- ``sparse_topk_indices``: top-k logical indices with the indexer's contract.
- ``ref_sparse_mla``: float32 absorbed-MLA attention over given cache slots.
- ``fp8_representable``: round a tensor to what an fp8 cache stores.
"""

import torch

from tests.v1.attention.backends.harness import PagedBatch
from vllm.platforms import current_platform


def token_positions(batch: PagedBatch) -> list[int]:
    """Absolute position of every query token, in token order."""
    positions = []
    for q_len, seq_len in zip(batch.query_lens, batch.seq_lens):
        positions.extend(range(seq_len - q_len, seq_len))
    return positions


def sparse_topk_indices(
    batch: PagedBatch,
    topk: int,
    *,
    width: int | None = None,
    dense: bool = False,
    seed: int = 0,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Logical (per-request position) top-k indices for every query token.

    Follows the DSA indexer's output contract: the first ``min(pos + 1, topk)``
    entries of a row are distinct visible positions in arbitrary (score) order,
    the rest are -1. ``dense`` selects every visible position (only valid when
    every row fits). ``width`` > topk pads the buffer with -1 columns, as
    shared top-k buffers sized for a wider indexer are.
    """
    width = width or topk
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for pos in token_positions(batch):
        visible = pos + 1
        if dense or visible <= topk:
            assert visible <= topk, "dense indices need every row to fit in topk"
            chosen = torch.randperm(visible, generator=gen)
        else:
            chosen = torch.randperm(visible, generator=gen)[:topk]
        row = torch.full((width,), -1, dtype=torch.int32)
        row[: chosen.numel()] = chosen.to(torch.int32)
        rows.append(row)
    return torch.stack(rows).to(device)


def logical_to_slots(
    batch: PagedBatch, topk_indices: torch.Tensor
) -> list[torch.Tensor]:
    """Cache slots each query token attends to (invalid entries dropped)."""
    slots = []
    token = 0
    for req, q_len in enumerate(batch.query_lens):
        req_slots = batch.slots[req]
        for _ in range(q_len):
            row = topk_indices[token].long()
            slots.append(req_slots[row[row >= 0]])
            token += 1
    return slots


def fp8_representable(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Values a per-tensor fp8 cache with ``scale`` stores exactly."""
    fp8 = current_platform.fp8_dtype()
    return ((x.float() / scale).to(fp8).float() * scale).to(x.dtype)


def ref_sparse_mla(
    q: torch.Tensor,
    kv_rows: torch.Tensor,
    token_slots: list[torch.Tensor],
    scale: float,
    kv_lora_rank: int,
    *,
    sinks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 absorbed MLA: out[t, h] = softmax(scale * q[t, h] . K[S_t]) V[S_t].

    Args:
        q: [num_tokens, H, kv_lora_rank + rope] absorbed query.
        kv_rows: [num_slots, kv_lora_rank + rope] latent cache rows by slot.
        token_slots: per token, the cache slots it attends to.
        sinks: [H] per-head sink logits (an extra softmax column, no value).

    Returns:
        ([num_tokens, H, kv_lora_rank] latent output, [num_tokens, H] LSE).
        A token with no slots and no sink gets zeros and LSE -inf.

    """
    num_tokens, num_heads, _ = q.shape
    out = torch.zeros(
        num_tokens, num_heads, kv_lora_rank, dtype=torch.float32, device=q.device
    )
    lse = torch.full(
        (num_tokens, num_heads), float("-inf"), dtype=torch.float32, device=q.device
    )
    for t, slots in enumerate(token_slots):
        k = kv_rows[slots].float()
        logits = torch.einsum("hd,kd->hk", q[t].float(), k) * scale
        if sinks is not None:
            logits = torch.cat([logits, sinks.float()[:, None]], dim=-1)
        if logits.shape[-1] == 0:
            continue
        lse[t] = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        if sinks is not None:
            probs = probs[:, :-1]
        out[t] = probs @ k[:, :kv_lora_rank]
    return out, lse
