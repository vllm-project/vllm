# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Some utilities for logprobs, including logits."""

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY


@torch.compile(backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Counts elements in each row of x that are greater than the corresponding
    value in values.  Use torch.compile to generate an optimized kernel for
    this function. otherwise, it will create additional copies of the input
    tensors and cause memory issues.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.

    """
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)


def constrained_topk(
    scores: torch.Tensor,
    k: int,
    candidates: dict[int, list[int] | bytes] | None = None,
    row_mapping: torch.Tensor | None = None,
    max_rows_per_request: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select within per-request candidates without changing the score source."""
    if k and candidates and any(isinstance(ids, bytes) for ids in candidates.values()):
        sparse = {}
        packed = {}
        for key, ids in candidates.items():
            if isinstance(ids, bytes):
                bits = np.unpackbits(
                    np.frombuffer(ids, dtype=np.uint8), bitorder="little"
                )
                if bits.sum() < k:
                    sparse[key] = np.flatnonzero(bits).tolist()
                else:
                    packed[key] = ids
            else:
                sparse[key] = ids
        values, indices = constrained_topk(
            scores, k, sparse, row_mapping, max_rows_per_request
        )
        if packed:
            return _packed_candidate_topk(
                scores, k, packed, row_mapping, max_rows_per_request, values, indices
            )
        return values, indices
    values, indices = torch.topk(scores, k, dim=-1)
    if not candidates or k == 0:
        return values, indices

    pin_memory = PIN_MEMORY and scores.device.type != "cpu"
    keys = sorted(candidates)
    width = max(len(candidates[key]) for key in keys)
    ids = torch.zeros(
        (len(keys), width), dtype=torch.long, device="cpu", pin_memory=pin_memory
    )
    lengths = [len(candidates[key]) for key in keys]
    ids_np = ids.numpy()
    for i, key in enumerate(keys):
        ids_np[i, : lengths[i]] = candidates[key]
    ids = ids.to(scores.device, non_blocking=True)
    lengths_tensor = torch.tensor(
        lengths, dtype=torch.long, device="cpu", pin_memory=pin_memory
    ).to(scores.device, non_blocking=True)
    rows = torch.tensor(keys, dtype=torch.long, device="cpu", pin_memory=pin_memory).to(
        scores.device, non_blocking=True
    )
    num_rows = scores.shape[0]
    if row_mapping is not None:
        # Fixed-size row selection avoids a CUDA nonzero synchronization.
        matches = rows[:, None] == row_mapping[None, :]
        rows = torch.where(
            matches, torch.arange(num_rows, device=scores.device), num_rows
        )
        repeats = min(max_rows_per_request, num_rows)
        rows = rows.topk(repeats, largest=False).values.flatten()
        ids = ids.repeat_interleave(repeats, dim=0)
        lengths_tensor = lengths_tensor.repeat_interleave(repeats)

    valid = torch.arange(width, device=scores.device)[None, :] < lengths_tensor[:, None]
    candidate_scores = scores[rows.clamp(max=num_rows - 1)[:, None], ids]
    candidate_scores.masked_fill_(~valid, -torch.inf)
    n = min(k, width)
    selected, offsets = torch.topk(candidate_scores, n, dim=-1)
    selected_ids = ids.gather(1, offsets)
    padding = torch.arange(n, device=scores.device)[None, :] >= lengths_tensor[:, None]
    selected_ids = torch.where(padding, selected_ids[:, :1], selected_ids)
    selected = torch.where(padding, selected[:, :1], selected)
    if n < k:
        selected = torch.cat((selected, selected[:, :1].expand(-1, k - n)), dim=1)
        selected_ids = torch.cat(
            (selected_ids, selected_ids[:, :1].expand(-1, k - n)), dim=1
        )
    # Missing padded rows write only to the discarded sentinel row.
    values = torch.cat((values, values.new_empty((1, k))), dim=0)
    indices = torch.cat((indices, indices.new_empty((1, k))), dim=0)
    values[rows] = selected
    indices[rows] = selected_ids
    return values[:-1], indices[:-1]


def _packed_candidate_topk(
    scores: torch.Tensor,
    k: int,
    candidates: dict[int, bytes],
    row_mapping: torch.Tensor | None,
    max_rows_per_request: int,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import xgrammar as xgr

    pin = PIN_MEMORY and scores.device.type != "cpu"
    keys = sorted(candidates)
    vocab_size = scores.shape[1]
    masks_cpu = torch.empty(
        (len(keys), (vocab_size + 31) // 32), dtype=torch.int32, pin_memory=pin
    )
    fallback_cpu = torch.zeros((len(keys), 2 * k), dtype=torch.long, pin_memory=pin)
    fallback_valid_cpu = torch.zeros(
        (len(keys), 2 * k), dtype=torch.bool, pin_memory=pin
    )
    for i, key in enumerate(keys):
        masks_cpu.numpy()[i] = np.frombuffer(candidates[key], dtype="<i4")
        bits = np.unpackbits(
            np.frombuffer(candidates[key], dtype=np.uint8), bitorder="little"
        )[:vocab_size]
        first_ids = np.flatnonzero(bits)[: 2 * k]
        fallback_cpu.numpy()[i, : len(first_ids)] = first_ids
        fallback_valid_cpu.numpy()[i, : len(first_ids)] = True
    masks = masks_cpu.to(scores.device, non_blocking=True)
    fallback = fallback_cpu.to(scores.device, non_blocking=True)
    fallback_valid = fallback_valid_cpu.to(scores.device, non_blocking=True)
    rows = torch.tensor(keys, dtype=torch.long, pin_memory=pin).to(
        scores.device, non_blocking=True
    )
    num_rows = scores.shape[0]
    if row_mapping is not None:
        matches = rows[:, None] == row_mapping[None, :]
        rows = torch.where(
            matches, torch.arange(num_rows, device=scores.device), num_rows
        )
        repeats = min(max_rows_per_request, num_rows)
        rows = rows.topk(repeats, largest=False).values.flatten()
        masks = masks.repeat_interleave(repeats, dim=0)
        fallback = fallback.repeat_interleave(repeats, dim=0)
        fallback_valid = fallback_valid.repeat_interleave(repeats, dim=0)
    source_rows = rows.clamp(max=num_rows - 1)
    selection = scores.index_select(0, source_rows)
    xgr.apply_token_bitmask_inplace(selection, masks)
    selected_ids = selection.topk(k, dim=-1).indices
    allowed = ((masks.gather(1, selected_ids >> 5) >> (selected_ids & 31)) & 1).bool()
    # Masked and allowed -inf scores can tie: fill with distinct allowed IDs.
    available = fallback_valid & ~(
        fallback[:, :, None] == selected_ids[:, None, :]
    ).any(dim=-1)
    replacements = fallback.gather(1, available.to(torch.int32).topk(k, dim=-1).indices)
    missing_rank = (~allowed).long().cumsum(dim=-1).sub(1).clamp(min=0)
    selected_ids = torch.where(
        allowed, selected_ids, replacements.gather(1, missing_rank)
    )
    selected = scores[source_rows[:, None], selected_ids]
    values = torch.cat((values, values.new_empty((1, k))))
    indices = torch.cat((indices, indices.new_empty((1, k))))
    values[rows] = selected
    indices[rows] = selected_ids
    return values[:-1], indices[:-1]
