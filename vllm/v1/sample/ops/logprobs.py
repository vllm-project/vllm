# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Some utilities for logprobs, including logits."""

import torch

from vllm.platforms import current_platform


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
    candidates: dict[int, list[int]] | None = None,
    row_mapping: torch.Tensor | None = None,
    max_rows_per_request: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select within per-request candidates without changing the score source."""
    values, indices = torch.topk(scores, k, dim=-1)
    if not candidates or k == 0:
        return values, indices

    keys = sorted(candidates)
    width = max(len(candidates[key]) for key in keys)
    ids = torch.zeros((len(keys), width), dtype=torch.long, device="cpu")
    lengths = [len(candidates[key]) for key in keys]
    for i, key in enumerate(keys):
        ids[i, : lengths[i]] = torch.tensor(
            candidates[key], dtype=torch.long, device="cpu"
        )
    ids = ids.to(scores.device)
    lengths_tensor = torch.tensor(lengths, device=scores.device)
    rows = torch.tensor(keys, device=scores.device)
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
