# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request history inputs at microbatch boundaries."""

import torch


def slice_lookback_token_ids(
    history: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_slice: slice,
    token_slice: slice,
) -> torch.Tensor:
    """Rebase request lookbacks to the first token executed by this microbatch.

    Rows are padded to the token count so graph replay can change request
    lengths without changing the captured history buffer's address or shape.
    """
    num_tokens = token_slice.stop - token_slice.start
    num_reqs = request_slice.stop - request_slice.start
    depth = history.shape[1]
    output = history.new_full((num_tokens, depth), -1)
    if not num_tokens or not num_reqs:
        return output
    if num_reqs > num_tokens:
        raise ValueError("A microbatch cannot have more request rows than token rows")
    if history.shape[0] == 0 or query_start_loc.numel() < 2:
        return output
    req_ids = torch.arange(
        request_slice.start, request_slice.stop, device=history.device
    )
    num_rows = min(history.shape[0], query_start_loc.numel() - 1)
    safe_req_ids = req_ids.clamp(0, num_rows - 1)
    starts = query_start_loc[safe_req_ids].to(torch.int64)
    ends = query_start_loc[safe_req_ids + 1]
    active = (
        (req_ids < num_rows) & (starts < token_slice.stop) & (ends > token_slice.start)
    )
    chunk_starts = starts.clamp_min(token_slice.start)
    positions = chunk_starts[:, None] - 1 - torch.arange(depth, device=history.device)
    from_batch = positions >= starts[:, None]
    batch_ids = input_ids[positions.clamp(0, input_ids.numel() - 1)].to(history.dtype)
    history_cols = starts[:, None] - 1 - positions
    previous_ids = history[safe_req_ids[:, None], history_cols.clamp(0, depth - 1)]
    valid = active[:, None] & (from_batch | (history_cols < depth))
    output[:num_reqs] = torch.where(
        valid, torch.where(from_batch, batch_ids, previous_ids), -1
    )
    return output


def update_captured_lookback(
    captured: torch.Tensor | None, current: torch.Tensor | None
) -> None:
    """Refresh graph inputs without replacing their captured storage."""
    if captured is None and current is None:
        return
    if captured is None or current is None or captured.shape != current.shape:
        raise ValueError(
            "Microbatch lookback inputs changed their CUDA graph signature"
        )
    captured.copy_(current)
