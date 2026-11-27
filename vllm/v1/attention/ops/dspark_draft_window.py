# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StreamingLLM page table for the DSpark draft.

The draft attends to the first ``sink`` tokens plus the last ``window``
tokens. The target verification backend is separate and still sees the
full context. RoPE is already stored in K, so the dropped middle is not
reapplied. The column count is fixed so a CUDA graph can capture the gather.
"""

import math

import torch

# The draft attends to this span. The cache keeps the same span: a full
# 262k bf16 draft page would pad the UltraQuant pages up to that size.
DSPARK_DRAFT_SINK = 4
DSPARK_DRAFT_WINDOW = 2048


def build_draft_window_sink_tables(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    window: int,
    sink: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack sink pages in front of the recent window.

    Short sequences, those no longer than ``sink + window``, keep their real
    length. The returned ``seq_lens`` are the packed lengths the kernel walks.
    """
    if window < 0:
        return block_table, seq_lens

    device = block_table.device
    dtype = block_table.dtype
    if seq_lens.device != device:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return block_table, seq_lens
        seq_lens = seq_lens.to(device=device)

    b = int(block_size)
    sink_blocks = math.ceil(int(sink) / b) if sink > 0 else 0
    win_blocks = math.ceil(max(1, int(window)) / b) + 1
    width = max(1, sink_blocks + win_blocks)
    num_reqs = seq_lens.shape[0]
    max_blocks = block_table.shape[1]
    max_col = max(0, int(max_blocks) - 1)

    used = torch.div(seq_lens + (b - 1), b, rounding_mode="floor").clamp(
        min=1, max=max_blocks
    )
    short = seq_lens <= (int(sink) + int(window))
    win_start = torch.div(seq_lens - int(window), b, rounding_mode="floor").clamp_min(0)
    win_start = torch.maximum(win_start, torch.full_like(win_start, sink_blocks))
    win_start = torch.where(short, torch.zeros_like(win_start), win_start)
    sink_n = torch.where(
        short,
        used,
        torch.full_like(used, sink_blocks).clamp(max=used),
    )
    win_n = (used - win_start).clamp_min(0)
    win_n = torch.where(short, torch.zeros_like(win_n), win_n)

    sink_tokens = sink_n * b
    window_tokens = seq_lens - win_start * b
    packed = torch.where(short, seq_lens, sink_tokens + window_tokens).to(
        dtype=seq_lens.dtype
    )

    cols = torch.arange(width, device=device)
    sink_idx = cols.unsqueeze(0).expand(num_reqs, width)
    in_sink = sink_idx < sink_n.unsqueeze(1)
    win_col = sink_idx - sink_blocks
    in_win = (sink_idx >= sink_blocks) & (win_col < win_n.unsqueeze(1)) & (win_col >= 0)
    src = torch.where(in_win, win_start.unsqueeze(1) + win_col, sink_idx)
    src = src.clamp(0, max_col)
    gathered = block_table.gather(1, src.to(dtype=torch.long))
    valid = in_sink | in_win
    page = torch.where(valid, gathered, torch.zeros_like(gathered))
    return page.to(dtype=dtype), packed
