# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch equivalent of the ``gpu/mm/rope.py`` Triton kernel.

Reached by every M-RoPE and XD-RoPE model, text-only inputs included, so
without it the whole Qwen-VL family fails to start on CPU.
"""

from typing import Any

import torch

from vllm.v1.worker.cpu.input_batch import _run_offsets


def prepare_rope_positions(
    grid: tuple[int, ...],
    positions: torch.Tensor,
    positions_stride: int,
    prefill_positions: torch.Tensor,
    prefill_positions_stride0: int,
    prefill_positions_stride1: int,
    prefill_delta: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    NUM_DIMS: int = 3,
    **kwargs: Any,
) -> None:
    num_reqs = grid[0]
    req = idx_mapping[:num_reqs].long()
    starts = query_start_loc[:num_reqs].long()
    query_lens = query_start_loc[1 : num_reqs + 1].long() - starts

    total, offsets = _run_offsets(query_lens)
    if total == 0:
        return

    num_computed = num_computed_tokens[req].long()
    orig_pos = torch.repeat_interleave(num_computed, query_lens) + offsets
    dst = torch.repeat_interleave(starts, query_lens) + offsets

    # A prefilling request reads the per-modality positions staged for it; a
    # decoding one keeps counting from the prompt, offset by its own delta.
    prefilling = torch.repeat_interleave(
        num_computed < prefill_lens[req].long(), query_lens
    )
    decode_pos = orig_pos + torch.repeat_interleave(
        prefill_delta[req].long(), query_lens
    )

    # Rows are [num_dims * req + dim], which is what the kernel's two strides
    # come out to on the staged buffer.
    rows = torch.repeat_interleave(req, query_lens) * NUM_DIMS
    # The gather covers decode tokens as well, whose position can sit past the
    # staged prompt; clamping keeps it in bounds without splitting the batch.
    cols = orig_pos.clamp_max(prefill_positions.shape[1] - 1)

    for dim in range(NUM_DIMS):
        staged = prefill_positions[rows + dim, cols]
        positions[dim, dst] = torch.where(prefilling, staged.long(), decode_pos).to(
            positions.dtype
        )
