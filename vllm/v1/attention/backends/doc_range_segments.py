# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLM blocks as varlen segments: a query in an inclusive block `[s, t]`
sees keys `[0, t]`, so each block's queries run as one dense segment."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d


@dataclass
class DocRangeSegments:
    """Varlen inputs that replace the per-request ones."""

    cu_seqlens_q: torch.Tensor  # (num_segments + 1,) int32
    seqused_k: torch.Tensor  # (num_segments,) int32
    block_table: torch.Tensor  # (num_segments, max_blocks) int32
    max_seqlen_q: int
    max_seqlen_k: int
    # True: every segment causal; False: none; tensor: int32 per segment
    # (1 = causal), FA4's dynamic_causal.
    causal: bool | torch.Tensor


def split_doc_range_segments(
    ranges: dict[int, list[tuple[int, int]]],
    query_start_loc_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    num_reqs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment `(q_start, req_idx, k_len, causal)`, plus a final q_start.
    A `k_len` of -1 means the request's device `seq_lens` entry."""
    q_starts: list[int] = []
    reqs: list[int] = []
    k_lens: list[int] = []
    causal: list[bool] = []
    for i in range(num_reqs):
        q0 = int(query_start_loc_cpu[i])
        q_len = int(query_start_loc_cpu[i + 1]) - q0
        if q_len == 0:
            continue
        if not ranges.get(i):
            # One query sees the keys up to itself either way; keep it dense
            # so a batch of blocks and decodes stays one dense call.
            q_starts.append(q0)
            reqs.append(i)
            k_lens.append(-1)
            causal.append(q_len > 1)
            continue
        # The blocks tile the prompt, and a step ends at the end of a block.
        end = int(seq_lens_cpu[i])
        begin = end - q_len
        for s, t in ranges[i]:
            if begin <= t and s < end:
                q_starts.append(q0 + max(s, begin) - begin)
                reqs.append(i)
                k_lens.append(t + 1)
                causal.append(False)
    q_starts.append(int(query_start_loc_cpu[num_reqs]))
    return (
        np.asarray(q_starts, dtype=np.int32),
        np.asarray(reqs, dtype=np.int64),
        np.asarray(k_lens, dtype=np.int32),
        np.asarray(causal, dtype=np.bool_),
    )


def build_doc_range_segments(
    ranges: dict[int, list[tuple[int, int]]],
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_reqs: int,
) -> DocRangeSegments:
    q_starts, reqs, k_lens, causal = split_doc_range_segments(
        ranges,
        query_start_loc_cpu.numpy(),
        seq_lens_cpu.numpy(),
        num_reqs,
    )
    device = block_table.device
    # Host k_len of default segments: their upper bound, for max_seqlen_k.
    host_k = np.where(k_lens >= 0, k_lens, seq_lens_cpu.numpy()[reqs])
    req_idx = async_tensor_h2d(reqs, device)
    k_override = async_tensor_h2d(k_lens, device)
    seqused_k = torch.where(
        k_override >= 0, k_override, seq_lens.index_select(0, req_idx)
    ).to(torch.int32)
    if causal.all():
        seg_causal: bool | torch.Tensor = True
    elif not causal.any():
        seg_causal = False
    else:
        seg_causal = async_tensor_h2d(causal.astype(np.int32), device)
    return DocRangeSegments(
        cu_seqlens_q=async_tensor_h2d(q_starts, device),
        seqused_k=seqused_k,
        block_table=block_table.index_select(0, req_idx),
        max_seqlen_q=int(np.diff(q_starts).max()),
        max_seqlen_k=int(host_k.max()),
        causal=seg_causal,
    )
