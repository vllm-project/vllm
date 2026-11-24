# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLM blocks as varlen segments: a query in an inclusive block `[s, t]`
sees keys `[0, t]`, so each block's queries run as one dense segment."""

from typing import NamedTuple

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata


class DocRangeSegments(NamedTuple):
    """Varlen inputs that replace the per-request ones of FlashAttentionMetadata."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    # One flag for all segments, or int32 per segment (1 = causal).
    causal: bool | torch.Tensor


def build_doc_range_segments(
    common_attn_metadata: CommonAttentionMetadata,
) -> DocRangeSegments | None:
    """The step's segments, or None if no request carries blocks."""
    ranges = common_attn_metadata.mm_req_doc_ranges
    if not ranges or not any(ranges.values()):
        return None
    query_start_loc = common_attn_metadata.query_start_loc_cpu.numpy()
    # Exact for the prefill rows that carry blocks.
    assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound.numpy()
    # (q_start, request, key cap, causal) per segment.
    segments = []
    for i in range(common_attn_metadata.num_reqs):
        q0, q1 = int(query_start_loc[i]), int(query_start_loc[i + 1])
        if q0 == q1:
            continue
        if not ranges.get(i):
            # One query sees the keys up to itself either way; keep it dense
            # so a batch of blocks and decodes stays one dense call.
            segments.append((q0, i, np.iinfo(np.int32).max, q1 - q0 > 1))
            continue
        # The blocks tile the prompt, and a step ends at the end of a block.
        end = int(seq_lens_cpu[i])
        begin = end - (q1 - q0)
        for s, t in ranges[i]:
            if begin <= t and s < end:
                segments.append((q0 + max(s, begin) - begin, i, t + 1, False))
    q_starts, reqs, key_caps, causal = (np.array(c) for c in zip(*segments))
    q_starts = np.append(q_starts, query_start_loc[common_attn_metadata.num_reqs])
    device = common_attn_metadata.block_table_tensor.device
    req_idx = async_tensor_h2d(reqs, device)
    if causal.all() or not causal.any():
        seg_causal: bool | torch.Tensor = bool(causal[0])
    else:
        seg_causal = async_tensor_h2d(causal.astype(np.int32), device)
    return DocRangeSegments(
        query_start_loc=async_tensor_h2d(q_starts.astype(np.int32), device),
        # Decode rows read their exact length on the device.
        seq_lens=torch.minimum(
            common_attn_metadata.seq_lens.index_select(0, req_idx),
            async_tensor_h2d(key_caps.astype(np.int32), device),
        ),
        block_table=common_attn_metadata.block_table_tensor.index_select(0, req_idx),
        max_query_len=int(np.diff(q_starts).max()),
        max_seq_len=int(np.minimum(seq_lens_cpu[reqs], key_caps).max()),
        causal=seg_causal,
    )
