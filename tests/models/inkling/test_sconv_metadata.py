# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity test for the fused sconv seq-metadata kernel.

``sconv_seq_metadata`` fills the per-token ``seq_idx`` (owning request) and
``query_start`` (first x-row of that request) buffers in a single launch. Actual
tokens must match the searchsorted-based reference, while CUDA graph padding
must be initialized to safe zero values.
"""

import pytest
import torch

from vllm.models.inkling.nvidia.ops.sconv import sconv_seq_metadata
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvMetadataBuilder

CASES = [
    # (query_lens, extra_pad_tokens)
    ([1], 0),  # bsz1 decode
    ([1] * 8, 0),  # uniform decode
    ([1] * 8, 3),  # uniform decode, padded tokens past the last request
    ([2] * 4, 0),  # uniform spec-decode
    ([2048], 0),  # single prefill
    ([517, 1, 1, 33, 1, 128], 0),  # mixed prefill/decode
    ([517, 1, 1, 33, 1, 128], 5),  # mixed, padded
    ([1] * 500, 0),  # many requests (deep binary search)
]


def _ref(query_start_loc: torch.Tensor, num_reqs: int, num_tokens: int):
    cu_seqlens = query_start_loc[: num_reqs + 1].to(torch.int64)
    token_idx = torch.arange(num_tokens, device=cu_seqlens.device, dtype=torch.int64)
    seq_idx = (torch.searchsorted(cu_seqlens, token_idx, right=True) - 1).clamp(
        max=num_reqs - 1
    )
    return seq_idx.to(torch.int32), cu_seqlens[seq_idx].to(torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_lens,extra_pad", CASES)
def test_sconv_seq_metadata_matches_searchsorted(query_lens, extra_pad):
    device = "cuda"
    num_reqs = len(query_lens)
    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0)), dtype=torch.int32
    ).to(device)
    num_actual_tokens = int(query_start_loc[-1])
    num_padded_tokens = num_actual_tokens + extra_pad

    ref_seq, ref_qs = _ref(query_start_loc, num_reqs, num_actual_tokens)

    seq_idx = torch.full((num_padded_tokens,), -1, dtype=torch.int32, device=device)
    query_start = torch.full_like(seq_idx, -1)
    sconv_seq_metadata(
        query_start_loc,
        num_reqs,
        num_actual_tokens,
        seq_idx,
        query_start,
        num_padded_tokens,
    )

    torch.testing.assert_close(seq_idx[:num_actual_tokens], ref_seq, rtol=0, atol=0)
    torch.testing.assert_close(query_start[:num_actual_tokens], ref_qs, rtol=0, atol=0)
    assert torch.count_nonzero(seq_idx[num_actual_tokens:]) == 0
    assert torch.count_nonzero(query_start[num_actual_tokens:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sconv_metadata_reuses_padded_static_buffers():
    device = torch.device("cuda")
    builder = object.__new__(InklingSconvMetadataBuilder)
    builder.seq_idx_buffer = torch.empty(8, dtype=torch.int32, device=device)
    builder.query_start_buffer = torch.empty(8, dtype=torch.int32, device=device)

    class CommonMetadata:
        num_reqs = 1
        num_actual_tokens = 8
        query_start_loc = torch.tensor([0, 3], dtype=torch.int32, device=device)
        query_start_loc_cpu = torch.tensor([0, 3], dtype=torch.int32)
        block_table_tensor = torch.zeros((1, 1), dtype=torch.int32, device=device)
        slot_mapping = torch.tensor(
            [0, 1, 2, -1, -1, -1, -1, -1],
            dtype=torch.int64,
            device=device,
        )

    common = CommonMetadata()
    first = builder.build(0, common)
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            first.block_table,
            first.slot_mapping,
            first.seq_idx,
            first.query_start,
        )
    )

    common.query_start_loc[1] = 5
    common.query_start_loc_cpu[1] = 5
    common.slot_mapping[:5] = torch.arange(5, dtype=torch.int64, device=device)
    second = builder.build(0, common)

    assert second.slot_mapping.shape == (8,)
    assert second.seq_idx.shape == (8,)
    assert second.query_start.shape == (8,)
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            second.block_table,
            second.slot_mapping,
            second.seq_idx,
            second.query_start,
        )
    )
    assert torch.count_nonzero(second.seq_idx[5:]) == 0
    assert torch.count_nonzero(second.query_start[5:]) == 0
    assert torch.all(second.slot_mapping[5:] == -1)
