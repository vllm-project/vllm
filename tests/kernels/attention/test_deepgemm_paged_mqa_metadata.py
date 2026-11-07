# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM's paged MQA logits metadata kernel handles all-zero context
lengths itself (profile dummy runs, a DCP rank that owns no KV of the batch).

get_paged_mqa_logits_metadata relies on this and passes such batches through
without a host-side check, which would sync the GPU on every indexer build.

Run this on both SM90 (Hopper) and SM100 (Blackwell): the metadata kernel has
a separate implementation per architecture, and each handles the all-zero case
in its own way (SM90 checks for a zero total, SM100 takes its tail-sentinel
branch).
"""

import pytest
import torch

from vllm.utils import deep_gemm


def _metadata(context_lens: torch.Tensor, next_n: int):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # The indexer always builds varlen metadata (the 26/09/30 DeepGEMM
    # release dropped the fixed next_n form): one row per query token,
    # ``indices`` maps each row to its request.
    batch = context_lens.shape[0]
    indices = torch.arange(batch * next_n, dtype=torch.int32, device="cuda")
    return deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.reshape(-1, 1).contiguous(),
        64,
        num_sms,
        indices=indices // next_n,
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and deep_gemm.has_deep_gemm()),
    reason="needs CUDA and DeepGEMM",
)
@pytest.mark.parametrize("next_n", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 7, 300])
def test_all_zero_context_lens_schedule_no_work(batch, next_n):
    lens = torch.zeros((batch, next_n), dtype=torch.int32, device="cuda")

    meta = _metadata(lens, next_n).cpu()

    # SM i works on [meta[i], meta[i + 1]) in (q, kv split) order. Every range
    # is empty: all entries are one sentinel past the last query, split 0.
    assert torch.equal(meta, meta[:1].expand_as(meta))
    assert meta[0, 0] >= batch
    assert meta[0, 1] == 0


@pytest.mark.skipif(
    not (torch.cuda.is_available() and deep_gemm.has_deep_gemm()),
    reason="needs CUDA and DeepGEMM",
)
def test_mixed_zero_context_lens_schedule_work():
    lens = torch.zeros((8, 1), dtype=torch.int32, device="cuda")
    lens[5] = 4096

    meta = _metadata(lens, 1).cpu()

    # Work exists, so the ranges are not all empty.
    assert not torch.equal(meta, meta[:1].expand_as(meta))
