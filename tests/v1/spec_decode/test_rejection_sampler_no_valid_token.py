# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-checkable coverage for MRV2 spec-decode all-invalid-row handling.

The MRV2 rejection sampler itself needs CUDA/Triton kernels, but the mapping
from all-(-inf) target rows to the requests that consumed them is pure tensor
logic and is pinned here against a reference loop.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import _consumed_invalid_rows


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_consumed_invalid_rows_matches_reference(seed: int):
    rng = torch.Generator().manual_seed(seed)
    num_reqs = 6
    block_sizes = [int(x) for x in torch.randint(1, 5, (num_reqs,), generator=rng)]
    cu_num_logits = torch.tensor([0, *block_sizes], dtype=torch.int32).cumsum(0)
    num_logits = int(cu_num_logits[-1])
    max_rows = max(block_sizes)

    num_sampled = torch.tensor(
        [int(torch.randint(0, bs + 1, (1,), generator=rng)) for bs in block_sizes],
        dtype=torch.int32,
    )
    invalid_rows = torch.rand(num_logits, generator=rng) < 0.5

    got = _consumed_invalid_rows(invalid_rows, cu_num_logits, num_sampled, max_rows)

    ref = torch.zeros(num_reqs, dtype=torch.bool)
    for r in range(num_reqs):
        start = int(cu_num_logits[r])
        ref[r] = any(bool(invalid_rows[start + p]) for p in range(int(num_sampled[r])))

    assert torch.equal(got, ref), f"got={got.tolist()} ref={ref.tolist()}"


def test_consumed_invalid_rows_empty_batch():
    """No requests and no logits rows must return an empty mask, not crash."""
    invalid_rows = torch.zeros(0, dtype=torch.bool)
    cu_num_logits = torch.zeros(1, dtype=torch.int32)
    num_sampled = torch.zeros(0, dtype=torch.int32)
    got = _consumed_invalid_rows(invalid_rows, cu_num_logits, num_sampled, 1)
    assert got.shape == (0,)
