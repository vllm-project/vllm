# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stale positive topk_ids must never index past expert_map.

topk_ids buffers come from torch.empty and the routers do not overwrite the
padded rows of a CUDA-graph batch, so an entry can be arbitrary POSITIVE
data (the bit pattern of whatever float tensor previously owned the
allocation) -- a `< 0` check alone does not gate the gather, and
expert_map[stale_id] reads past the table: an illegal access that takes
down every TP rank at once. Reported by alexbi29 in
vllm-project/vllm#41834; each site now bounds by the length of the map it
actually indexes (the GLOBAL expert count -- under EP the rank-local count
is the smaller number and is not a safe bound)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

NUM_GLOBAL = 16
NUM_LOCAL = 8


def _expert_map() -> torch.Tensor:
    # EP shard: first half local, rest -1 (foreign).
    m = torch.full((NUM_GLOBAL,), -1, dtype=torch.int32, device="cuda")
    m[:NUM_LOCAL] = torch.arange(NUM_LOCAL, dtype=torch.int32, device="cuda")
    return m


def test_count_expert_num_tokens_survives_stale_positive_ids():
    topk_ids = torch.tensor(
        [[0, 1], [7, 3], [123456789, 2], [98765, 2071690107]],
        dtype=torch.int32,
        device="cuda",
    )
    counts = count_expert_num_tokens(topk_ids, NUM_LOCAL, _expert_map())
    torch.cuda.synchronize()
    expected = torch.zeros(NUM_LOCAL, dtype=torch.int32, device="cuda")
    for e in (0, 1, 7, 3, 2):
        expected[e] += 1
    assert torch.equal(counts, expected), (counts.tolist(), expected.tolist())


def test_moe_fused_mul_sum_skips_stale_positive_ids():
    num_tokens, top_k, size = 4, 2, 32
    inputs = torch.ones(num_tokens, top_k, size, device="cuda", dtype=torch.float32)
    weights = torch.ones(num_tokens, top_k, device="cuda", dtype=torch.float32)
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [123456789, 4], [98765, 2071690107]],
        dtype=torch.int32,
        device="cuda",
    )
    out = moe_fused_mul_sum(
        inputs, weights, topk_ids=topk_ids, expert_map=_expert_map()
    )
    torch.cuda.synchronize()
    # rows 0-1: both experts valid -> 2.0; row 2: one stale id skipped -> 1.0;
    # row 3: both stale -> 0.0
    expected = torch.tensor([2.0, 2.0, 1.0, 0.0], device="cuda")
    torch.testing.assert_close(out[:, 0], expected)
