# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM90 candidate source's block selection: on SM90 the specialized
decode's `select_candidate_blocks_cooperative` selects the same candidate
set as vLLM's `select_candidate_blocks` (`torch.topk` over the whole
width), eagerly and replayed from a CUDA graph, at the source's decode shapes
(rows of 2048 blocks of 8 positions in logits as wide as a 1M-token model's
262,144 compressed positions) and around the cooperative kernel's limits."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(90),
    reason="needs an SM90 GPU",
)

DEVICE = "cuda"
BLOCK = 8


@pytest.mark.parametrize(
    ("ends", "k", "width"),
    [
        # Six and 24 decode rows of one request each at the source's shape.
        ((2048 * 8 - 5, 9000, 65536, 131072, 262144, 1), 2048, 262144),
        (tuple(range(1, 262145, 262144 // 24))[:24], 2048, 262144),
        # Rows around the 16384-block register-path limit.
        (
            (16384 * 8, 16384 * 8 - 3, 16385 * 8 + 1, 3000 * 8 + 5, 20000 * 8),
            2048,
            1 << 20,
        ),
    ],
)
def test_cooperative_candidates_match_stock(workspace_init, ends, k, width):
    """Each row's candidate set equals the stock selection, including the
    pinned newest block and -inf blocks left as -1."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.candidate_topk import (  # noqa: E501
        select_candidate_blocks_cooperative,
    )

    rows = len(ends)
    logits = torch.empty(rows, width, device=DEVICE)
    row_ends = torch.tensor(ends, dtype=torch.int32, device=DEVICE)

    def fill_logits(seed):
        torch.manual_seed(seed)
        logits.normal_()
        logits[0, : 50 * BLOCK] = float("-inf")
        end = ends[-1]
        logits[-1, (end - 1) // BLOCK * BLOCK : end] = -100.0  # lowest, but pinned

    def select(selector):
        # In-range stale ids: a slot the selection skips would leak one.
        out = torch.full((rows, k), 7, dtype=torch.int32, device=DEVICE)
        selector(out)
        return out

    def stock():
        return select(
            lambda out: select_candidate_blocks(logits, None, row_ends, k, BLOCK, out)
        )

    def cooperative():
        return select(
            lambda out: select_candidate_blocks_cooperative(
                logits, row_ends, BLOCK, out
            )
        )

    def assert_same_sets(got, want):
        assert torch.equal(got.sort(dim=1).values, want.sort(dim=1).values)

    fill_logits(0)
    assert_same_sets(cooperative(), stock())
    cooperative()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = cooperative()
    fill_logits(1)
    graph.replay()
    assert_same_sets(out, stock())
