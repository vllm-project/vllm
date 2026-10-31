# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA candidate-block scores come from a workspace buffer reserved before
the workspace is locked, and the selection is identical to a fresh scores
tensor at every step size the buffer covers."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for the candidate-block kernels", allow_module_level=True
    )

from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    reserve_candidate_scores,
    select_candidate_blocks,
)
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
)

DEVICE = torch.device("cuda")
WIDTH = 65536
BLOCK = 8
NBLOCKS = WIDTH // BLOCK
TOPK_BLOCKS = 16
MAX_ROWS = 48


def _select(logits: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    out = torch.full(
        (logits.shape[0], TOPK_BLOCKS), -7, dtype=torch.int32, device=DEVICE
    )
    select_candidate_blocks(
        logits, torch.zeros_like(visible), visible, TOPK_BLOCKS, BLOCK, out
    )
    return out


@pytest.fixture
def workspace():
    reset_workspace_manager()
    init_workspace_manager(DEVICE)
    yield
    reset_workspace_manager()


@pytest.mark.parametrize("rows", [MAX_ROWS, 30, 1])
def test_reserved_buffer_matches_fresh_scores(workspace, rows: int) -> None:
    gen = torch.Generator(device=DEVICE).manual_seed(rows)
    logits = torch.randn(rows, WIDTH, device=DEVICE, generator=gen) * 4.0
    visible = torch.randint(
        BLOCK, WIDTH + 1, (rows,), dtype=torch.int32, device=DEVICE, generator=gen
    )

    # Without a workspace the kernel scores into a fresh tensor: the reference.
    reset_workspace_manager()
    ref = _select(logits, visible)

    init_workspace_manager(DEVICE)
    reserve_candidate_scores(MAX_ROWS * NBLOCKS, DEVICE)
    lock_workspace()
    # A previous, larger step leaves stale scores in the buffer.
    stale = torch.randn(MAX_ROWS, WIDTH, device=DEVICE, generator=gen)
    _select(stale, torch.full((MAX_ROWS,), WIDTH, dtype=torch.int32, device=DEVICE))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    got = _select(logits, visible)
    torch.cuda.synchronize()
    peak_growth = torch.cuda.max_memory_allocated() - before

    assert torch.equal(got, ref)
    # The scores (rows x NBLOCKS fp32) were not allocated for this step; the
    # top-k values and indices (rows x TOPK_BLOCKS) are all that is left.
    assert peak_growth < rows * NBLOCKS * 4


def test_step_larger_than_the_buffer_falls_back(workspace) -> None:
    rows = MAX_ROWS + 1
    gen = torch.Generator(device=DEVICE).manual_seed(rows)
    logits = torch.randn(rows, WIDTH, device=DEVICE, generator=gen) * 4.0
    visible = torch.full((rows,), WIDTH, dtype=torch.int32, device=DEVICE)

    reset_workspace_manager()
    ref = _select(logits, visible)

    init_workspace_manager(DEVICE)
    reserve_candidate_scores(MAX_ROWS * NBLOCKS, DEVICE)
    lock_workspace()
    assert torch.equal(_select(logits, visible), ref)
