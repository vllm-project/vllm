# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter's step-word kernels count one selection generation per catch-up."""

import pytest
import torch

from vllm.models.glm53.nvidia import draft as glm53_draft

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _words() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a fresh drafter's position, accept and context + generation words."""
    device = torch.device("cuda")
    return (
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.ones(2, dtype=torch.int32, device=device),
    )


def test_a_captured_catchup_counts_a_generation_and_sets_the_words():
    positions = torch.tensor([100], dtype=torch.int64, device="cuda")
    accept = torch.tensor([2], dtype=torch.int64, device="cuda")
    position, accepted, context = _words()
    for _ in range(3):
        glm53_draft._catchup_words_kernel[(1,)](
            positions, accept, position, accepted, context
        )
    assert context.tolist() == [103, 4]
    assert (position.item(), accepted.item()) == (100, 2)


def test_an_eager_catchup_counts_a_generation_and_moves_the_last_rows():
    rows_per_step, width, tokens = 4, 1024, 8
    device = torch.device("cuda")
    last = torch.tensor([6], dtype=torch.int64, device=device)
    positions = torch.arange(tokens, dtype=torch.int64, device=device) + 50
    token_ids = torch.arange(tokens, dtype=torch.int64, device=device)
    rows = torch.randn(tokens, width, device=device).to(torch.bfloat16)
    kept = rows.clone()
    position, accepted, context = _words()
    glm53_draft._sync_inputs_kernel[(1,)](
        last,
        positions,
        token_ids,
        rows,
        position,
        accepted,
        context,
        tokens,
        T=rows_per_step,
        WIDTH=width,
        BLOCK=1024,
    )
    assert (position.item(), accepted.item()) == (53, 3)
    assert context.tolist() == [57, 2]
    assert torch.equal(rows[:rows_per_step], kept[3:7])
    assert token_ids[:rows_per_step].tolist() == [3, 4, 5, 6]
