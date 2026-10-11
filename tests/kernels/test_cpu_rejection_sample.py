# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU C++ override of the V1 rejection_random_sample_kernel."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

import vllm._C  # noqa: F401, E402

from vllm.utils.cpu_triton_utils import (  # noqa: E402
    _rejection_random_sample_kernel_impl,
)

PLACEHOLDER = -1


@pytest.mark.parametrize("with_draft_probs", [False, True])
def test_placeholder_draft_is_rejected(with_draft_probs: bool):
    # Request 0 is a normal decode whose second row puts all mass on the last
    # vocab id. Request 1 is padded with placeholder drafts: its first row is
    # the real row (only token 3 allowed), its second row is padding.
    # Reading target_probs at a placeholder id would hit the previous row's
    # last entry (1.0) and accept the placeholder.
    vocab = 16
    target = torch.zeros(4, vocab)
    target[0, 5] = 1.0
    target[1, vocab - 1] = 1.0
    target[2, 3] = 1.0
    target[3, 7] = 1.0
    draft_ids = torch.tensor([5, 5, PLACEHOLDER, PLACEHOLDER])
    cu = torch.tensor([2, 4])
    recovered = torch.tensor([5, vocab - 1, 3, 7])
    bonus = torch.tensor([[9], [11]])
    uniform = torch.full((4,), 0.5, dtype=torch.float64)
    draft_probs = None
    if with_draft_probs:
        # Request 0's second row also gives the last vocab id draft mass, so
        # a misread placeholder would see p / q = 2 and be accepted.
        draft_probs = torch.zeros(4, vocab)
        draft_probs[0, 5] = 1.0
        draft_probs[1, 5] = draft_probs[1, vocab - 1] = 0.5
        draft_probs[2:, 0] = 1.0

    out = torch.full((2, 3), PLACEHOLDER, dtype=torch.int64)
    _rejection_random_sample_kernel_impl(
        out,
        cu,
        draft_ids,
        draft_probs,
        target,
        bonus,
        recovered,
        uniform,
        None,
        2,
        vocab,
        NO_DRAFT_PROBS=draft_probs is None,
    )

    assert out.tolist() == [[5, vocab - 1, PLACEHOLDER], [3, PLACEHOLDER, PLACEHOLDER]]
