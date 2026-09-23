# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU bounds on seq_lens that FlashInfer plans from, and the check of the
exact seq_lens against them. No GPU needed."""

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.v1.attention.backends.utils import check_seq_lens_bounds
from vllm.v1.worker.gpu.model_runner import compute_seq_lens_cpu_lower_bound
from vllm.v1.worker.gpu.spec_decode.speculator import (
    compute_draft_seq_lens_cpu_lower_bound,
)

NUM_SPEC = 3
NUM_REQS = 6
# Six requests padded to eight. Row 3 is prefilling; rows 4 and 5 are the
# first tokens of a request, where the drafts in flight would go below zero.
UPPER = [1328, 21, 466, 65, 2, 3, 0, 0]
IS_PREFILLING = [False, False, False, True, False, False]
LOWER = [1325, 18, 463, 65, 0, 0, 0, 0]
EXACT = [1327, 20, 463, 65, 1, 3]


def test_runner_lower_bound_subtracts_the_drafts_in_flight() -> None:
    upper_np = np.array(UPPER, dtype=np.int32)
    lower_np = compute_seq_lens_cpu_lower_bound(
        upper_np, np.array(IS_PREFILLING), NUM_SPEC, NUM_REQS
    )
    assert lower_np.tolist() == LOWER
    assert lower_np.dtype == np.int32
    # A new array; the upper bound is left as it was.
    assert upper_np.tolist() == UPPER


def test_runner_lower_bound_without_drafts_is_the_upper_bound() -> None:
    upper_np = np.array(UPPER, dtype=np.int32)
    lower_np = compute_seq_lens_cpu_lower_bound(
        upper_np, np.array(IS_PREFILLING), 0, NUM_REQS
    )
    assert lower_np.tolist() == UPPER
    assert lower_np is not upper_np


@pytest.mark.parametrize(
    "step, expected",
    [
        (1, [1323, 16, 461, 63, 0, 0]),
        (2, [1324, 17, 462, 64, 0, 0]),
        (3, [1325, 18, 463, 65, 0, 0]),
    ],
)
def test_draft_lower_bound_per_step(step: int, expected: list[int]) -> None:
    """Draft step ``step`` has ``step`` more tokens than the target lower bound
    and up to NUM_SPEC fewer: the verification may have rejected that many."""
    target_lower = torch.tensor(LOWER, dtype=torch.int32)
    draft_lower = compute_draft_seq_lens_cpu_lower_bound(
        target_lower, step, NUM_SPEC, NUM_REQS, num_reqs_padded=8
    )
    assert draft_lower.tolist() == expected + [0, 0]
    assert draft_lower.dtype == torch.int32
    assert target_lower.tolist() == LOWER


def _bounds() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(LOWER, dtype=torch.int32),
        torch.tensor(UPPER, dtype=torch.int32),
    )


def test_check_passes_within_the_bounds() -> None:
    seq_lens = torch.tensor(EXACT, dtype=torch.int32)
    # Longer bounds are cut to the requests; the bounds are inclusive.
    check_seq_lens_bounds(seq_lens, *_bounds())
    check_seq_lens_bounds(seq_lens, seq_lens, seq_lens)


@pytest.mark.parametrize(
    "row, value",
    [(0, 1329), (1, 17), (3, 64), (3, 66), (4, 3), (5, -1)],
)
def test_check_raises_outside_the_bounds(row: int, value: int) -> None:
    seq_lens = torch.tensor(EXACT, dtype=torch.int32)
    seq_lens[row] = value
    with pytest.raises(RuntimeError):
        check_seq_lens_bounds(seq_lens, *_bounds())


def test_check_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_DEBUG_SEQ_LENS_BOUNDS", raising=False)
    assert envs.VLLM_DEBUG_SEQ_LENS_BOUNDS is False
    monkeypatch.setenv("VLLM_DEBUG_SEQ_LENS_BOUNDS", "1")
    assert envs.VLLM_DEBUG_SEQ_LENS_BOUNDS is True
