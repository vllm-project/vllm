# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seeding the align-mode running state block for (resumed) requests.

Regression: the seed divided by the SCHEDULER block size (16) instead of the
MAMBA block size (880 after page unification on hybrid models). A request
resumed from a retained checkpoint at 107,360 tokens was seeded to column
6,709 instead of 121 — the first align precopy then dereferenced a garbage
block-table column: a neighbouring request's state at moderate lengths
(silent corruption), unmapped memory at ~100k+ (CUDA illegal memory access).
"""

import pytest

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

pytestmark = pytest.mark.cpu_test

MAMBA_BLOCK = 880
SCHEDULER_BLOCK = 16


@pytest.mark.parametrize(
    ("num_computed", "expected"),
    [
        (0, -1),  # fresh request: -1 under any divisor (the accidental safety)
        (880, 0),  # exactly one block computed: state lives in column 0
        (8800, 9),  # retention checkpoint at the first grid point
        (107360, 121),  # the observed crash case: NOT 6709
        (107359, 121),
        (107361, 122),
    ],
)
def test_seed_uses_mamba_block_size(num_computed: int, expected: int) -> None:
    assert (
        MambaHybridModelState._seed_state_block_idx(num_computed, MAMBA_BLOCK)
        == expected
    )


def test_scheduler_block_seed_is_out_of_range() -> None:
    """Documents the regression magnitude: the wrong divisor lands ~55x past
    the ~298-column block-table row of a 262k-context hybrid model."""
    wrong = MambaHybridModelState._seed_state_block_idx(107360, SCHEDULER_BLOCK)
    right = MambaHybridModelState._seed_state_block_idx(107360, MAMBA_BLOCK)
    assert wrong == 6709 and right == 121
