# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DCP a2a warmup."""

from vllm.model_executor.warmup.dcp_triton_warmup import _num_seqs_reps


def test_num_seqs_reps_cover_every_triton_int_class() -> None:
    """The packed-a2a pack kernel specializes on the Triton int class and the
    next-power-of-2 of ``num_seqs``; the reps must hit every combination, one
    batch size per (class, pow2) pair.
    """
    assert _num_seqs_reps(16) == [1, 2, 3, 5, 9, 16]
    assert _num_seqs_reps(128) == [1, 2, 3, 5, 9, 16, 17, 32, 33, 48, 65, 80]
    assert _num_seqs_reps(3) == [1, 2, 3]
