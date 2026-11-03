# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DCP a2a warmup."""

import torch

from vllm.model_executor.warmup.dcp_triton_warmup import (
    _combine_lse_dtypes,
    _num_seqs_reps,
)
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl


def test_num_seqs_reps_cover_every_triton_int_class() -> None:
    """The packed-a2a pack kernel specializes on the Triton int class and the
    next-power-of-2 of ``num_seqs``; the reps must hit every combination, one
    batch size per (class, pow2) pair.
    """
    assert _num_seqs_reps(16) == [1, 2, 3, 5, 9, 16]
    assert _num_seqs_reps(128) == [1, 2, 3, 5, 9, 16, 17, 32, 33, 48, 65, 80]
    assert _num_seqs_reps(3) == [1, 2, 3]


def test_combine_lse_dtypes_match_runtime_producers() -> None:
    """Triton MLA decode returns LSE in the query dtype and forward_impl
    passes it to the combine unconverted, so those layers must warm the
    model dtype alongside the fp32 LSE the prefill backends produce."""
    triton_impl = object.__new__(TritonMLAImpl)
    assert _combine_lse_dtypes(triton_impl, torch.bfloat16) == (
        torch.float32,
        torch.bfloat16,
    )
    # Flash-attention-style impls return fp32 LSE only.
    assert _combine_lse_dtypes(object(), torch.bfloat16) == (torch.float32,)
