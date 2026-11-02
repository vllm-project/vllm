# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential test for the AITER MLA uniform-MTP padding decision.

`_uniform_padded_mtp_qo_len` decides whether a full-CUDA-graph decode batch may
synthesize dummy query rows. It used to reach that decision through eight tensor
dispatches -- sum/item, boolean-mask index, all, eq, nonzero, item, any -- on a
CPU tensor holding one entry per request. It now reads the tensor out once and
decides on Python ints.

That rewrite also **reordered** the checks: the `num_decode_tokens <= sum(...)`
early-out moved below the divisibility and `uniform_qo_len <= 1` guards. Every
one of those paths returns 0, so the result should be unchanged -- but "should
be" is exactly the sort of claim worth checking exhaustively, so both
implementations are carried here and compared over the whole small input space.

Pure host arithmetic; no GPU required.
"""

import itertools

import pytest
import torch

from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder


def _reference(qo_len: torch.Tensor, max_qo_len: int, num_decode_tokens: int) -> int:
    """The pre-#58381 implementation, verbatim."""
    num_reqs = qo_len.numel()
    if num_reqs == 0 or num_decode_tokens <= 0:
        return 0
    if num_decode_tokens <= int(qo_len.sum().item()):
        return 0
    if num_decode_tokens % num_reqs != 0:
        return 0

    uniform_qo_len = num_decode_tokens // num_reqs
    if uniform_qo_len <= 1:
        return 0

    positive_qo_len = qo_len[qo_len > 0]
    if positive_qo_len.numel() == qo_len.numel():
        return 0
    if positive_qo_len.numel() > 0:
        if max_qo_len != uniform_qo_len:
            return 0
        if not torch.all(positive_qo_len == uniform_qo_len):
            return 0

    zero_positions = torch.nonzero(qo_len == 0, as_tuple=False).flatten()
    if zero_positions.numel() > 0:
        first_zero = int(zero_positions[0].item())
        if torch.any(qo_len[first_zero:] > 0):
            return 0

    return uniform_qo_len


_ACTUAL = AiterMLAMetadataBuilder._uniform_padded_mtp_qo_len


@pytest.mark.parametrize("num_reqs", [1, 2, 3, 4])
def test_matches_reference_exhaustively(num_reqs):
    """Every qo_len vector x decode-token count x max_qo_len in a small space."""
    checked = 0
    for lens in itertools.product(range(0, 4), repeat=num_reqs):
        qo = torch.tensor(lens, dtype=torch.int32)
        for num_decode_tokens in range(0, 13):
            for max_qo_len in range(0, 5):
                got = _ACTUAL(qo, max_qo_len, num_decode_tokens)
                want = _reference(qo, max_qo_len, num_decode_tokens)
                assert got == want, (
                    f"lens={lens} num_decode_tokens={num_decode_tokens} "
                    f"max_qo_len={max_qo_len}: got {got}, reference {want}"
                )
                checked += 1
    assert checked > 0


def test_empty_batch():
    assert _ACTUAL(torch.zeros(0, dtype=torch.int32), 4, 8) == 0
    assert _reference(torch.zeros(0, dtype=torch.int32), 4, 8) == 0


def test_the_padded_mtp_case_that_should_succeed():
    """A live prefix at the uniform width, then zero-padded dummies."""
    qo = torch.tensor([4, 4, 0, 0], dtype=torch.int32)
    # 4 requests x 4 tokens = 16 captured, 8 live -> dummies are synthesizable.
    assert _ACTUAL(qo, 4, 16) == 4
    assert _reference(qo, 4, 16) == 4


def test_positive_after_zero_is_rejected():
    """A live request after a padding hole breaks the contiguous-prefix layout."""
    qo = torch.tensor([4, 0, 4, 0], dtype=torch.int32)
    assert _ACTUAL(qo, 4, 16) == 0
    assert _reference(qo, 4, 16) == 0


def test_ragged_live_lengths_are_rejected():
    qo = torch.tensor([4, 2, 0, 0], dtype=torch.int32)
    assert _ACTUAL(qo, 4, 16) == 0
    assert _reference(qo, 4, 16) == 0
