# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence ids in prepare_chunk_indices when a sequence has no chunks."""

import torch

from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices


def test_prepare_chunk_indices_after_empty_sequence():
    # lengths 5, 0, 70 and chunk size 64 -> one chunk, none, then two.
    # The empty sequence owns no chunk, so the last two chunks stay on
    # sequence 2. Numbering chunks instead of sequences assigns them to 1.
    cu_seqlens = torch.tensor([0, 5, 5, 75], dtype=torch.int32)
    got = prepare_chunk_indices(cu_seqlens, 64)
    expected = torch.tensor([[0, 0], [2, 0], [2, 1]], dtype=torch.int32)
    assert torch.equal(got.cpu(), expected)


def test_prepare_chunk_indices_without_empty_sequence():
    cu_seqlens = torch.tensor([0, 5, 75], dtype=torch.int32)
    got = prepare_chunk_indices(cu_seqlens, 64)
    expected = torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int32)
    assert torch.equal(got.cpu(), expected)
