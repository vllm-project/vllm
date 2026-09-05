# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_utils import (
    MAX_TOPOLOGY_SEGMENTS,
    apply_topology_witnesses,
    scatter_topology_witnesses_,
    topology_witness_indices,
)


def _segment_of(token: int, context_len: int, num_segments: int) -> int:
    return (token * min(num_segments, context_len)) // context_len


def _random_case(generator, device, num_segments=None):
    rows = int(torch.randint(1, 6, (1,), generator=generator))
    topk = int(torch.randint(1, 40, (1,), generator=generator))
    segments = num_segments or int(torch.randint(1, 65, (1,), generator=generator))
    context = int(torch.randint(1, 200, (1,), generator=generator))
    lens = torch.randint(0, context + 1, (rows,), generator=generator).int()
    # Real top-k emits distinct in-range offsets padded with -1; duplicates in
    # the input would make the no-duplicate contract untestable.
    learned = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        row_len = int(lens[row])
        if row_len:
            picked = torch.randperm(row_len, generator=generator)[:topk]
            learned[row, : picked.numel()] = picked.int()
    return lens.to(device), learned.to(device), topk, segments


def test_witness_fills_only_the_segment_the_learned_row_misses():
    # Context 16 split into 4 segments starts them at 0, 4, 8, 12. The learned
    # row occupies segments 0, 1 and 2, so only segment 3 earns a witness.
    learned = torch.tensor([[0, 5, 9, -1]], dtype=torch.int32)
    context_lens = torch.tensor([16], dtype=torch.int32)

    witnesses = topology_witness_indices(context_lens, learned, 4)

    assert witnesses.dtype == torch.int32
    assert witnesses[0].tolist() == [-1, -1, -1, 12]


def test_merge_preserves_prefix_and_bounds_replacement():
    learned = torch.tensor([[0, 5, 9, -1]], dtype=torch.int32)
    context_lens = torch.tensor([16], dtype=torch.int32)

    merged = apply_topology_witnesses(
        learned, context_lens, learned_keep=2, num_segments=4, max_replacements=2
    )

    assert merged[0].tolist() == [0, 5, 12, -1]


def test_zero_length_row_gets_no_witness():
    # A prefill chunk with nothing gathered must stay all -1; a witness at token
    # 0 would point the sparse kernel at a token the row cannot attend to.
    learned = torch.full((1, 4), -1, dtype=torch.int32)
    context_lens = torch.tensor([0], dtype=torch.int32)

    merged = apply_topology_witnesses(
        learned, context_lens, learned_keep=0, num_segments=4, max_replacements=4
    )

    assert merged[0].tolist() == [-1, -1, -1, -1]


def test_zero_budget_is_a_passthrough():
    learned = torch.tensor([[0, 5, 9, 11]], dtype=torch.int32)
    context_lens = torch.tensor([16], dtype=torch.int32)

    merged = apply_topology_witnesses(
        learned, context_lens, learned_keep=2, num_segments=4, max_replacements=0
    )

    assert torch.equal(merged, learned)


def test_scatter_replaces_the_tail_in_place():
    learned = torch.tensor([[0, 5, 9, -1]], dtype=torch.int32)
    context_lens = torch.tensor([16], dtype=torch.int32)

    scatter_topology_witnesses_(learned, context_lens, num_segments=1)

    # One segment spans the whole context and the learned row already covers it,
    # so the reserved slot keeps its -1 rather than repeating a token.
    assert learned[0, :3].tolist() == [0, 5, 9]


@pytest.mark.parametrize("num_segments", [1, 3, 8, MAX_TOPOLOGY_SEGMENTS])
def test_witness_contracts_hold_over_random_rows(num_segments):
    """Causal, distinct, and disjoint from the learned row it augments.

    A witness at or past the context length points attention at a token the
    query cannot see; a repeated one doubles that token's softmax weight.
    """
    generator = torch.Generator().manual_seed(num_segments)
    for _ in range(50):
        lens, learned, _, _ = _random_case(generator, "cpu", num_segments)
        witnesses = topology_witness_indices(lens, learned, num_segments)

        for row in range(learned.shape[0]):
            row_len = int(lens[row])
            emitted = [int(x) for x in witnesses[row].tolist() if x >= 0]
            learned_row = {int(x) for x in learned[row].tolist() if x >= 0}

            assert all(0 <= token < row_len for token in emitted)
            assert len(set(emitted)) == len(emitted)
            assert not learned_row & set(emitted)

            if row_len:
                covered = {_segment_of(x, row_len, num_segments) for x in learned_row}
                covered |= {_segment_of(x, row_len, num_segments) for x in emitted}
                assert covered == set(range(min(num_segments, row_len)))


def test_merge_never_introduces_a_duplicate():
    generator = torch.Generator().manual_seed(5)
    for _ in range(50):
        lens, learned, topk, segments = _random_case(generator, "cpu")
        keep = max(0, topk - segments)
        merged = apply_topology_witnesses(learned, lens, keep, segments, segments)

        for row in range(learned.shape[0]):
            assert merged[row, :keep].tolist() == learned[row, :keep].tolist()
            tokens = [int(x) for x in merged[row].tolist() if x >= 0]
            assert len(set(tokens)) == len(tokens)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_segments": 0},
        {"num_segments": MAX_TOPOLOGY_SEGMENTS + 1},
        {"learned_keep": -1},
        {"max_replacements": -1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    learned = torch.zeros((2, 4), dtype=torch.int32)
    context_lens = torch.tensor([8, 8], dtype=torch.int32)
    call = {
        "learned_keep": 2,
        "num_segments": 4,
        "max_replacements": 2,
        **kwargs,
    }

    with pytest.raises(ValueError):
        apply_topology_witnesses(learned, context_lens, **call)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_kernel_matches_the_torch_reference():
    generator = torch.Generator().manual_seed(11)
    for _ in range(50):
        lens, learned, topk, segments = _random_case(generator, "cpu")
        keep = max(0, topk - segments)
        expected = apply_topology_witnesses(learned, lens, keep, segments, segments)
        actual = apply_topology_witnesses(
            learned.cuda(), lens.cuda(), keep, segments, segments
        )
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_kernel_is_deterministic():
    generator = torch.Generator().manual_seed(7)
    lens = torch.randint(1, 4096, (256,), generator=generator).int().cuda()
    learned = torch.randint(-1, 4096, (256, 2048), generator=generator).int().cuda()

    first = apply_topology_witnesses(learned, lens, 1984, 64, 64)
    for _ in range(4):
        assert torch.equal(apply_topology_witnesses(learned, lens, 1984, 64, 64), first)
