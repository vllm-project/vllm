# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side metadata construction for Mamba2 exact-replay mode."""

import pytest
import torch

from vllm.v1.attention.backends.mamba2_attn import build_exact_replay_metadata

CHUNK = 256
DEVICE = torch.device("cpu")


def _build(num_computed, query_lens):
    return build_exact_replay_metadata(num_computed, query_lens, CHUNK, DEVICE)


def test_decode_rows_mid_chunk_at_boundary_and_completing_chunk():
    # three decode rows: 300 computed (44 tokens into a chunk), exactly at a
    # boundary, and one token short of completing a chunk
    m = _build([300, 512, 767], [1, 1, 1])
    assert m.num_aug_tokens == 44 + 1 + 1 + 255 + 1
    assert m.cu_seqlens.tolist() == [0, 45, 46, 302]
    # every augmented sequence is a single chunk
    assert m.cu_chunk_seqlens.tolist() == [0, 45, 46, 302]
    assert m.last_chunk_indices.tolist() == [0, 1, 2]
    assert m.seq_idx.tolist() == [0, 1, 2]
    assert m.has_boundary_state.tolist() == [True, True, True]
    # only the third row fills its chunk this step
    assert m.boundary_rows.tolist() == [2]
    assert m.boundary_chunk_idx.tolist() == [2]
    # buffered tokens: 44 of row 0, none of row 1, 255 of row 2
    assert m.buffered_seq.tolist() == [0] * 44 + [2] * 255
    assert m.buffered_pos.tolist() == list(range(44)) + list(range(255))
    assert m.buffered_dst.tolist() == list(range(44)) + list(range(46, 301))
    assert m.step_dst.tolist() == [44, 45, 301]
    # rows 0 and 1 append their token to the buffer; row 2 completed a chunk
    assert m.store_src.tolist() == [44, 45]
    assert m.store_seq.tolist() == [0, 1]
    assert m.store_pos.tolist() == [44, 0]


def test_prefill_rows_fresh_and_resumed_at_unaligned_positions():
    # fresh 600-token prefill; resume at 300 with 213 tokens; resume at 100
    # with 413 tokens
    m = _build([0, 300, 100], [600, 213, 413])
    assert m.num_aug_tokens == 600 + (44 + 213) + (100 + 413)
    assert m.cu_seqlens.tolist() == [0, 600, 857, 1370]
    assert m.cu_chunk_seqlens.tolist() == [0, 256, 512, 600, 856, 857, 1113, 1369, 1370]
    assert m.last_chunk_indices.tolist() == [2, 4, 7]
    assert m.seq_idx.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    # row 1 resumes at 300 -> boundary 256 holds a state; the others start
    # from zero (row 2's boundary is 0)
    assert m.has_boundary_state.tolist() == [False, True, False]
    # last completed chunks: row 0 -> [256,512) (chunk 1); row 1 -> the chunk
    # ending at 512 (chunk 3); row 2 -> the chunk ending at 512 (chunk 6)
    assert m.boundary_rows.tolist() == [0, 1, 2]
    assert m.boundary_chunk_idx.tolist() == [1, 3, 6]
    # buffered tokens re-fed for rows 1 and 2
    assert m.buffered_seq.tolist() == [1] * 44 + [2] * 100
    assert m.buffered_dst.tolist() == list(range(600, 644)) + list(range(857, 957))
    # trailing partial chunks stored back: 88 tokens of row 0, 1 of row 1,
    # 1 of row 2, all at buffer positions starting from 0
    assert m.store_src.tolist() == list(range(512, 600)) + [856] + [1369]
    assert m.store_seq.tolist() == [0] * 88 + [1] + [2]
    assert m.store_pos.tolist() == list(range(88)) + [0] + [0]


@pytest.mark.parametrize("seed", range(20))
def test_augmented_layout_invariants(seed):
    gen = torch.Generator().manual_seed(seed)
    n = int(torch.randint(1, 6, (1,), generator=gen))
    num_computed = torch.randint(0, 5 * CHUNK, (n,), generator=gen).tolist()
    query_lens = torch.randint(1, 3 * CHUNK, (n,), generator=gen).tolist()
    # decode rows are the common case: make about half of them single-token
    for i in range(n):
        if torch.rand(1, generator=gen).item() < 0.5:
            query_lens[i] = 1
    m = _build(num_computed, query_lens)

    # buffered and step destinations partition the augmented layout
    dst = torch.cat([m.buffered_dst, m.step_dst])
    assert sorted(dst.tolist()) == list(range(m.num_aug_tokens))
    assert m.step_dst.numel() == sum(query_lens)
    # chunks tile the layout and never span sequences or exceed chunk_size
    chunks = m.cu_chunk_seqlens.tolist()
    assert chunks[0] == 0 and chunks[-1] == m.num_aug_tokens
    for a, b in zip(chunks, chunks[1:]):
        assert 0 < b - a <= CHUNK
    cu = m.cu_seqlens.tolist()
    for i, (lo, hi) in enumerate(zip(cu, cu[1:])):
        own = [c for c in range(len(chunks) - 1) if lo <= chunks[c] < hi]
        assert m.seq_idx.tolist()[own[0] : own[-1] + 1] == [i] * len(own)
        assert m.last_chunk_indices.tolist()[i] == own[-1]
        # chunks of a sequence start at its start plus multiples of chunk_size
        assert all((chunks[c] - lo) % CHUNK == 0 for c in own)
    # stored positions stay inside the buffer; every store goes to its own row
    assert (m.store_pos < CHUNK).all()
    assert (m.buffered_pos < CHUNK).all()
    # the metadata names batch rows only, never state slots
    assert (m.store_seq < n).all() and (m.buffered_seq < n).all()
    for i, (nc, q) in enumerate(zip(num_computed, query_lens)):
        n_pre = nc % CHUNK
        aug = n_pre + q
        if aug // CHUNK == 0:
            # nothing completed: this step's tokens are appended after n_pre
            mask = m.store_seq == i
            assert m.store_pos[mask].tolist() == list(range(n_pre, aug))
        else:
            assert i in m.boundary_rows.tolist()
