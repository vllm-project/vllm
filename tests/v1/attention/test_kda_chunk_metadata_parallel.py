# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 KDA chunk metadata: parallel grid must be value-identical.

The kernel builds the chunk index table the FLA scan walks. A wrong entry does
not crash -- the scan reads the wrong chunk and produces plausible garbage,
which surfaces as an accuracy regression far from the cause. So the bar is
bit-exact equality against the serial kernel, not "looks reasonable".

Both kernels are built from source in-process, so the test is independent of
whether the change is applied to the installed vllm.
"""

import pytest
import torch

from vllm.triton_utils import tl, triton

BT = 64
BLOCK_T = 256


@triton.jit(do_not_specialize=["N"])
def _serial(
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    N,
    BT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """The shipped kernel: one program per sequence, serial loop over chunks."""
    i_n = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    is_seq = offs_n < N
    bos = tl.load(cu_seqlens + offs_n, mask=is_seq, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens + offs_n + 1, mask=is_seq, other=0).to(tl.int32)
    nt = tl.where(is_seq, tl.cdiv(eos - bos, BT), 0)

    base = tl.sum(tl.where(offs_n < i_n, nt, 0))
    num_chunks = tl.sum(tl.where(offs_n == i_n, nt, 0))

    tl.store(chunk_offsets + i_n, base)
    if i_n == 0:
        tl.store(chunk_offsets + N, tl.sum(nt))

    for t0 in range(0, num_chunks, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < num_chunks
        row = (base + offs_t) * 2
        tl.store(chunk_indices + row, tl.full([BLOCK_T], i_n, tl.int32), mask=mask_t)
        tl.store(chunk_indices + row + 1, offs_t.to(tl.int32), mask=mask_t)


@triton.jit(do_not_specialize=["N"])
def _parallel(
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    N,
    BT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """The patched kernel: one program per (sequence, chunk block)."""
    i_n = tl.program_id(0)
    i_t = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    is_seq = offs_n < N
    bos = tl.load(cu_seqlens + offs_n, mask=is_seq, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens + offs_n + 1, mask=is_seq, other=0).to(tl.int32)
    nt = tl.where(is_seq, tl.cdiv(eos - bos, BT), 0)

    base = tl.sum(tl.where(offs_n < i_n, nt, 0))
    num_chunks = tl.sum(tl.where(offs_n == i_n, nt, 0))

    if i_t == 0:
        tl.store(chunk_offsets + i_n, base)
        if i_n == 0:
            tl.store(chunk_offsets + N, tl.sum(nt))

    t0 = i_t * BLOCK_T
    if t0 < num_chunks:
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < num_chunks
        row = (base + offs_t) * 2
        tl.store(chunk_indices + row, tl.full([BLOCK_T], i_n, tl.int32), mask=mask_t)
        tl.store(chunk_indices + row + 1, offs_t.to(tl.int32), mask=mask_t)


def _run(seq_lens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    dev = "cuda"
    n = len(seq_lens)
    cu = torch.zeros(n + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=dev).cumsum(0)

    chunks_per_seq = [(s + BT - 1) // BT for s in seq_lens]
    total = sum(chunks_per_seq)
    block_n = max(128, triton.next_power_of_2(n))

    out = []
    for parallel in (False, True):
        # Sentinel fill so any position left unwritten is detectable.
        idx = torch.full((total, 2), -7, dtype=torch.int32, device=dev)
        off = torch.full((n + 1,), -7, dtype=torch.int64, device=dev)
        if parallel:
            blocks = max(1, triton.cdiv(max(chunks_per_seq), BLOCK_T))
            _parallel[(n, blocks)](
                cu, idx, off, n, BT=BT, BLOCK_N=block_n, BLOCK_T=BLOCK_T
            )
        else:
            _serial[(n,)](cu, idx, off, n, BT=BT, BLOCK_N=block_n, BLOCK_T=BLOCK_T)
        torch.cuda.synchronize()
        out.append((idx, off))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "seq_lens",
    [
        # The case that motivates the change: one long prefill. At 194k tokens
        # and BT=64 this is ~3000 chunks walked by a single workgroup.
        [194_000],
        # Exactly one block of chunks, and one either side of the boundary.
        [BLOCK_T * BT],
        [BLOCK_T * BT + BT],
        [BLOCK_T * BT - BT],
        # Ragged multi-sequence: unequal lengths exercise the per-sequence
        # base offsets and the per-sequence chunk bound together.
        [10, 5000, 1, 33_000],
        # Lengths that are not multiples of the chunk size.
        [17, 4095, 100],
        # Single token.
        [1],
    ],
)
def test_chunk_metadata_parallel_is_value_identical(seq_lens: list[int]):
    (idx_s, off_s), (idx_p, off_p) = _run(seq_lens)

    assert not (idx_s == -7).any(), "serial left entries unwritten (test bug)"
    assert not (off_s == -7).any(), "serial left offsets unwritten (test bug)"
    assert not (idx_p == -7).any(), "parallel left entries unwritten"
    assert not (off_p == -7).any(), "parallel left offsets unwritten"

    torch.testing.assert_close(idx_p, idx_s, rtol=0, atol=0)
    torch.testing.assert_close(off_p, off_s, rtol=0, atol=0)
