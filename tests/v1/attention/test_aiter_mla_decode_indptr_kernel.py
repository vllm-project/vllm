# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AITER MLA full-graph decode indptr kernel against the eager ops it
replaces. The kernel is plain Triton, so this runs on any GPU platform."""

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    _build_decode_indptr_kernel,
)

_STALE = 12345


def _eager_reference(seq_lens, qo_lens, query_start_loc, pad, varlen, buf_len, w):
    """The eager full-graph build in AiterMLAMetadataBuilder._build_decode."""
    n = seq_lens.numel()
    seq = seq_lens.clone()
    if varlen:
        seq = seq + (pad - qo_lens)
    elif pad:
        seq = torch.where(qo_lens > 0, seq, seq.new_full((), pad))
    kv = torch.cat([seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32)])
    kv_indptr = kv.new_full((buf_len,), int(kv[-1]))
    kv_indptr[: n + 1] = kv
    if pad:
        qo = torch.arange(n + 1, dtype=torch.int32, device=seq.device) * pad
    else:
        qo = query_start_loc
    qo_indptr = qo.new_full((buf_len,), int(qo[-1]))
    qo_indptr[: n + 1] = qo
    q_rows: torch.Tensor | None = None
    o_rows: torch.Tensor | None = None
    if varlen:
        rows = torch.arange(n * w, device=seq.device)
        req, pos = rows // w, rows % w
        is_real = pos < qo_lens.to(torch.int64)[req]
        packed = query_start_loc.to(torch.int64)[req] + pos
        q_rows = torch.where(is_real, packed, 0)
        o_rows = torch.zeros(
            (buf_len - 1) * w + 1, dtype=torch.int64, device=seq.device
        )
        o_rows.scatter_(0, torch.where(is_real, packed, o_rows.numel() - 1), rows)
    return seq, kv_indptr, qo_indptr, q_rows, o_rows


def _cases():
    gen = torch.Generator().manual_seed(0)
    cases = [
        # No speculative decoding: one query token per request, no padding.
        (8, [1], 0, False),
        (8, [1] * 5, 0, False),
        (8, [1] * 8, 0, False),
        # Ragged query lengths without uniform padding.
        (8, [1, 3, 2, 4], 0, False),
    ]
    for w in (1, 3, 4, 7):
        cases += [(8, [w, w, 0, 0], w, False), (8, [w] * 8, w, False)]
    for w in (3, 4, 7):
        cases.append((8, [w - 1, 1, w, 0], w, True))
    for max_reqs in (64, 1024):
        for n in (1, max_reqs // 2, max_reqs):
            live = max(1, n // 2)
            mixed = torch.randint(0, 5, (n,), generator=gen)
            mixed[0] = 4
            cases += [
                (max_reqs, [1] * n, 0, False),
                (max_reqs, [4] * live + [0] * (n - live), 4, False),
                (max_reqs, mixed.tolist(), 4, True),
            ]
    return cases


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for Triton")
@pytest.mark.parametrize("max_reqs,qo_lens,pad,varlen", _cases())
def test_decode_indptr_kernel_matches_eager(max_reqs, qo_lens, pad, varlen):
    """Lengths, both indptrs (tails included) and the varlen row maps match the
    eager build bit for bit; buffer entries past the batch start out stale."""
    device = torch.device("cuda")
    n = len(qo_lens)
    w = pad if varlen else 1
    qo_lens_t = torch.tensor(qo_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(n + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = qo_lens_t.cumsum(0)
    gen = torch.Generator().manual_seed(n)
    seq_lens = torch.randint(1, 5000, (n,), generator=gen, dtype=torch.int32)
    seq_lens = torch.where(qo_lens_t.cpu() > 0, seq_lens, 0).to(device)

    buf_len = max_reqs + 1
    graph_seq_lens = torch.full((max_reqs,), _STALE, dtype=torch.int32, device=device)
    kv_indptr = torch.full((buf_len,), _STALE, dtype=torch.int32, device=device)
    qo_indptr = kv_indptr.clone()
    q_rows = torch.full((max_reqs * w,), _STALE, dtype=torch.int64, device=device)
    o_rows = torch.full((max_reqs * w + 1,), _STALE, dtype=torch.int64, device=device)
    rows = (q_rows, o_rows) if varlen else (qo_indptr, qo_indptr)
    _build_decode_indptr_kernel[(1,)](
        seq_lens,
        query_start_loc,
        graph_seq_lens,
        kv_indptr,
        qo_indptr,
        n,
        buf_len,
        pad,
        rows[0],
        rows[1],
        rows[1].shape[0],
        PAD=pad > 0,
        VARLEN=varlen,
        BLOCK=triton.next_power_of_2(buf_len),
        ROW_W=w,
        ROW_P2=triton.next_power_of_2(w),
    )

    seq, want_kv, want_qo, want_q, want_o = _eager_reference(
        seq_lens, qo_lens_t, query_start_loc, pad, varlen, buf_len, w
    )
    assert torch.equal(graph_seq_lens[:n], seq)
    assert torch.all(graph_seq_lens[n:] == _STALE)
    assert torch.equal(kv_indptr, want_kv)
    assert torch.equal(qo_indptr, want_qo)
    if varlen:
        assert torch.equal(q_rows[: n * w], want_q)
        assert torch.all(q_rows[n * w :] == _STALE)
        # The last entry is the dump slot padding rows scatter into; nothing
        # reads it.
        assert torch.equal(o_rows[:-1], want_o[:-1])
