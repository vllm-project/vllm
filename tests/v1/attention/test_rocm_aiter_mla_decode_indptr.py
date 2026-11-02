# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential test for the fused AITER MLA full-graph decode indptr build.

Under full CUDA graphs `_build_decode` used to fill the persistent kernel
lengths, `paged_kv_indptr` and `qo_indptr` buffers through ~20 eager ops. One
Triton program now does it. Every element of all three buffers -- including the
tails a captured graph reads past the live requests, and which a previous,
larger batch left stale -- must match the eager path bit for bit, so that path
is carried here verbatim as the reference.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER MLA tests", allow_module_level=True)

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (  # noqa: E402
    AiterMLAHelper,
    _build_decode_indptr_kernel,
)

DEVICE = torch.device("cuda:0")


def _eager(seq_lens, query_start_loc, bufs, num_reqs, max_qo_len, pad, varlen):
    """The pre-fusion full-graph path of `_build_decode`, verbatim."""
    graph_seq_lens, kv_indptr_buf, qo_indptr_buf = bufs[:3]
    seq_lens_for_kernel = seq_lens
    if pad:
        qo_lens = (query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]).to(
            torch.int32
        )
        seq_lens_for_kernel = torch.where(
            qo_lens > 0,
            seq_lens_for_kernel,
            seq_lens_for_kernel.new_full((), max_qo_len),
        )
    if varlen:
        seq_lens_for_kernel = seq_lens + (max_qo_len - qo_lens)

    graph_seq_lens[:num_reqs].copy_(seq_lens_for_kernel)
    seq_lens_for_kernel = graph_seq_lens[:num_reqs]
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            seq_lens_for_kernel.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    kv_indptr_buf[: 1 + num_reqs].copy_(kv_indptr)
    kv_indptr_buf[1 + num_reqs :].fill_(kv_indptr[-1])

    if pad:
        qo_indptr_src = AiterMLAHelper.qo_indptr_for_uniform_qlen(
            num_reqs, max_qo_len, DEVICE
        )
    else:
        qo_indptr_src = query_start_loc[: 1 + num_reqs]
    qo_indptr_buf[: 1 + num_reqs].copy_(qo_indptr_src)
    qo_indptr_buf[1 + num_reqs :] = qo_indptr_src[-1]

    if varlen:
        _eager_varlen_rows(
            query_start_loc, qo_lens, bufs[3], bufs[4], num_reqs, max_qo_len
        )


def _eager_varlen_rows(query_start_loc, qo_lens, q_rows_buf, o_rows, num_reqs, width):
    """`AiterMLAMetadataBuilder._fill_varlen_rows`, verbatim."""
    num_rows = num_reqs * width
    row_ids = torch.arange(q_rows_buf.numel(), dtype=torch.int64, device=DEVICE)
    row_req = (row_ids // width)[:num_rows]
    row_pos = (row_ids % width)[:num_rows]
    row_qlen = qo_lens.to(torch.int64).index_select(0, row_req)
    row_start = query_start_loc.to(torch.int64).index_select(0, row_req)
    is_real = row_pos < row_qlen
    packed_row = row_start + row_pos
    q_rows = q_rows_buf[:num_rows]
    q_rows.copy_(torch.where(is_real, packed_row, 0))
    o_rows.zero_()
    dump_slot = o_rows.size(0) - 1
    o_rows.scatter_(0, torch.where(is_real, packed_row, dump_slot), row_ids[:num_rows])


def _fused(seq_lens, query_start_loc, bufs, num_reqs, max_qo_len, pad, varlen):
    graph_seq_lens, kv_indptr_buf, qo_indptr_buf, q_rows, o_rows = bufs
    _build_decode_indptr_kernel[(1,)](
        seq_lens,
        query_start_loc,
        graph_seq_lens,
        kv_indptr_buf,
        qo_indptr_buf,
        num_reqs,
        kv_indptr_buf.shape[0],
        max_qo_len if pad else 0,
        q_rows,
        o_rows,
        o_rows.shape[0],
        PAD=pad,
        VARLEN=varlen,
        BLOCK=triton.next_power_of_2(graph_seq_lens.shape[0] + 1),
        ROW_W=max_qo_len,
        ROW_P2=triton.next_power_of_2(max_qo_len),
    )


def _stale_buffers(max_num_reqs, max_qo_len, gen):
    """Buffers holding what an earlier, larger batch would have left behind:
    kernel lengths, both indptrs, then the varlen q and o row maps."""
    rows = max_num_reqs * max_qo_len
    return [
        torch.randint(1, 1 << 20, (n,), generator=gen, dtype=dtype).to(DEVICE)
        for n, dtype in (
            (max_num_reqs, torch.int32),
            (max_num_reqs + 1, torch.int32),
            (max_num_reqs + 1, torch.int32),
            (rows, torch.int64),
            (rows + 1, torch.int64),
        )
    ]


def _batch(mode, num_reqs, max_qo_len, gen):
    if mode == "plain":
        qo_lens = torch.randint(1, max_qo_len + 1, (num_reqs,), generator=gen)
    elif mode == "pad":
        # A live prefix at the uniform width, then dummy rows with no queries.
        num_live = int(torch.randint(1, num_reqs + 1, (), generator=gen))
        qo_lens = torch.zeros(num_reqs, dtype=torch.int64)
        qo_lens[:num_live] = max_qo_len
    else:
        qo_lens = torch.randint(0, max_qo_len + 1, (num_reqs,), generator=gen)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = qo_lens.cumsum(0)
    seq_lens = torch.randint(1, 1 << 17, (num_reqs,), generator=gen)
    seq_lens = torch.where(qo_lens > 0, seq_lens, 0)
    return (
        seq_lens.to(dtype=torch.int32, device=DEVICE),
        query_start_loc.to(DEVICE),
    )


@pytest.mark.parametrize("mode", ["plain", "pad", "varlen"])
@pytest.mark.parametrize("max_num_reqs", [1, 7, 64, 1024])
@pytest.mark.parametrize("max_qo_len", [1, 4])
def test_fused_matches_eager(mode, max_num_reqs, max_qo_len):
    pad = mode != "plain"
    varlen = mode == "varlen"
    gen = torch.Generator().manual_seed(max_num_reqs * 10 + max_qo_len)
    for num_reqs in sorted({1, max(1, max_num_reqs // 3), max_num_reqs}):
        for _ in range(4):
            seq_lens, query_start_loc = _batch(mode, num_reqs, max_qo_len, gen)
            want = _stale_buffers(max_num_reqs, max_qo_len, gen)
            got = [b.clone() for b in want]
            args = (seq_lens, query_start_loc)
            _eager(*args, want, num_reqs, max_qo_len, pad, varlen)
            _fused(*args, got, num_reqs, max_qo_len, pad, varlen)
            # The o row map's last entry is the dump slot: written by every
            # padding slot in an unspecified order and never read.
            got[4], want[4] = got[4][:-1], want[4][:-1]
            for name, g, w in zip(
                ("graph_seq_lens", "paged_kv_indptr", "qo_indptr", "q_rows", "o_rows"),
                got,
                want,
            ):
                assert torch.equal(g, w), (
                    f"{name} differs: mode={mode} num_reqs={num_reqs} "
                    f"max_num_reqs={max_num_reqs} max_qo_len={max_qo_len}\n"
                    f"fused={g.tolist()}\neager={w.tolist()}"
                )
