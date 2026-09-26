# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the AITER MLA page-index expansion kernel.

`_expand_page_indices_kernel` turns a block table into the per-token flat page
indices the AITER MLA decode kernel consumes (it always runs with page_size=1
internally, so `kv_buffer` is flattened and each token needs its own index).

The kernel was parallelised over token chunks as well as requests; at
concurrency 1 the previous `(num_reqs,)` grid put a single workgroup in a serial
loop over the whole sequence. These tests pin the result to a straightforward
PyTorch expansion so the parallelisation cannot change what is produced.

Both block-table strides are passed by the caller, so a non-contiguous view is
covered explicitly rather than assumed away.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    _expand_page_indices_kernel,
)
from vllm.triton_utils import triton


def _reference(block_table, cu_num_tokens, kernel_block_size, out_len, device):
    """Per-token flat index, written the obvious way."""
    out = torch.full((out_len,), -1, dtype=torch.int64, device=device)
    num_reqs = cu_num_tokens.numel() - 1
    for r in range(num_reqs):
        start = int(cu_num_tokens[r])
        end = int(cu_num_tokens[r + 1])
        n = end - start
        if n <= 0:
            continue
        tok = torch.arange(n, device=device)
        blk = tok // kernel_block_size
        off = tok % kernel_block_size
        out[start:end] = block_table[r, blk].to(torch.int64) * kernel_block_size + off
    return out


def _run(block_table, cu_num_tokens, kernel_block_size, out_len, device):
    out = torch.full((out_len,), -1, dtype=torch.int64, device=device)
    num_reqs = cu_num_tokens.numel() - 1
    max_tokens_per_req = block_table.shape[1] * kernel_block_size
    num_chunks = max(1, -(-max_tokens_per_req // 1024))
    _expand_page_indices_kernel[(num_reqs, num_chunks)](
        out,
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        cu_num_tokens,
        KERNEL_BLOCK_SIZE=kernel_block_size,
        BLOCK_SIZE=1024,
    )
    return out


def _make(seq_lens, kernel_block_size, device, noncontig=False):
    num_reqs = len(seq_lens)
    max_blocks = max(1, max((s + kernel_block_size - 1) // kernel_block_size
                            for s in seq_lens)) + 2
    width = max_blocks * 2 if noncontig else max_blocks
    torch.manual_seed(0)
    bt = torch.randint(0, 10_000, (num_reqs, width), dtype=torch.int64, device=device)
    if noncontig:
        bt = bt[:, ::2]
        assert not bt.is_contiguous()
    cu = torch.zeros(num_reqs + 1, dtype=torch.int64, device=device)
    cu[1:] = torch.tensor(seq_lens, dtype=torch.int64, device=device).cumsum(0)
    return bt, cu, int(cu[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("seq_lens", [[1], [128], [4096], [1, 1], [4096, 7],
                                      [37, 4096, 1], [1024] * 8])
@pytest.mark.parametrize("kernel_block_size", [1, 16, 64])
def test_matches_reference(seq_lens, kernel_block_size):
    bt, cu, total = _make(seq_lens, kernel_block_size, "cuda")
    got = _run(bt, cu, kernel_block_size, total, "cuda")
    want = _reference(bt, cu, kernel_block_size, total, "cuda")
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("kernel_block_size", [1, 16])
def test_non_contiguous_block_table(kernel_block_size):
    """Column stride != 1 must be honoured, not assumed away."""
    bt, cu, total = _make([4096, 37], kernel_block_size, "cuda", noncontig=True)
    got = _run(bt, cu, kernel_block_size, total, "cuda")
    want = _reference(bt, cu, kernel_block_size, total, "cuda")
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_ragged_batch_leaves_other_rows_untouched():
    """A short request must not have its tail written by a longer one's chunks."""
    kbs = 16
    bt, cu, total = _make([4096, 1, 512], kbs, "cuda")
    out = torch.full((total + 8,), -999, dtype=torch.int64, device="cuda")
    num_chunks = max(1, -(-(bt.shape[1] * kbs) // 1024))
    _expand_page_indices_kernel[(3, num_chunks)](
        out, bt, bt.stride(0), bt.stride(1), cu,
        KERNEL_BLOCK_SIZE=kbs, BLOCK_SIZE=1024,
    )
    want = _reference(bt, cu, kbs, total, "cuda")
    torch.testing.assert_close(out[:total], want, atol=0, rtol=0)
    # The slack past the last request must be untouched.
    assert torch.all(out[total:] == -999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_zero_length_request_writes_nothing():
    """A zero-token request in a padded batch must produce no stores."""
    kbs = 16
    bt, cu, total = _make([256, 0, 128], kbs, "cuda")
    got = _run(bt, cu, kbs, total, "cuda")
    want = _reference(bt, cu, kbs, total, "cuda")
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert int(cu[2] - cu[1]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_over_provisioned_chunks_do_not_change_result():
    """Extra chunks must be inert.

    Each program covers `chunk_idx * BLOCK_SIZE + [0, BLOCK_SIZE)`, so the grid
    has to span the request: 8192 tokens need at least ceil(8192/1024) == 8
    chunks, and fewer would simply leave the tail unwritten. The caller sizes
    the grid from `block_table.shape[1] * kernel_block_size`, an upper bound
    over the batch, so in a ragged batch most requests get more chunks than
    they need. That over-provisioning is the case worth pinning: the surplus
    programs must return early and change nothing.
    """
    kbs = 1
    bt, cu, total = _make([8192], kbs, "cuda")
    ref = _reference(bt, cu, kbs, total, "cuda")
    required = -(-total // 1024)
    for num_chunks in (required, required * 2, required * 8):
        out = torch.full((total,), -1, dtype=torch.int64, device="cuda")
        _expand_page_indices_kernel[(1, num_chunks)](
            out, bt, bt.stride(0), bt.stride(1), cu,
            KERNEL_BLOCK_SIZE=kbs, BLOCK_SIZE=1024,
        )
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
