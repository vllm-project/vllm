# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 decode attention metadata of DeepSeek-V4.1 against the vLLM ops
that it replaces.

topk_pack.launch replaces compute_global_topk_ragged_indices_and_indptr. For
each random decode batch vLLM's ops run first. Then every output buffer is
overwritten with -7, the fused launch runs, and every output that a reader
uses must be equal. swa_decode_meta, which also runs on gfx950, is tested in
tests/models/test_dsv41_mono_numerics.py.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the fused launches are for ROCm"
)

MAX_SEQS = 128
TOPK = 512


def _topk_case(
    g, max_rows
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """A decode batch of top-512 rows as the indexer writes them: valid local
    indices first and -1 after them, some rows shorter than 512, some rows
    with a -1 inside the valid entries, and padded tokens."""
    n = int(torch.randint(1, max_rows + 1, (1,), generator=g))
    num_reqs = int(torch.randint(1, n + 1, (1,), generator=g))
    block_size = (32, 64, 128)[int(torch.randint(0, 3, (1,), generator=g))]
    comp_len = int(torch.randint(1, 70000, (1,), generator=g))
    blocks = (comp_len + block_size - 1) // block_size
    block_table = torch.randint(
        0, 300000, (MAX_SEQS, blocks + 1), generator=g, dtype=torch.int32
    )
    topk = torch.full((n, TOPK), -1, dtype=torch.int32)
    for t in range(n):
        valid = min(TOPK, int(torch.randint(0, comp_len + 1, (1,), generator=g)))
        if float(torch.rand(1, generator=g)) < 0.5:
            valid = min(TOPK, comp_len)
        topk[t, :valid] = torch.randperm(comp_len, generator=g)[:valid].to(torch.int32)
        if valid > 4 and float(torch.rand(1, generator=g)) < 0.2:
            topk[t, int(torch.randint(0, valid, (1,), generator=g))] = -1
    t2r = torch.sort(torch.randint(0, num_reqs, (n,), generator=g)).values
    is_valid = torch.rand(n, generator=g) > 0.15
    return (
        topk.cuda(),
        t2r.to(torch.int32).cuda(),
        block_table.cuda(),
        block_size,
        is_valid.cuda(),
    )


def test_topk_pack_matches_vllm():
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.models.deepseek_v41.amd.ops import topk_pack

    g = torch.Generator().manual_seed(4321)
    failed = []
    for i in range(500):
        case = _topk_case(g, topk_pack.MAX_ROWS)
        ref = rocm._compute_global_topk_ragged_indices_and_indptr_old(*case)
        out = topk_pack.launch(*case, poison=True)
        nnz = int(ref[1][-1])
        bad = [
            name
            for name, a, b in (
                ("topk_lens", out[2], ref[2]),
                ("topk_indptr", out[1], ref[1]),
                ("global_topk_ragged", out[0][:nnz], ref[0][:nnz]),
            )
            if a.shape != b.shape or not torch.equal(a, b)
        ]
        if out[0].shape != ref[0].shape:
            bad.append("ragged shape")
        if bad:
            failed.append((i, case[0].shape[0], bad))
    assert not failed, f"batches that differ (case, rows, outputs): {failed}"
