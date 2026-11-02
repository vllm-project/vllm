# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The in-place ``expand_pools_and_append_tail`` call (int32 pool ids, int64
positions, ``seq_len_offset=1``, ``out=`` a full-width buffer row slice) writes
the same rows as the old path (int64 ids, ``positions.to(int32) + 1``, copy into
a ``-1``-filled buffer)."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("AMD kpool ops", allow_module_level=True)

from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops  # noqa: E402


@pytest.mark.parametrize("rows", [1, 16, 64])
@pytest.mark.parametrize("pool_size", [4])
def test_inplace_matches_copy_path(rows: int, pool_size: int):
    torch.manual_seed(rows)
    n_groups = 512
    # topk_indices_buffer rows are wider than topk + pool_size - 1
    width = n_groups * pool_size + 128
    positions = torch.randint(0, 60000, (rows,), dtype=torch.int64, device="cuda")
    seq_lens = positions + 1
    pools = (seq_lens // pool_size).clamp(min=1)
    rand = torch.rand(rows, n_groups, device="cuda")
    pool_ids = (rand * pools[:, None]).to(torch.int32)
    pool_ids[rand < 0.15] = -1  # unused top-k slots

    ref_rows = kpool_ops.expand_pools_and_append_tail(
        pool_ids.to(torch.int64), seq_lens.to(torch.int32), pool_size
    )
    ref = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    ref[:, : ref_rows.shape[1]] = ref_rows

    buf = torch.full((rows + 3, width), 7, dtype=torch.int32, device="cuda")
    kpool_ops.expand_pools_and_append_tail(
        pool_ids, positions, pool_size, seq_len_offset=1, out=buf[:rows]
    )

    torch.testing.assert_close(buf[:rows], ref, rtol=0, atol=0)
    assert (buf[rows:] == 7).all()  # rows past the slice are untouched
