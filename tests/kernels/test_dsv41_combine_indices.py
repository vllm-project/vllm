# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""combine_topk_swa_indices must leave -1 sentinels past ``combined_lens``.

The DeepSeek V4.1 prefill paths hand the kernel a reused workspace through
``out=``; stale valid-looking indices there are ignored by
``flash_mla_sparse_fwd`` (it caps at ``topk_length``) but the FlashMLA fused
prefill still reads them and propagates NaN from whatever row they name.
"""

import pytest
import torch

from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
    combine_topk_swa_indices,
)

WINDOW = 128
STALE = 12345


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compress_ratio,topk", [(1, 512), (2, 512), (0, 0)])
def test_reused_out_buffer_matches_fresh_result(compress_ratio: int, topk: int):
    device = torch.device("cuda")
    seq_lens = [700, 40, 1300]
    query_lens = [700, 40, 300]
    query_start_loc = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(query_lens, device=device).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    gather_lens = torch.tensor(
        [q + min(s - q, WINDOW - 1) for s, q in zip(seq_lens, query_lens)],
        dtype=torch.int32,
        device=device,
    )
    ratio = max(compress_ratio, 1)
    n = (max(seq_lens) + ratio - 1) // ratio
    m = n + int(gather_lens.max()) + 8
    gen = torch.Generator(device="cpu").manual_seed(0)
    topk_indices = torch.randint(
        0, n, (num_tokens, max(topk, 1)), generator=gen, dtype=torch.int32
    ).to(device)

    def run(out):
        return combine_topk_swa_indices(
            topk_indices,
            query_start_loc,
            seq_lens_t,
            gather_lens,
            WINDOW,
            compress_ratio,
            topk,
            m,
            n,
            out=out,
        )

    fresh_indices, fresh_lens = run(None)
    assert fresh_indices.shape[1] % 128 == 0
    stale_indices = torch.full_like(fresh_indices, STALE)
    stale_lens = torch.full_like(fresh_lens, STALE)
    reused_indices, reused_lens = run((stale_indices, stale_lens))

    torch.testing.assert_close(reused_lens, fresh_lens)
    torch.testing.assert_close(reused_indices, fresh_indices)
    col = torch.arange(fresh_indices.shape[1], device=device).view(1, -1)
    tail = col >= fresh_lens.view(-1, 1)
    assert tail.any(), "case must leave unused slots to exercise the sentinel fill"
    assert (reused_indices[tail] == -1).all()
    assert (reused_indices[~tail] != STALE).all()
