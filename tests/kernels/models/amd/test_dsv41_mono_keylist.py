# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""keylist.launch, the Triton launch that writes K2's key lists on a gfx942
index layer of the DeepSeek-V4.1 mono decode, against a reference built in
torch from its documented format. Triton, so it runs on every ROCm GPU."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the mono decode is for ROCm"
)


def _keylist_reference(slot, pos, swa_idx, swa_lens, t2r, ratio, topk, bt, cb):
    from vllm.models.deepseek_v41.amd.mono.attention.plan import KEYS, TOPK

    S = slot.shape[0]
    kt = torch.full((S, KEYS), -1, dtype=torch.int64)
    klen = torch.zeros(2 * S, dtype=torch.int32)
    for t in range(S):
        live = int(slot[t]) >= 0
        nswa = int(swa_lens[t]) if live else 0
        ntopk = min((int(pos[t]) + 1) // ratio, TOPK) if live and ratio else 0
        for k in range(ntopk):
            local = int(topk[t, k])
            if local >= 0:
                # The kernel sets bit 31 of the int32 slot. For a slot below
                # 2^31 that is the slot minus 2^31.
                blk = int(bt[int(t2r[t]), local // cb])
                kt[t, k] = blk * cb + local % cb - (1 << 31)
        kt[t, ntopk : ntopk + nswa] = swa_idx[t, 0, :nswa].long()
        klen[t], klen[S + t] = ntopk + nswa, ntopk
    return kt.to(torch.int32).view(-1), klen


@pytest.mark.parametrize("ratio", [0, 1, 2, 4])
def test_keylist_matches_torch_reference(ratio):
    from vllm.models.deepseek_v41.amd.mono import keylist
    from vllm.models.deepseek_v41.amd.mono.attention.plan import KEYS, TOPK, WINDOW

    g = torch.Generator().manual_seed(ratio)
    S, reqs, cb, blocks = 6, 2, 64, 3000
    # Short and long positions, and a padded token with slot -1.
    pos = torch.tensor([3, 40, 700, 131071, 131072, 2047], dtype=torch.int64)
    slot = torch.randint(0, 1 << 30, (S,), generator=g, dtype=torch.int64)
    slot[2] = -1
    swa_lens = torch.minimum(pos + 1, torch.tensor(WINDOW)).to(torch.int32)
    swa_idx = torch.randint(0, 1 << 24, (S, 1, WINDOW), generator=g, dtype=torch.int32)
    t2r = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    topk = torch.full((S, TOPK), -1, dtype=torch.int32)
    for t in range(S):
        n = min((int(pos[t]) + 1) // max(ratio, 1), TOPK)
        topk[t, :n] = torch.randperm(max(n, (int(pos[t]) + 1) // max(ratio, 1)))[:n]
    topk[4, 10:20] = -1
    bt = torch.randint(0, 1 << 20, (reqs, blocks), generator=g, dtype=torch.int32)
    kt = torch.full((S * KEYS,), -7, dtype=torch.int32, device="cuda")
    klen = torch.full((2 * S,), -7, dtype=torch.int32, device="cuda")
    keylist.launch(
        kt,
        klen,
        slot.cuda(),
        pos.cuda(),
        swa_idx.cuda(),
        swa_lens.cuda(),
        t2r.cuda(),
        ratio,
        topk.cuda() if ratio else None,
        bt.cuda() if ratio else None,
        cb,
    )
    torch.accelerator.synchronize()
    ref_kt, ref_klen = _keylist_reference(
        slot, pos, swa_idx, swa_lens, t2r, ratio, topk, bt, cb
    )
    assert torch.equal(klen.cpu(), ref_klen)
    assert torch.equal(kt.cpu(), ref_kt)
