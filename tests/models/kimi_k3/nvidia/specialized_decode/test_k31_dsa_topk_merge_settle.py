# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DCP top-k exchange merge's "digits" round with MERGE_SETTLE_ALL (a lane
re-polls all of its pairs in every pass) against the pair-by-pair settle, on one
GPU: 16 simulated ranks push their local top-N B' rows (IndexerLocalTopk) into
the exchange slots, all before the merge or three of them late from a side
stream. The rows, counts, saturation words and epochs match bit for bit."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the DCP top-k merge kernels need SM100",
)

ROWS, TOPK, WORLD, N, PAGE = 8, 2048, 16, 192, 64
MAX_MODEL_LEN = 600_000
LATE = (3, 9, 15)


def _scores(ctx, kind, rank0):
    """Row scores over `ctx` keys and the per-row lengths; "sat": rank0's keys
    lead every other row (a saturated rank, the rescue round)."""
    g = torch.Generator(device="cuda").manual_seed(1017)
    x = torch.randn(ROWS, ctx, generator=g, device="cuda")
    if kind == "smooth":
        z = torch.randn(ROWS, ctx + 2048, generator=g, device="cuda")
        c = torch.cumsum(z, 1)
        x = 2.0 * (c[:, 2048:] - c[:, :-2048]) / 2048**0.5 + 0.5 * x
        x[:, :4] += 8.0
    elif kind == "ints":
        x = x.round()
    else:
        x = 2.0 * x
        if kind == "sat":
            owned = torch.arange(ctx, device="cuda") % WORLD == rank0
            x[::2] += owned.float() * 8
    return x.bfloat16(), [ctx - 7 + t for t in range(ROWS)]


@pytest.mark.parametrize("rank0", [0, 13])
def test_merge_settle_all_matches_pairwise(rank0, monkeypatch):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import (
        packed_row_words,
        rescue_slots,
    )
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge.kernels import (
        DCPTopkRescueWorkspace,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_fake,
        dynamic_view,
        fake,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        indexer,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        topk_exchange as tx,
    )

    dev = torch.device("cuda", torch.accelerator.current_device_index())
    i32 = dict(dtype=torch.int32, device=dev)
    max_local = -(-MAX_MODEL_LEN // WORLD)
    pitch = -(-max_local // 512) * 512
    width = N + N // 32
    logits16 = torch.zeros(ROWS * pitch, dtype=torch.int16, device=dev)
    logits = logits16.view(torch.bfloat16).view(ROWS, pitch)
    ends, valid = torch.zeros(ROWS, **i32), torch.zeros(ROWS, **i32)
    out = torch.zeros(ROWS, width, **i32)
    local = cute.compile(
        indexer.IndexerLocalTopk(ROWS, N, pitch, WORLD),
        fake((ROWS * pitch,), (1,), cutlass.Int16),
        fake((ROWS,), (1,), torch.int32, 4),
        fake((ROWS, width), (width, 1), torch.int32, 4),
        fake((ROWS,), (1,), torch.int32, 4),
        dynamic_fake(1, torch.int64),
        dynamic_fake(1, torch.int32),
        Int32(0),
        Int32(1),
        make_fake_stream(),
    )

    def exchange():
        recv = torch.zeros(2, WORLD, ROWS, 4 * tx.ll_pairs(N), **i32)
        peers = torch.tensor([recv.data_ptr()] * WORLD, dtype=torch.int64, device=dev)
        return SimpleNamespace(
            recv=recv,
            peers=peers,
            epoch=torch.zeros(ROWS, **i32),
            rows=ROWS,
            world=WORLD,
            rank=rank0,
            local_topk=N,
        )

    def workspace():
        ws = DCPTopkRescueWorkspace.__new__(DCPTopkRescueWorkspace)
        recv = torch.zeros(ROWS, rescue_slots(TOPK, N), packed_row_words(TOPK), **i32)
        flags = torch.zeros(WORLD, ROWS, **i32)
        peer_recv, peer_flags = (
            torch.tensor([t.data_ptr()] * WORLD, device=dev) for t in (recv, flags)
        )
        ws._init(recv, flags, peer_recv, peer_flags, ROWS, TOPK, N)
        ws.world_size, ws.rank, ws.device_group = WORLD, rank0, None
        return ws

    def push(ex, rank):
        local(
            to_cute(logits16),
            to_cute(ends, 4),
            to_cute(out, 4),
            to_cute(valid, 4),
            dynamic_view(ex.peers),
            dynamic_view(ex.epoch),
            Int32(rank),
            Int32(ROWS),
            stream(dev),
        )

    arms = {}
    for settle in (False, True):
        table = {WORLD: "1-8:1"} if settle else {}
        monkeypatch.setattr(tx, "MERGE_SETTLE_ALL", table)
        assert tx.merge_settle_all(ROWS, WORLD, True) == settle
        ex, ws = exchange(), workspace()
        merge = tx.ExchangeMerge(ex, ws, ROWS, TOPK, PAGE)
        assert merge.select == "digits", merge.select
        arms[settle] = (ex, ws, merge)

    cols = -(-MAX_MODEL_LEN // (PAGE * WORLD))
    table = (torch.randperm(1 << 20, device=dev)[:cols].int() + 1).view(1, cols)
    req_id = torch.zeros(ROWS, **i32)
    for ctx in (32768, 131072):
        for kind in ("randn", "smooth", "ints", "sat"):
            for late in (False, True):
                scores, lens = _scores(ctx, kind, rank0)
                pending, own = {}, None
                for rank in range(WORLD):
                    logits.zero_()
                    for t in range(ROWS):
                        mine = scores[t, : lens[t]][rank::WORLD]
                        logits[t, : mine.numel()].copy_(mine)
                        ends[t] = mine.numel()
                    if rank == rank0:
                        own = (logits.clone(), ends.clone())
                    for settle, (ex, _, _) in arms.items():
                        if late and rank in LATE:
                            # Pushed into a scratch exchange now, copied in late.
                            scratch = exchange()
                            scratch.epoch.copy_(ex.epoch)
                            push(scratch, rank)
                            torch.accelerator.synchronize()
                            pending[settle, rank] = scratch.recv.clone()
                        else:
                            push(ex, rank)
                torch.accelerator.synchronize()
                assert own is not None
                got = {}
                for settle, (ex, ws, merge) in arms.items():
                    rows_out = torch.full((ROWS, TOPK), -9, **i32)
                    valid_out = torch.full((ROWS,), -9, **i32)
                    if late:
                        side = torch.cuda.Stream()
                        with torch.cuda.stream(side):
                            torch.cuda._sleep(1_000_000)
                            for rank in LATE:
                                ex.recv[:, rank].copy_(pending[settle, rank][:, rank])
                    merge(
                        own[0][:, :max_local],
                        own[1],
                        req_id,
                        table,
                        PAGE,
                        rows_out,
                        valid_out,
                    )
                    torch.accelerator.synchronize()
                    got[settle] = (rows_out, valid_out, ws.sat_word.clone(), ex.epoch)
                case = (rank0, ctx, kind, late)
                for a, b in zip(got[True], got[False]):
                    assert torch.equal(a, b), case
