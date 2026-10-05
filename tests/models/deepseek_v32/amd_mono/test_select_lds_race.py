# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU model of the fused paged indexer's select-stage LDS protocol
(mono/kernel/glm/kernel.py, index_paged branch), from the last radix digit
(select_digit) through the gt / eq compaction scans, with an adversarial wave scheduler.

Each of the 8 waves is a list of LDS ops grouped into barrier phases; the scheduler runs
the waves one at a time between barriers in a chosen order (a legal GPU interleaving).
Without the barrier after select_digit some schedules corrupt the CSR (wrong slots,
stores past the row); with it every schedule gives the canonical result.
"""

import itertools
import random

import torch

WAVES, LANES, THREADS, TOPK, MAXS = 8, 64, 512, 2048, 4096
ITEMS = MAXS // THREADS


def order_key(x: torch.Tensor) -> torch.Tensor:
    b = x.contiguous().view(torch.int32).long()
    return torch.where(b >= 0, b ^ (-(2**31)), ~b) & 0xFFFFFFFF


def run(scores: torch.Tensor, barrier_after_digit: bool, order) -> tuple[list, list]:
    """-> (CSR positions written in [0, topk) order, out-of-range offsets written).
    ``order``: wave order within every barrier phase."""
    L = scores.numel()
    key = order_key(scores).tolist() + [0] * (MAXS - L)
    thr_true = sorted(key[:L], reverse=True)[TOPK - 1]
    # final select_digit's hit thread
    keys = {
        256: thr_true,
        257: TOPK - sum(k > thr_true for k in key[:L]),
    }
    wave_state: list[dict] = [dict() for _ in range(WAVES)]
    out, oob = {}, []

    # phase A (after select_digit's last barrier): every wave reads (prefix, remain) =
    # keys[256], keys[257]
    def a_read(w):
        wave_state[w]["thr"] = keys[256]

    # phase B: item flags + gt scan_flags -> lane 63 of wave w stores keys[256 + w] (its
    # gt count)
    def b_write(w):
        thr = wave_state[w]["thr"]
        cnt = []
        for lane in range(LANES):
            t = w * LANES + lane
            items = [t * ITEMS + j for j in range(ITEMS)]
            cnt.append(sum((i < L) and key[i] > thr for i in items))
        wave_state[w]["cnt"] = cnt
        keys[256 + w] = sum(cnt)

    # phase C (after scan barrier): offsets from keys[256..263], gt stores
    def c_store(w):
        thr = wave_state[w]["thr"]
        before = sum(keys[256 + v] for v in range(w))
        run_ = before
        for lane in range(LANES):
            t = w * LANES + lane
            for j in range(ITEMS):
                i = t * ITEMS + j
                if i < L and key[i] > thr:
                    if run_ < TOPK:
                        out[run_] = i
                    else:
                        oob.append(run_)
                    run_ += 1

    phases = (
        [[a_read], [b_write], [c_store]]
        if barrier_after_digit
        else [[a_read, b_write], [c_store]]
    )
    for ph in phases:
        for w in order:
            for op in ph:
                op(w)
    return [out.get(k, -1) for k in range(TOPK)], oob


def test_barrier_after_select_digit_is_schedule_independent():
    g = torch.Generator().manual_seed(0)
    L = 3500
    scores = torch.randn(L, generator=g)
    k = order_key(scores)
    thr = torch.sort(k, descending=True).values[TOPK - 1]
    want = torch.nonzero(k > thr).view(-1).tolist()  # canonical gt part (ascending)
    orders = [list(range(WAVES)), list(reversed(range(WAVES)))]
    rnd = random.Random(1)
    for _ in range(30):
        o = list(range(WAVES))
        rnd.shuffle(o)
        orders.append(o)
    # without the barrier: the in-order schedule (lagging waves read after wave 0 / 1
    # stored their counts) breaks
    bad = 0
    for o in orders:
        got, oob = run(scores, False, o)
        if got[: len(want)] != want or oob:
            bad += 1
    assert bad > 0, "the model should expose the race without the barrier"
    got, oob = run(scores, False, list(range(WAVES)))
    # with the barrier: every schedule gives the canonical result
    for o in orders + [
        list(p) for p in itertools.islice(itertools.permutations(range(WAVES)), 200)
    ]:
        got, oob = run(scores, True, o)
        assert got[: len(want)] == want and not oob, o
