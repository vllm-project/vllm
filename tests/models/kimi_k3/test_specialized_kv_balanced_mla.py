# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's kv_balanced MLA core (BalancedCore, the code path
PreparedMla.stock_core takes under DCP with mla_core=kv_balanced) against the
tokenspeed core it replaces (dcp.stock_decode, as PreparedMla calls it) and an FP32
reference: DCP shards on rank 3 with interleave 1 and 64, q = 1 / 5 / 6, an empty
(padded) request, and a captured plan + decode replayed after the lengths change."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip("SM100-family only", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.balanced import (  # noqa: E402
    BalancedCore,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (  # noqa: E402
    fill_splits,
    local_max,
    stock_decode,
)

PAGE, LAT, ROPE, H, DCP, RANK = 64, 512, 64, 96, 8, 3
SCALE = 1.0 / math.sqrt(192)
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def cuda_context():
    # CuTe's HardwareInfo (cluster occupancy) needs a current CUDA context, which the
    # serving runtime always has by the time the specialized decode is built.
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()


def local_len(total, interleave):
    """Tokens of a ``total``-token sequence on rank RANK (block interleave)."""
    full, rest = divmod(total, interleave * DCP)
    return full * interleave + min(max(rest - RANK * interleave, 0), interleave)


def global_pos(n, interleave, dev):
    i = torch.arange(n, device=dev)
    return (i // interleave) * interleave * DCP + RANK * interleave + i % interleave


def make(totals, q, interleave, dev, pages_per_req=None):
    local = [local_len(t, interleave) for t in totals]
    pages = pages_per_req or max(1, max(-(-n // PAGE) for n in local))
    cache = (torch.randn(len(totals) * pages + 4, PAGE, LAT + ROPE, device=dev) / 2).to(
        FP8
    )
    bt = torch.arange(len(totals) * pages, device=dev, dtype=torch.int32).view(
        len(totals), pages
    )
    query = (torch.randn(len(totals) * q, H, LAT + ROPE, device=dev) / 2).to(FP8)
    i32 = dict(dtype=torch.int32, device=dev)
    return query, cache, bt, torch.tensor(local, **i32), torch.tensor(totals, **i32)


def reference(query, cache, bt, lens, totals, q, interleave):
    out = torch.zeros(query.shape[0], H, LAT, device=query.device)
    lse = torch.full((query.shape[0], H), -math.inf, device=query.device)
    for b, n in enumerate(lens.tolist()):
        if n == 0:
            continue
        kv = cache[bt[b].long()].reshape(-1, LAT + ROPE)[:n].float()
        rows = query[b * q : (b + 1) * q].float()
        s = torch.einsum("qhd,kd->qhk", rows, kv) * SCALE
        bound = totals[b] - q + torch.arange(q, device=query.device)
        pos = global_pos(n, interleave, query.device)
        s = s.masked_fill(pos[None, None, :] > bound[:, None, None], -math.inf)
        out[b * q : (b + 1) * q] = torch.einsum("qhk,kd->qhd", s.softmax(-1), kv[:, :LAT])
        lse[b * q : (b + 1) * q] = torch.logsumexp(s, -1) / math.log(2)
    return out, lse


def run_stock(query, cache, bt, lens, totals, q, interleave):
    reqs = lens.numel()
    out = torch.empty(reqs, q, H, LAT, dtype=torch.bfloat16, device=query.device)
    lse = torch.empty(reqs, q, H, dtype=torch.float32, device=query.device)
    cp = dict(cp=(DCP, RANK, interleave), totals=totals, pack=True) if q > 1 else {}
    stock_decode(
        query.view(reqs, q, H, -1), cache, bt, lens, local_max(1 << 20, DCP),
        (SCALE, 1.0), out, lse, fill_splits(reqs * q, query.device), **cp,
    )
    return out.view(-1, H, LAT), lse.view(-1, H)


def run_balanced(core, query, cache, bt, lens, totals, interleave, plan=True):
    reqs, q = lens.numel(), core.core.q
    out = torch.empty(reqs * q, H, LAT, dtype=torch.bfloat16, device=query.device)
    lse = torch.empty(reqs * q, 1, H, dtype=torch.float32, device=query.device)
    core.run(
        query, cache, bt, lens, totals, (SCALE, 1.0), out, lse,
        (DCP, RANK, interleave), plan=plan,
    )
    return out, lse.view(-1, H)


def rel(a, b):
    return ((a.float() - b).norm() / b.norm().clamp_min(1e-6)).item()


@pytest.mark.parametrize("interleave", [1, 64])
@pytest.mark.parametrize("q", [1, 5, 6])
def test_matches_stock_and_reference(q, interleave):
    dev = torch.device("cuda")
    torch.manual_seed(q * 100 + interleave)
    # Lengths ending inside this rank's interleave chunks, short and long, plus an
    # empty (padded) request.
    totals = [70_001, 0, 24_577] if q > 1 else [70_001, 1_234, 24_577, 0, 9_017]
    core = BalancedCore(H, q, 64 // q, dev)
    args = make(totals, q, interleave, dev)
    ref, ref_lse = reference(*args, q, interleave)
    out, lse = run_balanced(core, *args, interleave)
    live = torch.tensor(
        [t > 0 and local_len(t, interleave) > 0 for t in totals for _ in range(q)],
        device=dev,
    )
    assert rel(out[live], ref[live]) < 0.03
    assert (out[~live] == 0).all()
    assert torch.allclose(lse[live], ref_lse[live], atol=0.05, rtol=0.0)
    assert (lse[~live] <= -1e30).all()
    stock, stock_lse = run_stock(*args, q, interleave)
    assert rel(out[live], stock[live].float()) < 0.03
    assert torch.allclose(lse[live], stock_lse[live], atol=0.05, rtol=0.0)


def test_graph_replay_with_new_lengths():
    dev, q, interleave = torch.device("cuda"), 5, 1
    core = BalancedCore(H, q, 64 // q, dev)
    pages = 300_000 // DCP // PAGE + 2
    query, cache, bt, lens, totals = make([200_000, 3_000, 50_000], q, 1, dev, pages)
    out = torch.empty(query.shape[0], H, LAT, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(query.shape[0], 1, H, dtype=torch.float32, device=dev)

    def step():
        core.run(query, cache, bt, lens, totals, (SCALE, 1.0), out, lse,
                 (DCP, RANK, interleave), plan=True)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for new in ([3_000, 290_000, 64], [120_000, 120_000, 120_000]):
        totals.copy_(torch.tensor(new, dtype=torch.int32))
        lens.copy_(torch.tensor([local_len(t, interleave) for t in new], dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        ref, ref_lse = reference(query, cache, bt, lens, totals, q, interleave)
        assert rel(out, ref) < 0.03
        assert torch.allclose(lse.view(-1, H), ref_lse, atol=0.05, rtol=0.0)
