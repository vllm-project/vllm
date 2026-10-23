# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused FlyDSL norm + RoPE + NVFP4 insert against the HIP norm + RoPE
kernel followed by the reference NVFP4 inserts: every output byte, stale page
bytes included, and qkv left untouched."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm (gfx950) FlyDSL kernel", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 instructions", allow_module_level=True)

from tests.kernels.attention.minimax_m3_nvfp4_ref import (  # noqa: E402
    insert_nvfp4_index,
    insert_nvfp4_kv,
)
from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.minimax_m3.amd.ops.norm_rope_nvfp4_insert.host import (  # noqa: E402
    minimax_m3_norm_rope_nvfp4_insert,
)

D = 128
BLK = 128
EPS = 1e-6
ROT = 64


def _request_slots(starts, lens, num_pages, gen):
    """VLLM's slot mapping: request r's tokens at positions starts[r].. of its
    own random pages."""
    perm = torch.randperm(num_pages, generator=gen)
    out, p = [], 0
    for st, n in zip(starts, lens):
        npg = (st + n + BLK - 1) // BLK
        pos = torch.arange(st, st + n)
        out.append(perm[p : p + npg][pos // BLK] * BLK + pos % BLK)
        p += npg
    return torch.cat(out) if out else torch.empty(0, dtype=torch.int64)


class _Batch:
    def __init__(self, starts, lens, pad, heads, scales, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.nq, self.nkv, self.niq = heads
        n = self.n = sum(lens) + pad
        row = (self.nq + 2 * self.nkv + self.niq + 1) * D
        # magnitudes 1e-3 .. 3e3 per 16-value group, some all-zero groups
        x = torch.randn(n, row // 16, 16, generator=g)
        mags = torch.tensor([1e-3, 0.05, 1.0, 1.0, 30.0, 3e3])
        x *= mags[torch.randint(0, len(mags), (n, row // 16), generator=g)][..., None]
        x[torch.rand(n, row // 16, generator=g) < 0.02] = 0.0
        self.qkv = x.view(n, row).to(torch.bfloat16).cuda()
        self.w = [
            (torch.randn(D, generator=g) * 0.5).to(torch.bfloat16).cuda()
            for _ in range(4)
        ]
        self.cs = (torch.rand(4096, ROT, generator=g) * 2 - 1).to(torch.bfloat16).cuda()
        self.pos = torch.randint(0, 4096, (n,), generator=g).cuda()
        pages = sum((s + m + BLK - 1) // BLK for s, m in zip(starts, lens)) + 2
        pad_slots = torch.full((pad,), -1, dtype=torch.int64)
        self.slots = torch.cat(
            [_request_slots(starts, lens, pages, g), pad_slots]
        ).cuda()
        self.islots = torch.cat(
            [_request_slots(starts, lens, pages, g), pad_slots]
        ).cuda()
        # stale bytes; the layer views have a block stride past one page
        main = 2 * self.nkv * BLK * 72
        self.kv_full = torch.randint(0, 256, (pages, main + 256), generator=g)
        self.kv_full = self.kv_full.to(torch.uint8).cuda()
        self.idx_full = torch.randint(0, 256, (pages, BLK * 72 + 64), generator=g)
        self.idx_full = self.idx_full.to(torch.uint8).cuda()
        self.ks = torch.tensor(scales[0], dtype=torch.float32, device="cuda")
        self.vs = torch.tensor(scales[1], dtype=torch.float32, device="cuda")

    def kv(self, full):
        return full[:, : 2 * self.nkv * BLK * 72].view(-1, 2 * self.nkv, BLK, 72)

    def idx(self, full):
        return full[:, : BLK * 72].view(-1, BLK, 72)

    def reference(self):
        qkv = self.qkv.clone()
        q = torch.empty(self.n, self.nq * D, dtype=torch.bfloat16, device="cuda")
        iq = torch.empty(self.n, self.niq * D, dtype=torch.float8_e4m3fn, device="cuda")
        kv_full, idx_full = self.kv_full.clone(), self.idx_full.clone()
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, self.w[0], self.w[1], self.cs, self.pos, self.nq, self.nkv, ROT,
            EPS, self.w[2], self.w[3], self.niq, q_out=q, index_q_out=iq,
        )  # fmt: skip
        k0, v0 = self.nq * D, (self.nq + self.nkv) * D
        ik0 = (self.nq + 2 * self.nkv + self.niq) * D
        insert_nvfp4_kv(
            qkv[:, k0:v0].view(self.n, self.nkv, D),
            qkv[:, v0 : v0 + self.nkv * D].view(self.n, self.nkv, D),
            self.kv(kv_full),
            self.slots,
            self.ks,
            self.vs,
        )
        insert_nvfp4_index(self.idx(idx_full), qkv[:, ik0 : ik0 + D], self.islots)
        return q, iq, kv_full, idx_full

    def run(self, qkv, pos, slots, islots, kv_full, idx_full, q, iq):
        minimax_m3_norm_rope_nvfp4_insert(
            qkv, *self.w, self.cs, pos, slots, islots, self.kv(kv_full),
            self.idx(idx_full), self.ks, self.vs, q, iq, num_heads=self.nq,
            num_kv_heads=self.nkv, num_index_heads=self.niq, eps=EPS,
        )  # fmt: skip

    def fused(self, iq):
        qkv = self.qkv.clone()
        q = torch.empty(self.n, self.nq * D, dtype=torch.bfloat16, device="cuda")
        kv_full, idx_full = self.kv_full.clone(), self.idx_full.clone()
        self.run(qkv, self.pos, self.slots, self.islots, kv_full, idx_full, q, iq)
        assert torch.equal(qkv.view(torch.int16), self.qkv.view(torch.int16))
        return q, kv_full, idx_full


def _check(batch, iq_mode="contiguous"):
    q_ref, iq_ref, kv_ref, idx_ref = batch.reference()
    n, w = batch.n, batch.niq * D
    if iq_mode == "rank":  # the context-parallel indexer's [4, n, 128] buffer
        buf = torch.full((4, n, w), 0x7F, dtype=torch.uint8, device="cuda")
        iq = buf[2]
    elif iq_mode == "cols":
        buf = torch.full((n, 3 * w), 0x7F, dtype=torch.uint8, device="cuda")
        iq = buf[:, w : 2 * w]
    else:
        buf = iq = torch.full((n, w), 0x7F, dtype=torch.uint8, device="cuda")
    q, kv, idx = batch.fused(iq.view(torch.float8_e4m3fn))
    assert torch.equal(q.view(torch.int16), q_ref.view(torch.int16))
    assert torch.equal(iq, iq_ref.view(torch.uint8))
    assert torch.equal(kv, kv_ref)
    assert torch.equal(idx, idx_ref)
    iq.fill_(0x7F)
    assert bool((buf == 0x7F).all()), "bytes outside index_q_out written"


TP4, TP1 = (16, 1, 1), (64, 4, 4)
DSPARK = ([3 + 9 * i for i in range(43)], [4] * 43)  # 172 decode rows


@pytest.mark.parametrize("heads", [TP4, TP1])
@pytest.mark.parametrize(
    ("starts", "lens", "pad"),
    [
        ([5], [1], 0),  # one decode token
        ([0, 13], [7, 4], 1),  # a run of 7, runs across an octet edge
        ([125, 254, 4091], [6, 6, 6], 12),  # Q6 crossing page boundaries
        (*DSPARK, 20),  # DSpark decode rows, graph padding
        ([0, 3, 250, 128], [131, 61, 9, 99], 0),  # unaligned prefill chunks
        ([0, 100], [4096, 4000], 3),  # the prefill tier
    ],
)
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.002, 3.0)])
def test_matches_two_kernel_path(heads, starts, lens, pad, scales):
    _check(_Batch(starts, lens, pad, heads, scales))


@pytest.mark.parametrize("iq_mode", ["rank", "cols"])
@pytest.mark.parametrize(("starts", "lens"), [DSPARK, ([7], [600])])
def test_index_q_out_view(iq_mode, starts, lens):
    _check(_Batch(starts, lens, 5, TP4, (1.0, 1.0)), iq_mode)


def test_64k_tokens():
    _check(_Batch([0, 777], [32768, 32768], 0, TP4, (1.0, 1.0)))


def test_all_padding():
    """A graph-capture dummy batch: every slot -1, the pages stay as they were."""
    batch = _Batch([], [], 64, TP4, (1.0, 1.0))
    _check(batch)
    iq = torch.empty(batch.n, D, dtype=torch.float8_e4m3fn, device="cuda")
    _, kv, idx = batch.fused(iq)
    assert torch.equal(kv, batch.kv_full) and torch.equal(idx, batch.idx_full)


def test_cuda_graph_replay():
    """Captured once, replayed on new qkv / positions / slots in place."""
    a = _Batch(*DSPARK, 4, TP4, (1.0, 1.0), seed=1)
    b = _Batch(*DSPARK, 4, TP4, (1.0, 1.0), seed=2)
    b.w, b.cs = a.w, a.cs  # captured with the layer's weights and RoPE cache
    qkv, pos = a.qkv.clone(), a.pos.clone()
    slots, islots = a.slots.clone(), a.islots.clone()
    kv_full, idx_full = b.kv_full.clone(), b.idx_full.clone()
    q = torch.empty(a.n, a.nq * D, dtype=torch.bfloat16, device="cuda")
    iq = torch.empty(a.n, D, dtype=torch.float8_e4m3fn, device="cuda")
    args = (qkv, pos, slots, islots, kv_full, idx_full, q, iq)
    a.run(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a.run(*args)
    new = (b.qkv, b.pos, b.slots, b.islots, b.kv_full, b.idx_full)
    for dst, src in zip(args[:6], new):
        dst.copy_(src)
    graph.replay()
    q_ref, iq_ref, kv_ref, idx_ref = b.reference()
    assert torch.equal(q.view(torch.int16), q_ref.view(torch.int16))
    assert torch.equal(iq.view(torch.uint8), iq_ref.view(torch.uint8))
    assert torch.equal(kv_full, kv_ref) and torch.equal(idx_full, idx_ref)
