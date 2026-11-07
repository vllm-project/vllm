# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One gfx942 specialized-decode layer launch on 8 MI300X GPUs (TP = EP = 8)
against a float64 reference with the stock rounding points, at random weights
of the real shapes.

Every rank builds its own weights (shared seeds for the replicated tensors),
runs the kernel twice (fresh epochs: bitwise equal), and computes the stock
layer on its GPU with all-gathers and rank-order sums. Also prints the launch
time. Run: ``pytest -s test_layer_tp8.py`` or, from the repo root,
``python -m tests.models.muse_spark.amd_specialized_decode.test_layer_tp8 1 8``.
"""

import itertools
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import reference as ref

TP = 8
FP8 = torch.float8_e4m3fnuz


def _gpus_ok() -> bool:
    if torch.accelerator.device_count() < TP:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 3


# --------------------------------------------------------------------------
# numerics of the kernels' quant sites
# --------------------------------------------------------------------------


def quant_rows(x, q, nudge=False):
    """Per-row FP8 as the kernels (``layout.Quant``): f32 scale by divide or by
    ``* (1 / max)``, the floor, codes ``x * (1 / scale)`` saturated. The
    kernels' ``1 / scale`` (v_rcp_f32) is within an ulp of this one; with
    bf16 inputs codes on FP8 rounding midpoints are common, so ``nudge`` (the
    factor an ulp down) gives the reference's own noise floor."""
    amax = x.abs().amax(-1, keepdim=True).float()
    if q.divide:
        scale = amax / q.fp8_max
    else:
        scale = amax * torch.tensor(1.0 / q.fp8_max, dtype=torch.float32)
    scale = scale.clamp(min=q.min_scale)
    factor = 1.0 / scale
    if nudge:
        factor = torch.nextafter(factor, torch.zeros_like(factor))
    codes = (x.float() * factor).clamp(-q.fp8_max, q.fp8_max).to(FP8)
    return codes.double(), scale.double()


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


class Gen:
    def __init__(self, seed: int, dev):
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.dev = dev

    def randn(self, *shape, scale=1.0):
        return torch.randn(*shape, generator=self.g, device=self.dev) * scale

    def rand(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.g, device=self.dev)

    def fp8(self, n, k):
        """FP8 codes [n, k] and f32 per-row scales [n]."""
        w = self.randn(n, k)
        s = w.abs().amax(1, keepdim=True) / 240.0
        q = (w / s).clamp(-240, 240).to(FP8)
        return q, (s.squeeze(1) * (0.02 / s.mean())).float()


def make_layer(rank: int, dev, seed: int):
    """The rank's kernel weights, and the replicated tensors the reference
    needs whole."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime import router_planes
    from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
        E_LOCAL,
        HIDDEN,
        INTER,
        POOL_ROWS,
        POOLED,
        Q_ROWS,
        QKVG,
        UNPOOL_ROWS,
    )

    shared = Gen(seed, dev)
    full = dict(
        router=shared.randn(TP * E_LOCAL, HIDDEN, scale=0.02),
        bias=shared.randn(TP * E_LOCAL, scale=0.01),
        pooler=shared.randn(POOLED, HIDDEN, scale=0.02).bfloat16(),
        unpooler=shared.randn(HIDDEN, POOLED, scale=0.02).bfloat16(),
        attn_alpha=shared.rand(HIDDEN, lo=0.5, hi=1.0),
        attn_beta=shared.rand(HIDDEN, lo=0.5, hi=1.5),
        ffn_alpha=shared.rand(HIDDEN, lo=0.5, hi=1.0),
        ffn_beta=shared.rand(HIDDEN, lo=0.5, hi=1.5),
        pre_ffn_w1=1.0 + shared.randn(HIDDEN, scale=0.1),
        pre_expert_w1=1.0 + shared.randn(POOLED, scale=0.1),
        post_expert_w1=1.0 + shared.randn(POOLED, scale=0.1),
        post_ffn_w1=1.0 + shared.randn(HIDDEN, scale=0.1),
        pre_qkv_w1=1.0 + shared.randn(HIDDEN, scale=0.1),
    )
    own = Gen(seed * 1000 + 1 + rank, dev)
    o_w, o_ws = own.fp8(HIDDEN, Q_ROWS)
    qkvg_w, qkvg_ws = own.fp8(QKVG, HIDDEN)
    w13 = torch.empty(E_LOCAL, 2 * INTER, POOLED, dtype=FP8, device=dev)
    w13s = torch.empty(E_LOCAL, 2 * INTER, dtype=torch.float32, device=dev)
    w2 = torch.empty(E_LOCAL, POOLED, INTER, dtype=FP8, device=dev)
    w2s = torch.empty(E_LOCAL, POOLED, dtype=torch.float32, device=dev)
    for e in range(E_LOCAL):
        w13[e], w13s[e] = own.fp8(2 * INTER, POOLED)
        w2[e], w2s[e] = own.fp8(POOLED, INTER)
    t = dict(
        o_w=o_w,
        o_ws=o_ws,
        router=router_planes(full["router"][rank * E_LOCAL : (rank + 1) * E_LOCAL]),
        pooler=full["pooler"][rank * POOL_ROWS : (rank + 1) * POOL_ROWS].contiguous(),
        unpooler=full["unpooler"][
            rank * UNPOOL_ROWS : (rank + 1) * UNPOOL_ROWS
        ].contiguous(),
        w13=w13,
        w13s=w13s,
        w2=w2,
        w2s=w2s,
        qkvg_w=qkvg_w,
        qkvg_ws=qkvg_ws,
    )
    for n in (
        "bias",
        "attn_alpha",
        "attn_beta",
        "ffn_alpha",
        "ffn_beta",
        "pre_ffn_w1",
        "pre_expert_w1",
        "post_expert_w1",
        "post_ffn_w1",
        "pre_qkv_w1",
    ):
        t[n] = full[n].float().contiguous()
    return t, full


# --------------------------------------------------------------------------
# the reference (float64 on the rank's GPU, stock rounding points)
# --------------------------------------------------------------------------


def gather_sum(part):
    """The TP sum of bf16 partials in rank order, one bf16 round."""
    parts = [torch.empty_like(part) for _ in range(TP)]
    dist.all_gather(parts, part.contiguous())
    acc = torch.zeros_like(parts[0], dtype=torch.float64)
    for p in parts:
        acc = acc + p.double()
    return ref.bf(acc)


def reference(rank, t, full, attn, og, stream, nudge=False):
    from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
        DENSE_QUANT,
        E_LOCAL,
        INTER,
        MOE_IN_QUANT,
        MOE_MID_QUANT,
        POST_EPS,
        TOPK,
    )

    d = lambda n: t[n].double()  # noqa: E731
    out = {}
    # o-path, o_proj, TP sum
    x = ref.o_gate_norm(attn.double(), og.double(), 64)
    q, s = quant_rows(x, DENSE_QUANT, nudge)
    y = gather_sum(ref.scaled_mm(q, s, t["o_w"].double(), d("o_ws")).bfloat16())
    # attention boundary (weightless post norm)
    x2, s1 = ref.boundary(
        y,
        stream.double(),
        None,
        False,
        d("attn_alpha"),
        d("attn_beta"),
        d("pre_ffn_w1") - 1.0,
        POST_EPS,
    )
    out["x2"] = x2
    # routing and pooling (replicated)
    logits = (x2 @ full["router"].double().T).float().double()
    biased = torch.sigmoid(logits) + full["bias"].double()[None]
    top = biased.topk(TOPK + 1, dim=-1).values
    out["margin"] = top[:, TOPK - 1] - top[:, TOPK]
    ids, tw = ref.route(logits.cpu(), full["bias"].double().cpu(), TOPK)
    ids, tw = ids.to(x2.device), tw.to(x2.device)
    out["ids"] = ids
    pooled = ref.bf(x2 @ full["pooler"].double().T)
    h = ref.bf(ref.rms(pooled, ref.RMS_EPS) * d("pre_expert_w1"))
    hq, hs = quant_rows(h, MOE_IN_QUANT, nudge)
    out.update(hq=hq, hs=hs)
    # the rank's experts; per local assignment (t, k) its act, y and the
    # owners' sums of squares (NaN elsewhere), for the stage diagnostics
    rows = x2.shape[0]
    nan = float("nan")
    out["act"] = torch.full(
        (rows, TOPK, INTER), nan, dtype=torch.float64, device=x2.device
    )
    out["y"] = torch.full(
        (rows, TOPK, pooled.shape[1]), nan, dtype=torch.float64, device=x2.device
    )
    out["sq"] = torch.full(
        (rows, TOPK, pooled.shape[1] // 64), nan, dtype=torch.float64, device=x2.device
    )
    part = torch.zeros_like(pooled)
    first = rank * E_LOCAL
    for e in range(E_LOCAL):
        tok, k = (ids == first + e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        a = ref.expert_act(hq[tok], hs[tok], t["w13"][e].double(), d("w13s")[e], INTER)
        aq, as_ = quant_rows(a, MOE_MID_QUANT, nudge)
        ye = ref.scaled_mm(aq, as_, t["w2"][e].double(), d("w2s")[e])
        xe = ye * d("post_expert_w1")
        out["act"][tok, k] = a
        out["y"][tok, k] = ye
        out["sq"][tok, k] = xe.pow(2).view(len(tok), -1, 64).sum(-1)
        scale = torch.rsqrt(xe.pow(2).mean(-1, keepdim=True) + POST_EPS)
        part.index_add_(0, tok, xe * scale * tw[tok, k][:, None])
    reduced = gather_sum(ref.bf(part).bfloat16())
    branch = ref.bf(reduced @ full["unpooler"].double().T)
    out.update(part=ref.bf(part), reduced=reduced, branch=branch)
    # FFN boundary (input-scaled post norm), next pre_qkv norm, quant, QKVG
    x3, s2 = ref.boundary(
        branch,
        s1,
        d("post_ffn_w1") - 1.0,
        True,
        d("ffn_alpha"),
        d("ffn_beta"),
        d("pre_qkv_w1") - 1.0,
        POST_EPS,
    )
    q3, s3 = quant_rows(x3, DENSE_QUANT, nudge)
    out.update(x3=x3, xq=q3, xqs=s3)
    out["stream"] = s2
    out["qkv"] = ref.scaled_mm(q3, s3, t["qkvg_w"].double(), d("qkvg_ws"))
    return out


# --------------------------------------------------------------------------
# a rank
# --------------------------------------------------------------------------


def _width(rank, rows, seed, iters, t, full, wtab, bufs, launches, dev) -> dict:
    """One width's launches on this rank: the metrics against the reference.
    ``launches`` counts every launch of the rank: they alternate barrier slots
    (a launch zeroes the next one's)."""
    from vllm.models.muse_spark.amd.specialized_decode.wide import layer as wide
    from vllm.models.muse_spark.amd.specialized_decode.wide.abi import FAILED
    from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
        HIDDEN,
        KV_ROWS,
        Q_ROWS,
        QKVG,
        TOPK,
    )
    from vllm.utils.torch_utils import current_stream

    launch, _ = wide.build(rows)
    g = Gen(seed * 7 + rows, dev)  # same inputs on every rank (replicated x)
    stream0 = g.randn(rows, HIDDEN)
    heads = Gen(seed * 11 + rows * 100 + rank, dev)  # the rank's heads
    attn = heads.randn(rows, Q_ROWS).bfloat16()
    qkv_in = heads.randn(rows, QKVG).bfloat16()
    og = qkv_in[:, Q_ROWS + 2 * KV_ROWS :]
    stream = bufs.stream(rows)

    def run(out, reset=True):
        if reset:
            stream.copy_(stream0)
        bufs.epoch.add_(1)
        sync, sync_next, folds = bufs.slots(next(launches))
        launch(
            wtab.data_ptr(),
            bufs.stab.data_ptr(),
            bufs.ptab.data_ptr(),
            attn.data_ptr(),
            qkv_in.data_ptr(),
            out.data_ptr(),
            bufs.epoch.data_ptr(),
            1,
            rank,
            sync,
            sync_next,
            folds,
            bufs.health.data_ptr(),
            stream=current_stream(),
        )

    outs = [
        torch.full((rows, QKVG), float("nan"), dtype=torch.bfloat16, device=dev)
        for _ in range(2)
    ]
    run(outs[0])
    torch.accelerator.synchronize()
    stream_a = stream.clone()
    x2 = bufs.scratch["x2"][: rows * HIDDEN * 2].view(torch.bfloat16)
    x2 = x2.view(rows, HIDDEN).clone()
    ids = bufs.scratch["ids"][: rows * TOPK * 4].view(torch.int32)
    ids = ids.view(rows, TOPK).long().clone()
    run(outs[1])
    torch.accelerator.synchronize()
    stream_b = stream.clone()

    r = reference(rank, t, full, attn, og, stream0)
    ok_tok = r["margin"] > 1e-4
    same_ids = (ids.sort(-1).values == r["ids"].sort(-1).values).all(-1)
    m = {
        "health": int(bufs.health[FAILED]),
        "repeat": bool(torch.equal(outs[0], outs[1]))
        and bool(torch.equal(stream_a, stream_b)),
        "x2": rel(x2, r["x2"]),
        "ids": float(same_ids[ok_tok].float().mean()) if ok_tok.any() else 1.0,
        "stream": rel(stream_a[same_ids], r["stream"][same_ids]),
        "qkv": rel(outs[0][same_ids], r["qkv"][same_ids]),
    }
    floor = reference(rank, t, full, attn, og, stream0, nudge=True)
    m["stream_floor"] = rel(floor["stream"][same_ids], r["stream"][same_ids])
    m["qkv_floor"] = rel(floor["qkv"][same_ids], r["qkv"][same_ids])
    # back-to-back launches (the stream drifts; only the time matters)
    start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
    dist.barrier()
    start.record()
    for _ in range(iters):
        run(outs[1], reset=False)
    end.record()
    torch.accelerator.synchronize()
    m["us"] = start.elapsed_time(end) * 1e3 / iters
    m["health_timed"] = int(bufs.health[FAILED])
    return m


def _rank(rank: int, widths, port: int, seed: int, iters: int):
    os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", "/tmp/flydsl_cache")
    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP
    )
    cpu_group = dist.new_group(backend="gloo")

    from vllm.models.muse_spark.amd.specialized_decode.runtime import Buffers
    from vllm.models.muse_spark.amd.specialized_decode.wide import layer as wide
    from vllm.models.muse_spark.amd.specialized_decode.wide.layout import MAX_TOKENS

    t, full = make_layer(rank, dev, seed)
    wtab = torch.tensor(
        [[t[n].data_ptr() for n in wide.WEIGHTS]], dtype=torch.int64, device=dev
    )
    bufs = Buffers(dev, cpu_group, TP)
    failures = []
    launches = itertools.count()
    for rows in widths:
        assert rows <= MAX_TOKENS
        m = _width(rank, rows, seed, iters, t, full, wtab, bufs, launches, dev)
        bad = (
            m["health"] != 0
            or m["health_timed"] != 0
            or not m["repeat"]
            or m["x2"] > 5e-3
            or m["ids"] < 1.0
            # FP8 midpoint flips compound through the layer: within a few
            # times the reference's own one-ulp noise
            or m["stream"] > max(5e-3, 3 * m["stream_floor"])
            or m["qkv"] > max(2e-2, 3 * m["qkv_floor"])
        )
        if rank == 0 or bad:
            print(f"rank {rank} rows {rows}: {m}", flush=True)
        if bad:
            failures.append((rows, m))
    bufs.close()
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise AssertionError(f"rank {rank}: {failures}")


def run_layer(widths, seed: int = 0, iters: int = 20, port: int = 29517):
    mp.spawn(_rank, args=(list(widths), port, seed, iters), nprocs=TP, join=True)


@pytest.mark.skipif(not _gpus_ok(), reason="needs 8 gfx942 GPUs and FlyDSL")
@pytest.mark.parametrize("rows", [1, 8, 16, 128])
def test_layer_tp8(rows):
    run_layer([rows], seed=rows)


if __name__ == "__main__":
    run_layer([int(a) for a in sys.argv[1:]] or [1, 8, 32, 128])
