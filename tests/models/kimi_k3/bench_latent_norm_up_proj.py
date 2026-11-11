# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark (one GPU): K3 latent RMSNorm + sharded up-proj addmm (old)
vs the fused Triton kernel (new), TP8 shard shapes (K 3584, N 896).

    python tests/models/kimi_k3/bench_latent_norm_up_proj.py

Each step uses one of NW distinct weights (48 x 6.4 MB, more than the MALL),
so the weight streams from HBM as in a real decode step. ITERS steps are
captured in one CUDA graph and replayed REPLAYS times (hip-event timing).
CFG=bm,bn,split_k,block_k,warps,stages overrides the kernel launch config.
"""

import os

import torch

import vllm.kernels  # noqa: F401
from vllm.models.kimi_k3.amd.ops import latent_norm_up_proj as mod

K, N, HIDDEN, EPS = 3584, 896, 7168, 1e-5
ITERS = int(os.environ.get("ITERS", "96"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
NW = int(os.environ.get("NW", "48"))
TOKENS = [int(t) for t in os.environ.get("TOKENS", "4,16,64,192").split(",")]


def _time(fn) -> float:
    for i in range(3):
        fn(i)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(ITERS):
            fn(i)
    g.replay()
    torch.accelerator.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPLAYS):
        g.replay()
    e.record()
    torch.accelerator.synchronize()
    return s.elapsed_time(e) * 1e3 / (REPLAYS * ITERS)


def main() -> None:
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(NW)]
    nw = (1 + 0.1 * torch.randn(K, device=dev)).to(torch.bfloat16)
    cfg = os.environ.get("CFG")
    if cfg:
        bm, bn, sk, bk, warps, stages = map(int, cfg.split(","))
        mod._launch_config = lambda m: (
            min(max(16, 1 << (m - 1).bit_length()), bm),
            bn,
            sk,
            bk,
            warps,
            stages,
        )
    print(
        f"{'tokens':>6} {'norm':>7} {'addmm':>7} {'old':>7} {'new':>7} "
        f"{'speedup':>7} {'maxerr':>7}"
    )
    for m in TOKENS:
        x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        shared = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16)
        start = 3 * N
        out = shared.narrow(-1, start, N)

        def norm(i, x=x, out=out):
            torch.ops.vllm_aiter.rms_norm(x, nw, EPS)

        def addmm(i, x=x, out=out):
            out.addmm_(x, ws[i % NW].t())

        def old(i, x=x, out=out):
            out.addmm_(torch.ops.vllm_aiter.rms_norm(x, nw, EPS), ws[i % NW].t())

        def wread(i, x=x, out=out):
            ws[i % NW].sum(dtype=torch.float32)

        def new(i, x=x, out=out):
            mod.latent_norm_up_proj_triton(x, nw, EPS, ws[i % NW], out)

        ref = shared.clone()
        ref[:, start : start + N].addmm_(
            torch.ops.vllm_aiter.rms_norm(x, nw, EPS), ws[0].t()
        )
        got = shared.clone()
        mod.latent_norm_up_proj_triton(x, nw, EPS, ws[0], got.narrow(-1, start, N))
        err = (got.float() - ref.float()).abs().max().item()
        t = {f.__name__: _time(f) for f in (norm, addmm, old, new, wread)}
        print(
            f"{m:6d} {t['norm']:7.2f} {t['addmm']:7.2f} {t['old']:7.2f} "
            f"{t['new']:7.2f} {t['old'] / t['new']:7.2f} {err:7.4f} "
            f"wread {t['wread']:6.2f}"
        )


if __name__ == "__main__":
    main()
