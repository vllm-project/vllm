# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The timed closures are built and run within one loop iteration (no late binding).
# ruff: noqa: B023
"""8-rank checks and microbenchmarks of the kdamoe amd_kernels tails (run on all 8
GPUs):

  attnres  fused all-reduce + AttnRes (norm route) vs AITER all-reduce + stock
           _apply_attn_res
  latent   latent all-reduce + RMSNorm + up-proj (+ 16-slot fold) vs AR +
           k3_latent_norm_up_proj

Each mode spawns 8 ranks (needs 8 GPUs):

    python tests/models/kimi_k3/akr/ranks8_akr.py test  [attnres,latent]
    python tests/models/kimi_k3/akr/ranks8_akr.py bench [attnres,latent]

Bench: CUDA-graph replay, two-graph differencing ((t(2N) - t(N)) / N), min of 15
replays, max over ranks.
"""

import os
import sys
import types

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.utils.network_utils import get_open_port

H, LAT, SHARD = 7168, 3584, 896
M_LIST = [
    int(t) for t in os.environ.get("TOKENS", "1,12,24,40,65,80,128,192").split(",")
]
ITERS = int(os.environ.get("ITERS", "50"))
ROUNDS = int(os.environ.get("ROUNDS", "15"))


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _attnres_objs(dev):
    mk = lambda: types.SimpleNamespace(  # noqa: E731
        weight=(1 + 0.1 * torch.randn(H, device=dev)).bfloat16(), variance_epsilon=1e-5
    )
    proj = types.SimpleNamespace(
        weight=(torch.randn(1, H, device=dev) * 0.05).bfloat16()
    )
    return proj, mk(), mk()


def _two_graph(body, dev, tp_cpu):
    from vllm.distributed.parallel_state import graph_capture

    def build(n):
        g = torch.cuda.CUDAGraph()
        with graph_capture(dev) as ctx, torch.cuda.graph(g, stream=ctx.stream):
            for _ in range(n):
                body()
        return g

    for _ in range(3):
        body()
    torch.accelerator.synchronize()
    g1, g2 = build(ITERS), build(2 * ITERS)
    out = []
    for g in (g1, g2):
        for _ in range(3):
            g.replay()
        torch.accelerator.synchronize()
        best = 1e9
        for _ in range(ROUNDS):
            dist.barrier(group=tp_cpu)
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            g.replay()
            b.record()
            torch.accelerator.synchronize()
            best = min(best, a.elapsed_time(b) * 1e3)
        out.append(best)
    t = torch.tensor([(out[1] - out[0]) / ITERS])
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp_cpu)
    return t.item()


def _worker(rank, world, port, mode, which):
    dev = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(dev)
    with ensure_current_vllm_config():
        init_test_distributed_environment(world, 1, rank, port, local_rank=rank)
        import vllm.kernels  # noqa: F401
        from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
        from vllm.models.kimi_k3.amd import akr_runtime
        from vllm.models.kimi_k3.amd.linear import (
            _apply_attn_res,
            _apply_attn_res_partial,
        )

        cpu = get_tp_group().cpu_group
        akr_runtime.init_comms(
            attnres="attnres" in which,
            latent="latent" in which,
            hidden=H,
            up_rows=SHARD,
            latent_dim=LAT,
        )
        torch.manual_seed(1234)
        proj, norm1, norm2 = _attnres_objs(dev)
        rows = []
        if "attnres" in which:
            for m in M_LIST:
                torch.manual_seed(100 + rank)
                partial = (torch.randn(m, H, device=dev) * 0.3).bfloat16()
                torch.manual_seed(7)
                prefix0 = (torch.randn(m, H, device=dev) * 0.5).bfloat16()
                bank = (torch.randn(m, 8, H, device=dev) * 0.5).bfloat16()
                variants = {
                    "pre_attn": dict(nvb=4, widx=-1, onorm=norm2, has_prefix=True),
                    "block_write": dict(nvb=4, widx=4, onorm=norm2, has_prefix=True),
                    "pre_mlp_prefix_none": dict(
                        nvb=5, widx=-1, onorm=norm2, has_prefix=False
                    ),
                    "final": dict(nvb=8, widx=-1, onorm=None, has_prefix=True),
                }
                for name, v in variants.items():

                    def stock(pfx, bk):
                        delta = tensor_model_parallel_all_reduce(partial)
                        p = pfx if v["has_prefix"] else delta
                        d = delta if v["has_prefix"] else None
                        h = _apply_attn_res(
                            p,
                            bk,
                            proj,
                            norm1,
                            v["nvb"],
                            delta=d,
                            output_norm=v["onorm"],
                            block_write_idx=v["widx"],
                        )
                        return h, p

                    def fused(pfx, bk):
                        h, p = _apply_attn_res_partial(
                            pfx if v["has_prefix"] else None,
                            bk,
                            proj,
                            norm1,
                            v["nvb"],
                            partial=partial,
                            output_norm=v["onorm"],
                            block_write_idx=v["widx"],
                            return_prefix=True,
                        )
                        return h, p

                    if mode == "test":
                        pa, ba = prefix0.clone(), bank.clone()
                        pb, bb = prefix0.clone(), bank.clone()
                        ha, qa = stock(pa, ba)
                        hb, qb = fused(pb, bb)
                        torch.accelerator.synchronize()
                        e_h, e_p, e_b = _rel(hb, ha), _rel(qb, qa), _rel(bb, ba)
                        ok = e_h < 1.5e-2 and e_p < 4e-3 and e_b < 4e-3
                        if rank == 0:
                            print(
                                f"attnres M={m:3d} {name:20s} out {e_h:.2e} prefix "
                                f"{e_p:.2e} bank {e_b:.2e} "
                                f"{'OK' if ok else 'FAIL'}",
                                flush=True,
                            )
                        assert ok, (m, name, e_h, e_p, e_b)
                    elif name in ("pre_attn", "pre_mlp_prefix_none"):
                        pa, ba = (prefix0 * 0.01).clone(), bank.clone()
                        pb, bb = (prefix0 * 0.01).clone(), bank.clone()
                        ts = _two_graph(lambda: stock(pa, ba), dev, cpu)
                        tf = _two_graph(lambda: fused(pb, bb), dev, cpu)
                        rows.append(("attnres " + name, m, ts, tf))
        if "latent" in which:
            comm = akr_runtime.latent_comm
            norm_w = (1 + 0.1 * torch.randn(LAT, device=dev)).bfloat16()
            torch.manual_seed(55)
            up_w = (torch.randn(SHARD, LAT, device=dev) * 0.02).bfloat16()
            from amd_kernels.moe import latent_moe_common as ll

            # the experts' sort capacity routes by the experts' weight shapes (meta
            # views)
            meta = lambda *s: torch.empty(s, dtype=torch.uint8, device="meta")  # noqa: E731
            experts = ll.moe_weights(
                meta(896, 768, LAT // 2),
                meta(896, 768, LAT // 32),
                meta(896, LAT, 192),
                meta(896, LAT, 16),
            )

            def sort_capacity(m):
                return ll.moe_sort_capacity(m, weights=experts)

            for m in M_LIST:
                torch.manual_seed(300 + rank)
                partial = (torch.randn(m, LAT, device=dev) * 0.1).bfloat16()
                shared0 = (torch.randn(m, H, device=dev)).bfloat16()
                start = rank * SHARD

                def stock_latent(sh):
                    lat = tensor_model_parallel_all_reduce(partial)
                    torch.ops.vllm.k3_latent_norm_up_proj(
                        lat, norm_w, 1e-5, up_w, sh, start
                    )

                def fused_latent(sh):
                    ll.allreduce_latent_rmsnorm_up_proj(
                        comm,
                        partial,
                        norm_w,
                        up_w,
                        sh.narrow(1, start, SHARD),
                        None,
                        1e-5,
                    )

                if m <= 64:
                    cap = sort_capacity(m)
                    slots = (torch.randn(cap, LAT, device=dev) * 0.05).bfloat16()
                    rev = torch.randperm(cap, device=dev)[: m * 16].int().contiguous()
                    summed = (
                        slots[rev.long()]
                        .view(m, 16, LAT)
                        .float()
                        .sum(1)
                        .bfloat16()
                        .contiguous()
                    )

                    def stock_slots(sh):
                        lat = tensor_model_parallel_all_reduce(summed)
                        torch.ops.vllm.k3_latent_norm_up_proj(
                            lat, norm_w, 1e-5, up_w, sh, start
                        )

                    def fold(sh):
                        ll.allreduce_slots_rmsnorm_up_proj(
                            comm,
                            slots,
                            rev,
                            norm_w,
                            up_w,
                            sh.narrow(1, start, SHARD),
                            None,
                            1e-5,
                        )

                if mode == "test":
                    sa, sb = shared0.clone(), shared0.clone()
                    stock_latent(sa)
                    fused_latent(sb)
                    torch.accelerator.synchronize()
                    e = _rel(sb.narrow(1, start, SHARD), sa.narrow(1, start, SHARD))
                    keep = (
                        (sb.narrow(1, 0, start) == shared0.narrow(1, 0, start))
                        .all()
                        .item()
                    )
                    ok = e < 8e-3 and keep
                    if rank == 0:
                        print(
                            f"latent M={m:3d} partial entry rel {e:.2e} untouched cols "
                            f"{keep} "
                            f"{'OK' if ok else 'FAIL'}",
                            flush=True,
                        )
                    assert ok
                    if m <= 64:
                        sa, sb = shared0.clone(), shared0.clone()
                        stock_slots(sa)
                        fold(sb)
                        torch.accelerator.synchronize()
                        e = _rel(sb.narrow(1, start, SHARD), sa.narrow(1, start, SHARD))
                        ok = e < 8e-3
                        if rank == 0:
                            print(
                                f"latent M={m:3d} slot fold rel {e:.2e} "
                                f"{'OK' if ok else 'FAIL'}",
                                flush=True,
                            )
                        assert ok
                else:
                    sa, sb = shared0.clone(), shared0.clone()
                    ts = _two_graph(lambda: stock_latent(sa), dev, cpu)
                    tf = _two_graph(lambda: fused_latent(sb), dev, cpu)
                    rows.append(("latent partial", m, ts, tf))
                    if m <= 64:
                        ts = _two_graph(lambda: stock_slots(sa), dev, cpu)
                        tf = _two_graph(lambda: fold(sb), dev, cpu)
                        rows.append(("latent slots(sum+AR+tail)", m, ts, tf))
        if mode == "bench" and rank == 0:
            print(
                f"{'op':28s} {'M':>4s} {'stock us':>9s} {'akr us':>8s} {'speedup':>8s}"
            )
            for name, m, ts, tf in rows:
                print(
                    f"{name:28s} {m:4d} {ts:9.2f} {tf:8.2f} {ts / tf:8.2f}x", flush=True
                )
        dist.barrier(group=cpu)


if __name__ == "__main__":
    mode = sys.argv[1]
    which = (sys.argv[2] if len(sys.argv) > 2 else "attnres,latent").split(",")
    spawn(_worker, args=(8, str(get_open_port()), mode, which), nprocs=8, join=True)
