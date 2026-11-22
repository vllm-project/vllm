# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""8-rank check of the static expert placement (akr_eplb) through the vLLM
MegaMoE experts class: one layer loaded twice from the same logical experts,
(a) identity placement (48 per rank), (b) the plan's placement (S slots per
rank, replicas), each through `DeepseekV4AiterMegaMoEExperts.weight_loader`,
`finalize_weights` and `forward` (id remap, then the fused EP8 route and the
staged AITER MegaMoEV2), eager and under CUDA graph replay. Both hold the same
logical weights, so the outputs may differ only by the summation order of the
routes (cosine >= 0.99999 required; bitwise equality reported). Also checks
that S = 49 runs AITER's tuned EP8 MegaMoE profile.

Not a pytest module (needs 8 GPUs, torchrun, mori, amd_kernels):

    torchrun --nproc-per-node 8 tests/models/deepseek_v41/amd/eplb_gpu8.py \\
        [plan.npz (default: the shipped S=49 plan)] [--layer 20] [--draft]
        [--no-fused] [--no-staged] [--max-num-batched-tokens 32768]

`--draft`: the DSpark draft shape (128 experts, top 3, staged only).
`--max-num-batched-tokens`: the scheduler budget that sizes the MegaMoE
capacity (default 32768, the serving value). The staged cases include ranks
with few local rows under a large DP-group maximum (the runtime is picked by
the DP maximum) and a CUDA graph replay of remap + staged MegaMoE (the draft's
decode path). Rank 0 prints the summary; the exit code is 1 on any mismatch.
"""

import argparse
import os
import sys
import types

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "64G")

import mori.shmem as ms  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import nn  # noqa: E402

H, INTER, WORLD = 5120, 2304, 8


def make_experts(n: int, seed: int, dev):
    """Random MXFP4 experts, row-major uint8 (w13, w13 e8m0, w2, w2 e8m0)."""
    g = torch.Generator(device=dev).manual_seed(seed)

    def rnd(lo, hi, shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device=dev, generator=g)

    return (
        rnd(0, 256, (n, 2 * INTER, H // 2)),
        rnd(118, 124, (n, 2 * INTER, H // 32)),
        rnd(0, 256, (n, H, INTER // 2)),
        rnd(118, 124, (n, H, INTER // 32)),
    )


def load(experts, raw, n_logical: int) -> int:
    """Every checkpoint expert through the weight loader, as the model's
    load_weights does; returns how many (expert, shard) calls succeeded."""
    w13, s13, w2, s2 = raw
    ok = 0
    for g in range(n_logical):
        for param, shard, t in (
            (experts.w13_weight, "w1", w13[g, :INTER]),
            (experts.w13_weight, "w3", w13[g, INTER:]),
            (experts.w13_weight_scale, "w1", s13[g, :INTER]),
            (experts.w13_weight_scale, "w3", s13[g, INTER:]),
            (experts.w2_weight, "w2", w2[g]),
            (experts.w2_weight_scale, "w2", s2[g]),
        ):
            ok += bool(
                experts.weight_loader(param, t, "w", shard, g, return_success=True)
            )
    return ok


def main() -> None:
    from vllm.models.deepseek_v41.amd import akr_eplb

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "plan",
        nargs="?",
        default=os.path.join(akr_eplb.CONFIG_DIR, "expert_plan_S49.npz"),
    )
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--no-fused", action="store_true")
    ap.add_argument("--no-staged", action="store_true")
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    args = ap.parse_args()
    draft = args.draft
    n_exp, topk = (128, 3) if draft else (384, 6)
    layer = args.layer if args.layer is not None else (41 if draft else 20)
    fused = not draft and not args.no_fused

    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=dev)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")

    import vllm.distributed
    from vllm.models.deepseek_v4.amd import mega_moe
    from vllm.models.deepseek_v41.amd import akr_config, akr_experts

    ep = types.SimpleNamespace(
        rank_in_group=rank,
        world_size=WORLD,
        cpu_group=dist.group.WORLD,
        device_group=dist.group.WORLD,
    )
    mega_moe.get_ep_group = lambda: ep
    vllm.distributed.get_ep_group = lambda: ep
    rows_max = {"m": 0}
    akr_config.dp_rows = lambda m: max(m, rows_max["m"])
    # the staged runtime is picked by the DP-group maximum, as under serving
    dp_md = types.SimpleNamespace(
        dp_metadata=types.SimpleNamespace(num_tokens_across_dp_cpu=torch.zeros(1))
    )

    def forward_context():
        dp_md.dp_metadata.num_tokens_across_dp_cpu = torch.tensor([rows_max["m"]])
        return dp_md

    mega_moe.is_forward_context_available = lambda: True
    mega_moe.get_forward_context = forward_context

    def build(path: str):
        cfg = types.SimpleNamespace(
            kernel_config=types.SimpleNamespace(
                amd_kernels=types.SimpleNamespace(expert_perm=path)
            ),
            scheduler_config=types.SimpleNamespace(
                max_num_batched_tokens=args.max_num_batched_tokens
            ),
        )
        # parameters on the GPU, as under the model loader's device context
        with torch.device(dev):
            return mega_moe.DeepseekV4AiterMegaMoEExperts(
                cfg,
                num_experts=n_exp,
                top_k=topk,
                hidden_size=H,
                intermediate_size=INTER,
                swiglu_limit=10.0,
                prefix=f"model.layers.{layer}.ffn.experts",
            )

    # every rank makes the same logical experts (seed per expert block)
    raw = [
        torch.cat(t)
        for t in zip(*(make_experts(16, 100 + b, dev) for b in range(n_exp // 16)))
    ]
    ident, placed = build(""), build(args.plan)
    assert ident._perm is None and placed._perm is not None
    S = placed.num_routed_local_experts
    loaded = (load(ident, raw, n_exp), load(placed, raw, n_exp))
    assert loaded[0] == 6 * (n_exp // WORLD), loaded
    want = sum(bool(placed._perm.local_slots(g, rank * S, S)) for g in range(n_exp))
    assert loaded[1] == 6 * want, (loaded, want)
    ident.finalize_weights()
    placed.finalize_weights()
    rep: list[str] = []
    errors: list[str] = []
    lp = placed._perm
    if rank == 0:
        rep.append(
            f"plan {args.plan} layer {layer}: S={S}, max replicas {lp.max_rep}, "
            f"{int((lp.count > 1).sum())} replicated experts; runtimes "
            f"identity {[c for c, _ in ident._runtimes]} placed "
            f"{[c for c, _ in placed._runtimes]}"
        )
    # S = 49 must run AITER's tuned EP8 profile: same configs as epr 48 at the
    # same capacity (not fixed-slot)
    if S == 49 and not draft:
        from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config as mc

        cap = placed._runtimes[-1][0]
        prof = mc._match_network(49, H, INTER)
        p49 = mc.build_mega_moe_bundle_plan(
            cap, experts_per_rank=49, model_dim=H, inter_dim=INTER, world_size=8
        )
        p48 = mc.build_mega_moe_bundle_plan(
            cap, experts_per_rank=48, model_dim=H, inter_dim=INTER, world_size=8
        )
        same = [a.config == b.config for a, b in zip(p49.entries, p48.entries)]
        rt_plan = placed._runtimes[-1][1]._bundle_plan
        if prof is None or not all(same) or rt_plan.entries != p49.entries:
            errors.append(f"rank {rank}: S=49 MegaMoE not on the tuned profile")
        elif rank == 0:
            rep.append(
                f"S=49 MegaMoE profile {prof.name}: {len(same)} buckets equal to "
                f"epr 48 at capacity {cap}"
            )

    def inputs(m: int, seed: int, kind: str):
        g = torch.Generator(device=dev).manual_seed(31 * seed + rank)
        gum = -torch.log(
            -torch.log(
                torch.rand((max(m, 1), n_exp), device=dev, generator=g).clamp_min(1e-20)
            )
        )
        if kind == "skew":  # hot experts on the plan's replicated ids
            hot = torch.from_numpy(lp.count > 1).to(dev)
            gum = gum + 2.5 * hot
        ids = gum.topk(topk, dim=1).indices.int()
        tw = torch.softmax(torch.randn((max(m, 1), topk), device=dev, generator=g), -1)
        x = torch.randn((max(m, 1), H), device=dev, generator=g).bfloat16()
        return ids[:m].contiguous(), tw[:m].contiguous(), x[:m].contiguous()

    def cmp(tag: str, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.numel() == 0:
            return
        cos = torch.nn.functional.cosine_similarity(
            a.double().flatten(), b.double().flatten(), dim=0
        ).item()
        rep.append(
            f"{tag}: cos(identity, placed)={cos:.8f} bitwise={torch.equal(a, b)}"
        )
        if not cos >= 0.99999 or not torch.isfinite(b.float()).all():
            errors.append(f"rank {rank} {tag}: cos {cos}")

    def run(experts, x, tw, ids):
        out = experts.forward(x, tw, ids, None)
        torch.accelerator.synchronize()
        return out

    if fused:
        ci, cp = nn.Module(), nn.Module()
        ci.add_module("experts", ident)
        cp.add_module("experts", placed)
        akr_experts.post_load(ci)
        akr_experts.post_load(cp)
        assert ident.akr is not None and placed.akr is not None
        assert placed.akr.ep is not ident.akr.ep and placed.akr.ep.epr == S
        cases = [
            ([16] * 8, "uniform"),
            ([64] * 8, "skew"),
            ([192] * 8, "skew"),
            ([848] * 8, "uniform"),
            ([1024] * 8, "skew"),
            ([0, 5, 300, 64, 1, 0, 128, 33], "skew"),
        ]
        for i, (counts, kind) in enumerate(cases):
            m = counts[rank]
            rows_max["m"] = max(counts)
            ids, tw, x = inputs(m, 10 + i, kind)
            a, b = run(ident, x, tw, ids), run(placed, x, tw, ids)
            cmp(f"fused rows {counts} {kind}", a, b)
            if m:
                from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8 import (  # noqa: E501
                    reference as ref,
                )

                k = min(m, 16)
                q, s = ref.quantize_mxfp8(x[:k])
                per_rank = [
                    tuple(t[r * 48 : (r + 1) * 48] for t in raw) for r in range(8)
                ]
                want = ref.ep_reference(q, s, per_rank, tw[:k], ids[:k])
                cw = torch.nn.functional.cosine_similarity(
                    b[:k].double().flatten(), want.double().flatten(), dim=0
                ).item()
                rep.append(f"fused rows {counts}: cos(placed, reference)={cw:.8f}")
                if not cw >= 0.99999:
                    errors.append(f"rank {rank} fused {counts}: reference cos {cw}")
        # remap + kernel captured in one graph, replayed on new routes
        m = 64
        rows_max["m"] = m
        ids, tw, x = inputs(m, 99, "uniform")
        run(placed, x, tw, ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = placed.forward(x, tw, ids, None)
        for seed in (1, 2):
            nids, ntw, nx = inputs(m, 200 + seed, "skew")
            x.copy_(nx)
            ids.copy_(nids)
            tw.copy_(ntw)
            graph.replay()
            torch.accelerator.synchronize()
            eager = run(placed, x, tw, ids)
            if not torch.equal(gout, eager):
                errors.append(f"rank {rank}: graph replay differs from eager")
            cmp(f"fused graph replay {seed}", run(ident, x, tw, ids), gout.clone())
        ident.akr = placed.akr = None
    if not args.no_staged:
        for i, (m, dp_max, kind) in enumerate(
            [
                (1, 0, "uniform"),
                (16, 0, "uniform"),
                (192, 0, "skew"),
                (768, 0, "uniform"),
                (2048, 0, "skew"),
                (4096, 0, "skew"),
                # few local rows under a large DP-group maximum
                (1 if rank % 2 else 3, 2048, "uniform"),
                (16, 4096, "skew"),
                (192, 1536, "skew"),
            ]
        ):
            rows_max["m"] = dp_max
            ids, tw, x = inputs(m, i, kind)
            b = run(placed, x, tw, ids)
            pick = max(m, dp_max)
            cap_i = next(c for c, _ in ident._runtimes if pick <= c)
            cap_p = next(c for c, _ in placed._runtimes if pick <= c)
            if cap_i != cap_p:
                # AITER's fixed-slot dispatch (<= 255 tokens per rank) exists
                # for 48 experts per rank only: S != 48 runs the indexed
                # MegaMoE at the larger capacity, another kernel (FP8 P2P);
                # compare against identity on that same runtime
                a = run(ident, x, tw, ids)
                cos = torch.nn.functional.cosine_similarity(
                    a.double().flatten(), b.double().flatten(), dim=0
                ).item()
                rep.append(
                    f"staged rows {m} {kind}: identity fixed-slot cap {cap_i} vs "
                    f"placed cap {cap_p}: cos {cos:.6f} (kernel difference, info)"
                )
                saved = ident._runtimes
                ident._runtimes = [r for r in saved if r[0] == cap_p]
                try:
                    a = run(ident, x, tw, ids)
                finally:
                    ident._runtimes = saved
            else:
                a = run(ident, x, tw, ids)
            cmp(f"staged rows {m} dp-max {dp_max} {kind} (capacity {cap_p})", a, b)
        # remap + staged MegaMoE captured in one graph, replayed on new routes
        m = 64
        rows_max["m"] = 1024  # one runtime for both placements
        ids, tw, x = inputs(m, 98, "uniform")
        run(placed, x, tw, ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = placed.forward(x, tw, ids, None)
        for seed in (1, 2):
            nids, ntw, nx = inputs(m, 300 + seed, "skew")
            x.copy_(nx)
            ids.copy_(nids)
            tw.copy_(ntw)
            graph.replay()
            torch.accelerator.synchronize()
            if not torch.equal(gout, run(placed, x, tw, ids)):
                errors.append(f"rank {rank}: staged graph replay differs from eager")
            cmp(f"staged graph replay {seed}", run(ident, x, tw, ids), gout.clone())
    all_err: list = [None] * WORLD
    dist.all_gather_object(all_err, errors)
    if rank == 0:
        print("\n".join(rep))
        flat = [e for r in all_err for e in r]
        print("ERRORS:" if flat else "ALL OK", *flat, sep="\n")
    sys.stdout.flush()
    os._exit(1 if any(all_err) else 0)


if __name__ == "__main__":
    main()
