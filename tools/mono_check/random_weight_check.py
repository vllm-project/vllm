# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Random-weight check of the vendored, poll-bounded GLM MonoKernel
# (vllm/models/deepseek_v32/amd/mono/kernel) against FlyDSL's torch golden, TP8,
# vLLM-like config:
#
#   BF16 attention weights, fused paged BF16 KV cache [slots, 576] (KvCacheLayout.ATOM),
#   external (precomputed) sparse indices in CSR form (physical slot ids), native MXFP4
#   experts, q=1 per request, `batch` requests per launch.
#
# Golden: FlyDSL main's torch references (kernels.monokernel.glm.reference), run per row
# on a gathered logical per-request view of the paged cache. Every rank checks its own
# shard; TP reductions in the golden go through gloo all_gather.
#
# Usage (ROCm vLLM env with this branch; FlyDSL repo root on PYTHONPATH for the golden):
#   PYTHONPATH=<FlyDSL repo> python tools/mono_check/random_weight_check.py \
#       --npes 8 --batch 4 --samples 4 --ctx 100,2047,3000,5000
import argparse
import os
from dataclasses import replace

import torch

MAX_POS = 8192
POLL_LIMIT = int(os.environ.get("MONO_POLL_LIMIT", "1000000")) or None  # 0 -> unbounded
# MONO_FP8_ATTN=1: the kernel's FP8_BLOCK128 attention with the paged ATOM KV layout;
# golden = FlyDSL golden_layer with the same FP8 weights (dequantized).
FP8_ATTN = os.environ.get("MONO_FP8_ATTN", "0") == "1"
N_SLOTS = 65536


def _check(name, got, ref, report, rel_l2_tol):
    got, ref = got.float(), ref.float()
    finite = bool(torch.isfinite(got).all())
    err = (got - ref).abs()
    rel_l2 = (err.norm() / ref.norm().clamp_min(1e-30)).item()
    rel_max = (err.max() / ref.abs().max().clamp_min(1e-30)).item()
    ok = finite and rel_l2 <= rel_l2_tol
    report.append(
        f"   {name:9s} max_abs={err.max().item():.3e} "
        f"ref_max={ref.abs().max().item():.3e} "
        f"rel_max={rel_max:.2e} rel_l2={rel_l2:.2e} finite={finite} "
        f"{'ok' if ok else 'BAD'}"
    )
    return ok


def stage_opts(samples):
    """MONO_STAGE_OPTS=cache_hoist,split_keys64 (build-time stage options)."""
    keys = [k for k in os.environ.get("MONO_STAGE_OPTS", "").split(",") if k]
    return {k: True for k in keys if k != "split_keys64" or samples <= 8}


def run_rank(rank, npes, batch, samples, ctxs, iters, seed=1234, pad=0):
    from kernels.monokernel.config import (
        GLM5_CONFIG,
        KV_LORA,
        PE_DIM,
    )
    from kernels.monokernel.config import (
        AttentionWeight as FAW,
    )
    from kernels.monokernel.glm.reference import golden_layer, golden_moe, make_weights
    from kernels.monokernel.reference import dequant, fp8_mats, rope_table
    from kernels.monokernel.weights import LayerWeights as FlyLayerWeights

    from vllm.models.deepseek_v32.amd.mono.kernel.config import (
        AttentionWeight,
        KvCacheLayout,
        glm5_tp_config,
    )
    from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import Glm5MonoKernel
    from vllm.models.deepseek_v32.amd.mono.kernel.weights import (
        LayerWeights as AtomLayerWeights,
    )

    dev = torch.device("cuda", rank)
    torch.accelerator.set_device_index(dev)
    topk = 2048
    # Random weights: FlyDSL's generator (MXFP4 experts), attention dequantized to BF16.
    Wf = make_weights(rank, heads=8, device=dev, seed=seed, expert_mxfp4=True)
    t = dict(Wf.t)
    # FP8 mode feeds the generator's FP8 + block scales straight to the kernel
    if not FP8_ATTN:
        for name, (_, _, bk) in fp8_mats(8).items():
            q, s = t.pop(f"w_{name}"), t.pop(f"s_{name}")
            bm = 128
            t[f"w_{name}"] = dequant(q, s, bk, bm).to(torch.bfloat16).contiguous()
    W_ref = FlyLayerWeights(
        8,
        t,
        config=replace(
            GLM5_CONFIG, attention_weight=FAW.FP8_BLOCK128 if FP8_ATTN else FAW.BF16
        ),
    )
    W_atom = AtomLayerWeights(8, t, config=glm5_tp_config(npes), rank=rank, npes=npes)

    cos32, sin32 = rope_table(MAX_POS, device=dev)
    # ATOM copy reads BF16
    cos, sin = (
        cos32.to(torch.bfloat16).contiguous(),
        sin32.to(torch.bfloat16).contiguous(),
    )
    cosf, sinf = cos.float(), sin.float()

    gen = torch.Generator(device=dev).manual_seed(seed + 99)  # same on all ranks
    cache0 = torch.randn(N_SLOTS, KV_LORA + PE_DIM, generator=gen, device=dev).to(
        torch.bfloat16
    )
    group = None
    op = Glm5MonoKernel(
        W_atom,
        samples,
        rank=rank,
        npes=npes,
        group=group,
        topk=topk,
        attention_weight=AttentionWeight.FP8_BLOCK128
        if FP8_ATTN
        else AttentionWeight.BF16,
        kv_cache_layout=KvCacheLayout.ATOM,
        kv_cache_dtype="bf16",
        poll_limit=POLL_LIMIT,
        **stage_opts(samples),
    )
    if npes == 1:
        allreduce = lambda x: x  # noqa: E731
    else:
        import torch.distributed as dist

        def allreduce(x):
            parts = [torch.empty_like(x.cpu()) for _ in range(npes)]
            dist.all_gather(parts, x.cpu().contiguous(), group=group)
            return sum(parts[1:], parts[0]).to(x.device)

    # Warmup launch (JIT compile / cache load) on a scratch cache with every row
    # inactive, then a barrier: per-rank compile skew otherwise exceeds the poll bound
    # on the first real launch.
    _wc = torch.zeros(64, KV_LORA + PE_DIM, dtype=torch.bfloat16, device=dev)
    _z64 = torch.zeros(batch, dtype=torch.int64, device=dev)
    op.forward(
        torch.zeros(batch, 6144, dtype=torch.bfloat16, device=dev),
        torch.zeros(1, dtype=torch.int32, device=dev),
        _wc,
        _wc,
        torch.zeros(1, dtype=torch.int32, device=dev),
        cos,
        sin,
        positions=_z64,
        slot_mapping=_z64 - 1,
        sparse_kv_indptr=torch.zeros(batch + 1, dtype=torch.int32, device=dev),
    )
    torch.accelerator.synchronize()
    op.poll_error()
    if npes > 1:
        import torch.distributed as dist

        dist.barrier()
    ok_all = True
    for it in range(iters):
        # page_size=1 paging: each request owns a random set of physical slots.
        perm = torch.randperm(N_SLOTS, generator=gen, device=dev)
        rows, positions, slots, indptr, flat = [], [], [], [0], []
        off = 0
        for r in range(batch):
            L = ctxs[r % len(ctxs)] + it  # context incl. the new token
            req_slots = perm[off : off + L]
            off += L
            p = L - 1
            if topk >= L:
                sel_pos = torch.arange(L, device=dev)
            else:
                sel_pos = torch.randperm(p, generator=gen, device=dev)[: topk - 1]
                sel_pos = (
                    torch.cat([sel_pos, torch.tensor([p], device=dev)]).sort().values
                )
            if r >= batch - pad:  # CUDA-graph padding row: slot -1, empty sparse range
                positions.append(0)
                slots.append(-1)
                indptr.append(indptr[-1])
                continue
            rows.append((req_slots, p, sel_pos))
            positions.append(p)
            slots.append(int(req_slots[p]))
            flat.append(req_slots[sel_pos].to(torch.int32))
            indptr.append(indptr[-1] + len(sel_pos))
        positions_t = torch.tensor(positions, dtype=torch.int64, device=dev)
        slots_t = torch.tensor(slots, dtype=torch.int64, device=dev)
        indptr_t = torch.tensor(indptr, dtype=torch.int32, device=dev)
        flat_t = torch.cat(flat).contiguous()
        h = torch.randn(batch, 6144, generator=gen, device=dev).to(torch.bfloat16)
        cache = cache0.clone()
        dummy_pos = torch.zeros(1, dtype=torch.int32, device=dev)
        torch.accelerator.synchronize()
        _e0, _e1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        _e0.record()
        out = op.forward(
            h,
            dummy_pos,
            cache,
            cache,
            flat_t,
            cos,
            sin,
            positions=positions_t,
            slot_mapping=slots_t,
            sparse_kv_indptr=indptr_t,
        )
        _e1.record()
        torch.accelerator.synchronize()
        launch_ms = _e0.elapsed_time(_e1)
        expired = op.poll_error()
        got = op.intermediates()  # last chunk only
        last0 = batch - samples
        report = [
            f"[rank {rank} iter {it}] batch={batch} S={samples} "
            f"ctx={[p + 1 for p in positions]}"
        ]
        ok = bool(torch.isfinite(out).all()) and not expired
        report[0] += f" poll_expired={list(expired)} launch_ms={launch_ms:.3f}"
        # Per-row golden on a gathered logical view.
        ref_rows = []
        for r, (req_slots, p, sel_pos) in enumerate(rows):
            kv_view = cache0[req_slots, :KV_LORA].clone()
            pe_view = cache0[req_slots, KV_LORA:].clone()
            idx = torch.zeros(1, topk, dtype=torch.int32, device=dev)
            idx[0, : len(sel_pos)] = sel_pos.to(torch.int32)
            ref = golden_layer(
                W_ref,
                h[r : r + 1],
                p,
                kv_view,
                pe_view,
                idx,
                cosf,
                sinf,
                allreduce,
                topk=topk,
            )
            ref["kv_row"] = torch.cat([kv_view[p], pe_view[p]])
            ref_rows.append(ref)
        cat = lambda k, rs: torch.cat([x[k] for x in rs])  # noqa: E731
        if pad:
            nact = batch - pad
            ok &= _check("x_out_e2e", out[:nact], cat("x_out", ref_rows), report, 5e-2)
            live = slots_t >= 0
            ok &= _check(
                "kv_write",
                cache[slots_t[live]],
                torch.stack([x["kv_row"] for x in ref_rows]),
                report,
                1e-2,
            )
            mask = torch.ones(N_SLOTS, dtype=torch.bool, device=dev)
            mask[slots_t[live]] = False
            untouched = torch.equal(cache[mask], cache0[mask])
            report.append(
                f"   padded_rows={pad} "
                f"out_finite_all_rows={bool(torch.isfinite(out).all())} "
                f"other_cache_rows_untouched={untouched}"
            )
            ok &= untouched
            report[0] += f" ok={ok}"
            if rank == 0 or not ok:
                print("\n".join(report), flush=True)
            ok_all &= ok
            continue
        last = ref_rows[last0:]
        for name in ("q_a", "kv_a", "q_nope", "q_pe", "q_lat", "o", "a"):
            ok &= _check(name, got[name], cat(name, last), report, 2e-2)
        moe = golden_moe(W_ref, got["a"].clone(), allreduce)
        ok &= _check("scores", got["scores"], moe["scores"], report, 1e-4)
        sel_ok = torch.equal(got["sel"], moe["sel"])
        report.append(f"   sel_equal={sel_ok}")
        ok &= sel_ok
        down = golden_moe(
            W_ref,
            got["a"].clone(),
            allreduce,
            got["mid"].clone(),
            got["sel"].clone(),
            got["prob"].clone(),
        )
        ok &= _check("x_out|own", out[last0:], down["x_out"], report, 1e-2)
        ok &= _check("x_out_e2e", out, cat("x_out", ref_rows), report, 5e-2)
        new_rows = cache[slots_t]
        ok &= _check(
            "kv_write",
            new_rows,
            torch.stack([x["kv_row"] for x in ref_rows]),
            report,
            1e-2,
        )
        # nothing else in the cache changed
        mask = torch.ones(N_SLOTS, dtype=torch.bool, device=dev)
        mask[slots_t] = False
        untouched = torch.equal(cache[mask], cache0[mask])
        report.append(f"   other_cache_rows_untouched={untouched}")
        ok &= untouched
        report[0] += f" ok={ok}"
        if rank == 0 or not ok:
            print("\n".join(report), flush=True)
        ok_all &= ok
    op.close()
    return ok_all


def bench_rank(rank, npes, batch, samples, ctx, iters=2000, seed=1234):
    """HIP-graph replay of 16 layer launches (same op, one step bump each) -> us per
    layer."""
    import torch.distributed as dist
    from kernels.monokernel.glm.reference import make_weights
    from kernels.monokernel.reference import dequant, fp8_mats, rope_table

    from vllm.models.deepseek_v32.amd.mono.kernel.config import (
        AttentionWeight,
        KvCacheLayout,
        glm5_tp_config,
    )
    from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import Glm5MonoKernel
    from vllm.models.deepseek_v32.amd.mono.kernel.weights import (
        LayerWeights as AtomLayerWeights,
    )

    dev = torch.device("cuda", rank)
    torch.accelerator.set_device_index(dev)
    Wf = make_weights(rank, heads=8, device=dev, seed=seed, expert_mxfp4=True)
    t = dict(Wf.t)
    if not FP8_ATTN:
        for name, (_, _, bk) in fp8_mats(8).items():
            q, s = t.pop(f"w_{name}"), t.pop(f"s_{name}")
            t[f"w_{name}"] = dequant(q, s, bk, 128).to(torch.bfloat16).contiguous()
    W_atom = AtomLayerWeights(8, t, config=glm5_tp_config(npes), rank=rank, npes=npes)
    cos, sin = (
        x.to(torch.bfloat16).contiguous() for x in rope_table(MAX_POS, device=dev)
    )
    cache = torch.randn(N_SLOTS, 576, device=dev).to(torch.bfloat16)
    # MONO_TIMELINE=1 builds the stage-timeline kernel (s_memrealtime stamps per task)
    # and prints op.timeline_report() of one extra launch after the timed replays
    timeline = os.environ.get("MONO_TIMELINE") == "1"
    extra = stage_opts(samples)
    op = Glm5MonoKernel(
        W_atom,
        samples,
        rank=rank,
        npes=npes,
        topk=2048,
        attention_weight=AttentionWeight.FP8_BLOCK128
        if FP8_ATTN
        else AttentionWeight.BF16,
        kv_cache_layout=KvCacheLayout.ATOM,
        kv_cache_dtype="bf16",
        poll_limit=POLL_LIMIT,
        timeline=timeline,
        **extra,
    )
    perm = torch.randperm(N_SLOTS, device=dev)
    nk = min(ctx, 2048)
    flat = torch.cat([perm[r * ctx : r * ctx + nk] for r in range(batch)]).to(
        torch.int32
    )
    indptr = torch.arange(0, (batch + 1) * nk, nk, dtype=torch.int32, device=dev)
    positions = torch.full((batch,), ctx - 1, dtype=torch.int64, device=dev)
    slots = torch.tensor(
        [int(perm[r * ctx + ctx - 1]) for r in range(batch)],
        dtype=torch.int64,
        device=dev,
    )
    h = torch.randn(batch, 6144, device=dev).to(torch.bfloat16)
    x = torch.empty_like(h)
    dpos = torch.zeros(1, dtype=torch.int32, device=dev)
    fwd = lambda: op.forward(
        h,
        dpos,
        cache,
        cache,
        flat,
        cos,
        sin,
        x_out=x,
        positions=positions,  # noqa: E731
        slot_mapping=slots,
        sparse_kv_indptr=indptr,
    )
    for _ in range(10):
        fwd()
    torch.accelerator.synchronize()
    dist.barrier()
    L = 16
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(L):
            fwd()
    for _ in range(30):
        g.replay()
    torch.accelerator.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n = iters // L
    e0.record()
    for _ in range(n):
        g.replay()
    e1.record()
    torch.accelerator.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / (n * L)
    # steady state: the stamps of the LAST layer of the last timed
    # graph replay -- an extra eager launch after a host barrier puts cross-rank
    # launch skew into the all-reduce waits and shifts later stages
    if timeline and rank == 0:
        print(
            f"[timeline-steady S={samples} ctx={ctx} extra={extra}]\n"
            + op.timeline_report(),
            flush=True,
        )
        out = os.environ.get("MONO_TIMELINE_OUT")
        if out:
            os.makedirs(out, exist_ok=True)
            tag = "_".join(f"{k}{v}" for k, v in sorted(extra.items())) or "default"
            torch.save(
                dict(timeline=op.timeline.cpu(), stages=list(op.stages)),
                f"{out}/raw_s{samples}_{tag}.pt",
            )
    op.close()
    return us


def _worker(rank, npes, batch, samples, ctxs, iters, bench, results, pad=0):
    import torch.distributed as dist

    port = os.environ.get("GLM5_MASTER_PORT", "29551")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes
    )
    if bench:
        us = bench_rank(rank, npes, batch, samples, ctxs[0])
        print(f"[rank {rank}] {us:.1f} us/layer", flush=True)
        results[rank] = True
    else:
        results[rank] = run_rank(rank, npes, batch, samples, ctxs, iters, pad=pad)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--samples", type=int, default=1, help="kernel chunk size (rows per launch)"
    )
    ap.add_argument(
        "--ctx", default="3000", help="comma list of context lengths per request"
    )
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument(
        "--pad",
        type=int,
        default=0,
        help="last N rows are graph padding (slot -1, empty range)",
    )
    a = ap.parse_args()
    ctxs = [int(c) for c in a.ctx.split(",")]
    import torch.multiprocessing as mp

    mgr = mp.Manager()
    res = mgr.dict()
    mp.spawn(
        _worker,
        args=(a.npes, a.batch, a.samples, ctxs, a.iters, a.bench, res, a.pad),
        nprocs=a.npes,
    )
    print("PASS" if all(res[r] for r in range(a.npes)) else "FAIL")
