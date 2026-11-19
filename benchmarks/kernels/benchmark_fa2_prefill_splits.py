# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FA2 paged prefill: `num_splits=0` vs `num_splits=1`, with FlashInfer as reference.

FA2 does not split KV for prefill on the paged varlen path, so `num_splits`
only selects the kernel tile:

| head dim | `num_splits=0`          | `num_splits=1`          |
|----------|-------------------------|-------------------------|
| 64       | 64x256 tile, 72 KB smem | 64x128 tile, 40 KB smem |
| 128      | 64x128 tile, 80 KB smem | 64x64 tile, 48 KB smem  |
| 256      | 64x64 tile, 96 KB smem  | same tile (control)     |

On GPUs with 100 KB of shared memory per SM (SM86/89/120/121), the
`num_splits=0` tile fits 1 CTA/SM and the `num_splits=1` tile fits 2.

Each arm is checked against FA2 on contiguous K/V, then timed with FlashInfer
CUPTI (CUDA graphs, cold L2). Arms are interleaved over rounds. The tile of
each FA2 arm is parsed from the kernel name in a torch profile.

Usage:
    python benchmarks/kernels/benchmark_fa2_prefill_splits.py
    python benchmarks/kernels/benchmark_fa2_prefill_splits.py --quick
    python benchmarks/kernels/benchmark_fa2_prefill_splits.py --no-flashinfer
    python benchmarks/kernels/benchmark_fa2_prefill_splits.py --out results.json

`--quick` runs head dim 128 at page size 16 only. `--no-flashinfer` skips the
FlashInfer arm, which needs nvcc and ninja to JIT-compile.
"""

import argparse
import json
import os
import platform
import statistics
import time
import warnings

import regex as re

# To suppress statically Defined Tracing events  (profiler_start / profiler_stop)
os.environ["KINETO_LOG_LEVEL"] = "6"

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func

# (name, num_seqs, query_len, context_len).
CASES = [
    ("8x512", 8, 512, 512),
    ("16x1K", 16, 1024, 1024),
    ("4x2K", 4, 2048, 2048),
    ("1x8K", 1, 8192, 8192),
    ("2K@32K", 1, 2048, 32768),
    ("8K@16K", 1, 8192, 16384),
]
# (label, q_heads, kv_heads, head_dim)
SHAPES = {
    "hd128": ("32q/8kv d128 (Llama-8B, Qwen3-4B/8B)", 32, 8, 128),
    "hd64": ("64q/8kv d64 (gpt-oss-like)", 64, 8, 64),
    "hd256": ("16q/8kv d256 (Gemma-like; control)", 16, 8, 256),
}
TRAITS_RE = re.compile(r"Flash_fwd_kernel_traits<(\d+), (\d+), (\d+), (\d+)")


def metadata():
    import vllm

    p = torch.cuda.get_device_properties(0)
    meta = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "smem_per_sm": p.shared_memory_per_multiprocessor,
        "regs_per_sm": p.regs_per_multiprocessor,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    try:
        import flashinfer

        meta["flashinfer"] = flashinfer.__version__
    except Exception as e:
        meta["flashinfer"] = f"unavailable: {e}"
    return meta


def attended_pairs(q, c):
    return q * (c - q) + q * (q + 1) // 2


def make_timer():
    from flashinfer.testing import bench_gpu_time_with_cupti

    def cupti(fn):
        return bench_gpu_time_with_cupti(fn, use_cuda_graph=True)

    # Without cupti-python, FlashInfer only warns and falls back to CUDA events
    # with a warm L2 (these closures give it no tensors to rotate).
    x = torch.zeros(1 << 20, device="cuda")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cupti(lambda: x.add_(1))
    fallback = [str(w.message) for w in caught if "CUPTI" in str(w.message)]
    if fallback:
        raise SystemExit(
            f"CUPTI unavailable: {fallback[0]}\n"
            "Install it with: uv pip install 'cupti-python>=13'"
        )
    return "cupti", cupti


def kernel_info(fn, smem_per_sm):
    """Capture the FA2 forward kernel's tile from its template name."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize()
    names = {e.name for e in prof.events() if "flash_fwd" in e.name}
    for n in names:
        m = TRAITS_RE.search(n)
        if m and "combine" not in n:
            d, bm, bn, warps = map(int, m.groups())
            smem = (bm * d + 2 * bn * d) * 2  # Q + K + V tiles, bf16
            return {
                "kernel": n.split("<")[0].split("(")[0].split()[-1],
                "kBlockM": bm,
                "kBlockN": bn,
                "warps": warps,
                "est_smem_bytes": smem,
                # Upper bound from smem alone (registers may limit further);
                # the driver reserves 1 KB per block.
                "est_ctas_per_sm_by_smem": smem_per_sm // (smem + 1024),
            }
    return {"kernel": sorted(names)[:2] or "not found"}


def run_case(shape_key, page, case, use_fi, fi_ws, timer, rounds, meta):
    _, hq, hkv, d = SHAPES[shape_key]
    name, b, ql, cl = case
    scale = d**-0.5
    torch.manual_seed(0)
    q = torch.randn(b * ql, hq, d, dtype=torch.bfloat16)
    k = torch.randn(b * cl, hkv, d, dtype=torch.bfloat16)
    v = torch.randn(b * cl, hkv, d, dtype=torch.bfloat16)
    cu_q = torch.arange(0, b + 1, dtype=torch.int32) * ql
    cu_k = torch.arange(0, b + 1, dtype=torch.int32) * cl

    pages_per_seq = cl // page
    n_pages = b * pages_per_seq
    block_table = torch.randperm(n_pages, dtype=torch.int32).view(b, pages_per_seq)
    k_cache = torch.empty(n_pages, page, hkv, d, dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    k_cache[block_table.flatten().long()] = k.view(n_pages, page, hkv, d)
    v_cache[block_table.flatten().long()] = v.view(n_pages, page, hkv, d)
    seqused_k = torch.full((b,), cl, dtype=torch.int32)

    ref = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=ref,
        cu_seqlens_q=cu_q,
        max_seqlen_q=ql,
        cu_seqlens_k=cu_k,
        max_seqlen_k=cl,
        softmax_scale=scale,
        causal=True,
        fa_version=2,
    )

    outs = {}
    arms = {}

    def fa_paged(ns):
        out = torch.empty_like(q)
        outs[f"fa2_s{ns}"] = out

        def run():
            flash_attn_varlen_func(
                q=q,
                k=k_cache,
                v=v_cache,
                out=out,
                cu_seqlens_q=cu_q,
                max_seqlen_q=ql,
                seqused_k=seqused_k,
                max_seqlen_k=cl,
                softmax_scale=scale,
                causal=True,
                block_table=block_table,
                num_splits=ns,
                fa_version=2,
            )

        return run

    arms["fa2_s0"] = fa_paged(0)
    arms["fa2_s1"] = fa_paged(1)

    fi_error = None
    if use_fi:
        try:
            import flashinfer

            out_fi = torch.empty_like(q)
            outs["fi"] = out_fi
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(fi_ws, "NHD")
            wrapper.plan(
                cu_q,
                torch.arange(0, b + 1, dtype=torch.int32) * pages_per_seq,
                block_table.flatten(),
                torch.full((b,), page, dtype=torch.int32),
                hq,
                hkv,
                d,
                page,
                causal=True,
                sm_scale=scale,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
            )
            arms["fi"] = lambda: wrapper.run(q, (k_cache, v_cache), out=out_fi)
            arms["fi"]()
        except Exception as e:  # noqa: BLE001
            fi_error = f"{type(e).__name__}: {str(e)[:300]}"
            arms.pop("fi", None)
            outs.pop("fi", None)

    for fn in arms.values():
        fn()
    torch.accelerator.synchronize()
    max_err = {}
    for arm, out in outs.items():
        max_err[arm] = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    tiles = {
        arm: kernel_info(arms[arm], meta["smem_per_sm"]) for arm in ("fa2_s0", "fa2_s1")
    }

    for fn in arms.values():  # warm up clocks and JIT before timing
        for _ in range(10):
            fn()
    torch.accelerator.synchronize()

    per_round = {arm: [] for arm in arms}
    order = list(arms)
    for r in range(rounds):
        for arm in order if r % 2 == 0 else reversed(order):
            per_round[arm].append(statistics.median(timer(arms[arm])) * 1e3)

    flops = 4 * hq * d * b * attended_pairs(ql, cl)
    us = {arm: statistics.median(v) for arm, v in per_round.items()}
    return {
        "shape": shape_key,
        "page": page,
        "case": name,
        "num_seqs": b,
        "query_len": ql,
        "context_len": cl,
        "flops": flops,
        "us": us,
        "us_rounds": per_round,
        "tflops": {arm: flops / (t * 1e6) for arm, t in us.items()},
        "max_abs_err_vs_contig": max_err,
        "tiles": tiles,
        "fi_error": fi_error,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument(
        "--pages", default="16,64", help="KV block sizes; vLLM's CUDA default is 16"
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-flashinfer", action="store_true")
    ap.add_argument("--quick", action="store_true", help="hd128, page 16 only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.quick:
        args.shapes, args.pages = "hd128", "16"

    torch.set_default_device("cuda")
    meta = metadata()
    timer_name, timer = make_timer()
    meta["timer"] = timer_name
    meta["argv"] = vars(args)
    print(json.dumps(meta, indent=2))

    fi_ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8)
    # Ramp clocks up from idle; otherwise round 1 of the first case runs slow.
    a = torch.randn(4096, 4096, dtype=torch.bfloat16)
    t_end = time.time() + 3
    while time.time() < t_end:
        a @ a
        torch.accelerator.synchronize()
    results = []
    hdr = (
        f"{'shape':6s} {'page':>4s} {'case':8s} {'s0 us':>9s} {'s1 us':>9s} "
        f"{'s1 gain':>8s} {'FI us':>9s} {'s0 TF':>6s} {'s1 TF':>6s} "
        f"{'tile s0 -> s1 (CTA/SM by smem)':s}"
    )
    print(hdr)
    for shape_key in args.shapes.split(","):
        for page in map(int, args.pages.split(",")):
            for case in CASES:
                res = run_case(
                    shape_key,
                    page,
                    case,
                    not args.no_flashinfer,
                    fi_ws,
                    timer,
                    args.rounds,
                    meta,
                )
                results.append(res)
                us, tf, t = res["us"], res["tflops"], res["tiles"]
                gain = 100 * (us["fa2_s0"] / us["fa2_s1"] - 1)

                def tile(x):
                    if "kBlockN" not in x:
                        return "?"
                    return (
                        f"{x['kBlockM']}x{x['kBlockN']} "
                        f"{x['est_smem_bytes'] // 1024}KB "
                        f"({x['est_ctas_per_sm_by_smem']})"
                    )

                fi = f"{us['fi']:9.1f}" if "fi" in us else f"{'n/a':>9s}"
                print(
                    f"{shape_key:6s} {page:4d} {case[0]:8s} {us['fa2_s0']:9.1f} "
                    f"{us['fa2_s1']:9.1f} {gain:+7.1f}% {fi} "
                    f"{tf['fa2_s0']:6.1f} {tf['fa2_s1']:6.1f} "
                    f"{tile(t['fa2_s0'])} -> {tile(t['fa2_s1'])}",
                    flush=True,
                )
                if res["fi_error"]:
                    print(f"       FlashInfer skipped: {res['fi_error']}")

    if args.out:
        out = args.out
        with open(out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\nwrote {out}")
    print("(s1 gain = s0_time / s1_time - 1; positive means num_splits=1 is faster)")


if __name__ == "__main__":
    main()
