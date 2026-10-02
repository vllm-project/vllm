# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune the CuTe skinny GEMM for Qwen3.8-Flash-Next on sm_121 (GB10).

Adapted from ``benchmark_k3_cutedsl_residual.py``: same sweep grid, same
rotating-working-set and CUDA-graph methodology, but parameterised over
``(N, K)`` instead of one hardcoded Kimi-K3 shape, driving the installed
``shape_dynamic_skinny_gemm`` (no ``--kernel`` path), and scoring every
candidate against ``F.linear`` -- the retention criterion upstream applies by
hand. A config is kept only if it beats ``F.linear`` both hot and cold.

Emits JSON lines plus a paste-ready plan dict for ``low_latency_gemm.py``.

Examples::

    python benchmark_skinny_gemm.py --mode sweep --out sweep.jsonl
    python benchmark_skinny_gemm.py --mode sweep --shape 248320,2560 --m 1 4
    python benchmark_skinny_gemm.py --mode sweep --num-config-shards 4 \
        --config-shard 0
    python benchmark_skinny_gemm.py --merge 'skinny_sweep.shard*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import statistics
import sys
from dataclasses import replace

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# Qwen3.8-Flash-Next (N, K) shapes, given as the unsharded TP=1 value plus the
# axis tensor parallelism splits. Derived from the TP=4 plan table in
# qwen4_exp/nvidia/low_latency_gemm.py; cross-checked against the model config:
# 62080*4 = vocab 248320, 320*4 = 2*moe_intermediate 1280, 1536*4 = 24 heads *
# head_dim 256. Confirm against a real run by logging (N, K, M) from the
# F.linear fallback in low_latency_gemm.py before trusting the derivation.
QWEN38_FN_SHAPES: tuple[tuple[int, int, str, str], ...] = (
    (16384, 2560, "N", "GDN fused QKVZ projection"),
    (2560, 6144, "K", "GDN and QSA output projections"),
    (96, 2560, "N", "GDN fused B/A projection"),
    (14336, 2560, "N", "QSA fused QKV/gate projection"),
    (640, 2560, "", "QSA indexer Q/K projection (replicated)"),
    (1280, 2560, "N", "Shared-expert fused gate/up projection"),
    (248320, 2560, "N", "LM head"),
    (336, 10240, "", "HC merged down/injection projection (replicated)"),
)


def shapes_for_tp(tp_sizes) -> dict[tuple[int, int], str]:
    """Local shapes seen by the kernel at each TP size. Shapes that collide
    across TP sizes (e.g. shared-expert at TP=2 equals the replicated indexer)
    fold into one entry, since the plan table is keyed by shape alone."""
    out: dict[tuple[int, int], str] = {}
    for tp in tp_sizes:
        for n, k, axis, label in QWEN38_FN_SHAPES:
            ln, lk = (
                (n // tp, k)
                if axis == "N"
                else ((n, k // tp) if axis == "K" else (n, k))
            )
            if axis and (n if axis == "N" else k) % tp:
                continue
            tag = f"{label}, TP={tp}" if axis else f"{label}, any TP"
            out[(ln, lk)] = out.get((ln, lk), tag)
    return out


BLOCK_SIZES = (32, 64, 128, 224, 448)
OUTPUTS_PER_BLOCK = (1, 2, 3, 4, 7, 8)
K_UNROLLS = (1, 2, 4, 5, 6)
VECTOR_WIDTHS = (2, 4, 8)


def candidate_configs(n: int, k: int, m: int) -> list[SkinnyGemmConfig]:
    """The sweep grid, pre-filtered by the two divisibility constraints that
    ``ShapeDynamicSkinnyGemm._compile`` enforces (N % opb, K % (bs * vw))."""
    out = []
    for bs, vw, opb, ku in itertools.product(
        BLOCK_SIZES, VECTOR_WIDTHS, OUTPUTS_PER_BLOCK, K_UNROLLS
    ):
        if n % opb or k % (bs * vw):
            continue
        base = SkinnyGemmConfig(m, bs, opb, k_unroll=ku, vector_width=vw)
        out.append(base)
        # static_k trades shape-genericity for a compile-time constant K.
        out.append(replace(base, static_k=k))
    return out


def l2_bytes() -> int:
    props = torch.accelerator.current_device_properties()
    for attr in ("L2_cache_size", "l2_cache_size", "l2CacheSize"):
        value = getattr(props, attr, None)
        if value:
            return int(value)
    return 32 << 20  # conservative fallback


def buffer_count(
    weight_bytes: int, target_bytes: int, max_buffers: int, max_working_set: int
) -> int:
    """How many distinct weight buffers to rotate through so the working set
    exceeds L2, bounded by --max-buffers and a total memory budget."""
    need = max(1, math.ceil(target_bytes / weight_bytes))
    by_memory = max(1, max_working_set // weight_bytes)
    return max(1, min(need, max_buffers, by_memory))


def time_graph(callables, replays: int, repeats: int) -> float:
    """Median per-call time in ms. One graph holds every rotation step, so a
    replay touches the whole working set; the total is divided by len()."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for fn in callables:
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in callables:
            fn()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        torch.accelerator.synchronize()
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.accelerator.synchronize()
        samples.append(start.elapsed_time(end) / (replays * len(callables)))
    return statistics.median(samples)


def sized_replays(callables, target_ms: float, cap: int) -> int:
    """Replay count that makes one repeat cost about target_ms. Without this a
    fixed count is either noise on a 17 us shape or minutes on the 7.5 ms LM
    head; the same count is then reused for baseline and candidates."""
    per_call = time_graph(callables, replays=3, repeats=1)
    one_replay = max(per_call * len(callables), 1e-4)
    return max(5, min(cap, int(target_ms / one_replay)))


def make_weights(n: int, k: int, count: int) -> list[torch.Tensor]:
    return [
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16) for _ in range(count)
    ]


def reference(a, weight):
    """FP32 truth plus the error F.linear itself makes against it."""
    ref = torch.nn.functional.linear(a.float(), weight.float())
    err = (torch.nn.functional.linear(a, weight).float() - ref).abs().max().item()
    return ref, max(err, ref.abs().max().item() * 1e-3)


def correct(a, weight, config, ref, budget: float) -> bool:
    """Score against the FP32 reference, not against cuBLAS BF16. The two BF16
    paths differ only in summation order, so an elementwise rtol explodes on the
    near-zero outputs that cancellation produces. The bar is instead "no worse
    than the kernel we are replacing"."""
    got = shape_dynamic_skinny_gemm(a, weight, config).float()
    return (got - ref).abs().max().item() <= 2.0 * budget


def plan_lines(winners, labels) -> list[str]:
    """Il dict dei piani, pronto da incollare in low_latency_gemm.py."""
    out = []
    for nk in sorted(winners, key=lambda t: -t[0]):
        out.append(f"    # {labels.get(nk, '')}.")
        out.append(f"    {nk}: {{")
        for m, cfg in sorted(winners[nk].items()):
            extra = "".join(
                f", {f}={getattr(cfg, f)}"
                for f in ("k_unroll", "vector_width", "static_k")
                if getattr(cfg, f) != SkinnyGemmConfig.__dataclass_fields__[f].default
            )
            out.append(
                f"        {m}: SkinnyGemmConfig({m}, {cfg.block_size}, "
                f"{cfg.outputs_per_block}{extra}),"
            )
        out.append("    },")
    return out


def merge_shards(patterns):
    """Vincitori globali su piu' file JSONL, uno per shard."""
    best: dict[tuple[int, int, int], dict] = {}
    labels: dict[tuple[int, int], str] = {}
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not files:
        raise SystemExit(f"no files matched {patterns}")
    rows = 0
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if not d.get("wins"):
                    continue
                rows += 1
                key = (d["n"], d["k"], d["m"])
                if key not in best or d["cold_ms"] < best[key]["cold_ms"]:
                    best[key] = d
                labels[(d["n"], d["k"])] = d.get("label", "")
    winners: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {}
    for (n, k, m), r in best.items():
        winners.setdefault((n, k), {})[m] = SkinnyGemmConfig(
            m,
            r["block_size"],
            r["outputs_per_block"],
            k_unroll=r["k_unroll"],
            vector_width=r["vector_width"],
            static_k=r["static_k"],
        )
    return winners, labels, files, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sweep", "heuristic"),
        default="sweep",
        help="heuristic: measure only what _config() would pick",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="N,K -- repeatable; overrides the Qwen3.8-FN set",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="+",
        default=(1, 2),
        help="TP sizes whose local shapes to tune",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16),
        help="token counts; the kernel requires 1 <= M <= 16",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="defaults to a per-shard name, since shards run concurrently and "
        "each truncates its own file",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="GLOB",
        help="merge JSONL files from a finished sweep and print the global plan, "
        "then exit; a shard only ever sees its own slice of the search space",
    )
    parser.add_argument("--config-shard", type=int, default=0)
    parser.add_argument("--num-config-shards", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--replays", type=int, default=200, help="cap on replays")
    parser.add_argument(
        "--target-ms",
        type=float,
        default=80.0,
        help="wall time each repeat should take; sizes --replays",
    )
    parser.add_argument("--cache-multiplier", type=float, default=3.0)
    parser.add_argument("--max-buffers", type=int, default=64)
    parser.add_argument("--max-working-set-gb", type=float, default=8.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="smoke test: cap candidates"
    )
    args = parser.parse_args()

    if args.merge:
        winners, labels, files, rows = merge_shards(args.merge)
        print(f"# merged {rows} winning points from {len(files)} files")
        print("QWEN4_EXP_SM121_GEMM_PLANS = {")
        print("\n".join(plan_lines(winners, labels)))
        print("}")
        return

    if args.out is None:
        if args.num_config_shards == 1:
            args.out = "skinny_sweep.jsonl"
        else:
            shard = f"{args.config_shard}-of-{args.num_config_shards}"
            args.out = f"skinny_sweep.shard{shard}.jsonl"

    if any(not 1 <= m <= 16 for m in args.m):
        raise SystemExit("the skinny GEMM requires 1 <= M <= 16")
    if not 0 <= args.config_shard < args.num_config_shards:
        raise SystemExit("config shard must be in [0, num_config_shards)")
    if not shape_dynamic_skinny_gemm.is_available():
        raise SystemExit("cuteDSL unavailable -- run inside the vLLM container")

    torch.accelerator.set_device_index(0)
    capability = torch.cuda.get_device_capability()  # noqa: TID251
    # Deliberately not gated: the upstream K3 benchmark hard-fails off SM103,
    # which is exactly what kept this kernel unmeasured on GB10.
    props = torch.accelerator.current_device_properties()
    meta = {
        "device": props.name,
        "capability": list(capability),
        "l2_bytes": l2_bytes(),
        "torch": torch.__version__,
    }
    print(json.dumps({"metadata": meta}), file=sys.stderr)

    if args.shape:
        shapes = {tuple(int(v) for v in s.split(",")): "cli" for s in args.shape}
    else:
        shapes = shapes_for_tp(args.tp)

    target = int(args.cache_multiplier * l2_bytes())
    max_ws = int(args.max_working_set_gb * (1 << 30))
    winners: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {}

    with open(args.out, "w", encoding="utf-8") as sink:
        sink.write(json.dumps({"metadata": meta}) + "\n")
        for (n, k), label in shapes.items():
            weight_bytes = n * k * 2
            count = buffer_count(weight_bytes, target, args.max_buffers, max_ws)
            cold_weights = make_weights(n, k, count)
            working_set = weight_bytes * count
            # Reported so a run where the cap kept the working set inside L2 is
            # visible rather than silently mislabelled "cold".
            truly_cold = working_set > l2_bytes()

            for m in args.m:
                torch.manual_seed(20260930 + m)
                a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)

                cold_lin = [
                    lambda w=w, a=a: torch.nn.functional.linear(a, w)
                    for w in cold_weights
                ]
                hot_lin = [
                    lambda a=a, w=cold_weights[0]: torch.nn.functional.linear(a, w)
                ]
                ref32, budget = reference(a, cold_weights[0])
                rc = sized_replays(cold_lin, args.target_ms, args.replays)
                rh = sized_replays(hot_lin, args.target_ms, args.replays)
                base_cold = time_graph(cold_lin, rc, args.repeats)
                base_hot = time_graph(hot_lin, rh, args.repeats)

                if args.mode == "heuristic":
                    cands = [
                        replace(shape_dynamic_skinny_gemm._config(m, n, k), num_rows=m)
                    ]
                else:
                    cands = candidate_configs(n, k, m)
                cands = cands[args.config_shard :: args.num_config_shards]
                if args.limit:
                    cands = cands[: args.limit]

                best = None
                for cfg in cands:

                    def cold_call(w, c=cfg, a=a):
                        return lambda: shape_dynamic_skinny_gemm(a, w, c)

                    try:
                        if not correct(a, cold_weights[0], cfg, ref32, budget):
                            continue
                        cold = time_graph(
                            [cold_call(w) for w in cold_weights], rc, args.repeats
                        )
                        hot = time_graph([cold_call(cold_weights[0])], rh, args.repeats)
                    except Exception as error:  # compile or launch rejection
                        sink.write(
                            json.dumps(
                                {
                                    "n": n,
                                    "k": k,
                                    "m": m,
                                    "config": str(cfg),
                                    "error": type(error).__name__,
                                }
                            )
                            + "\n"
                        )
                        continue

                    wins = cold < base_cold and hot < base_hot
                    row = {
                        "n": n,
                        "k": k,
                        "m": m,
                        "label": label,
                        "block_size": cfg.block_size,
                        "outputs_per_block": cfg.outputs_per_block,
                        "k_unroll": cfg.k_unroll,
                        "vector_width": cfg.vector_width,
                        "static_k": cfg.static_k,
                        "cold_ms": cold,
                        "hot_ms": hot,
                        "linear_cold_ms": base_cold,
                        "linear_hot_ms": base_hot,
                        "speedup_cold": base_cold / cold,
                        "working_set_bytes": working_set,
                        "replays_cold": rc,
                        "replays_hot": rh,
                        "truly_cold": truly_cold,
                        "wins": wins,
                    }
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    if wins and (best is None or cold < best[0]):
                        best = (cold, cfg)

                status = "-" if best is None else f"{base_cold / best[0]:.3f}x"
                print(
                    f"N={n:>7} K={k:>5} M={m:>2}  "
                    f"linear {base_cold:7.3f} ms  best {status}  "
                    f"({len(cands)} cand, cold={'y' if truly_cold else 'N'})",
                    file=sys.stderr,
                )
                if best is not None:
                    winners.setdefault((n, k), {})[m] = best[1]
                del ref32
            del cold_weights
            torch.accelerator.empty_cache()

    scope = (
        "best over this shard only -- rerun with --merge to combine shards"
        if args.num_config_shards > 1
        else "paste into QWEN4_EXP_GEMM_PLANS"
    )
    print(f"\n# {scope}", file=sys.stderr)
    for line in plan_lines(winners, shapes):
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
