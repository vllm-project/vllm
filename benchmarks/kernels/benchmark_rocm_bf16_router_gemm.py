# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE router GEMM on gfx950: hipBLASLt, the low-M Triton kernel and the
split-K kernel, alone and next to a shared expert on a second stream.

GateLinear's tier 6 computes bf16 x bf16 -> fp32 router logits. At up to
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD tokens the MoE runner launches the
shared expert on an aux stream before the gate, so the two GEMMs share the
CUs. Every measurement captures many such layers in one HIP graph, each
layer with its own weights so that they stream from HBM:

* alone: the router GEMM only, over enough weight copies to overflow
  --llc-mb twice;
* contended: --layers layers, each forking the shared expert's gate_up and
  down GEMMs (MXFP8 block32 by default) onto an aux stream and joining after
  the router GEMM. "layer" is that fork-join time per layer, which only drops
  when the router GEMM was the longer branch.

Router and shared-expert kernel times come from torch.profiler over graph
replays, layer times from CUDA events (median of --rounds). The last column
says which GEMM each batch size should use, and the summary gives the range
the split-K kernel should serve.

Variants (missing ones are skipped):
* hipblaslt: torch.mm(out_dtype=fp32), what tier 6 runs above main's low-M
  limit;
* low_m: main's rocm_fp32_router_gemm kernel. Up to --low-m-max it is also
  timed past its own M * N limit (marked *), to see whether raising that
  limit would do instead;
* split_k: the split-K kernel at its tier for M;
* cand<i>: other split-K configs (--candidates), up to M 64 by default;
* dispatch: fp32_router_gemm_dispatch, i.e. this tree's whole tier-6 choice.

Run it on main to get main's numbers and on the patched tree for split-K;
run it again with a shared-expert tiling PR applied to re-check contention.

    python benchmarks/kernels/benchmark_rocm_bf16_router_gemm.py \\
        --shapes 384x5120,128x5120 --json router_gemm.json
"""

import importlib
import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from types import ModuleType

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.router import (
    rocm_fp32_router_gemm as low_m_mod,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import (
    fp32_router_gemm_dispatch_impl,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

split_k_mod: ModuleType | None
try:
    split_k_mod = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.router.rocm_bf16_router_gemm"
    )
except ImportError:
    split_k_mod = None

QUICK_M = [1, 2, 4, 6, 8, 12, 16, 18, 20, 21, 22, 24, 28, 32, 33, 36, 40, 42]
QUICK_M += [48, 56, 64, 72, 96, 120, 128, 144, 160, 192, 224, 240, 256]
QUICK_M += [257, 288, 384, 512]
ALL_M = sorted(
    set(range(1, 65))
    | set(range(66, 257, 6))
    | set(range(72, 257, 8))
    | {257, 264, 288, 320, 384, 512}
)
# (block_m, block_n, block_k, splits, num_warps, num_stages, waves_per_eu)
LOW_M_CANDIDATES = [
    (16, 16, 256, 8, 4, 3, 0),
    (32, 16, 256, 8, 4, 3, 0),
    (32, 32, 256, 8, 4, 2, 0),
    (16, 32, 128, 8, 4, 3, 0),
]

Router = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _hipblaslt(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.mm(x, w.T, out_dtype=torch.float32)


def _low_m(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Main's low-M kernel, launched without its M * N admission check."""
    M, K = x.shape
    N = w.shape[0]
    block_m, block_k, num_warps, *rest = low_m_mod._launch_config(
        N, M, w.dtype == torch.bfloat16
    )
    block_k = min(block_k, triton.next_power_of_2(K))
    out = x.new_empty((M, N), dtype=torch.float32)
    low_m_mod._rocm_fp32_router_gemm_kernel[(triton.cdiv(M, block_m) * N,)](
        x,
        w,
        out,
        M=M,
        K=K,
        N=N,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        EVEN_K=K % block_k == 0,
        num_warps=num_warps,
        num_stages=rest[0] if rest else 1,
    )
    return out


def _split_k(cfg) -> Router:
    assert split_k_mod is not None

    def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        out = x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)
        split_k_mod._launch(x, w, out, cfg)
        return out

    return run


def _dispatch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return fp32_router_gemm_dispatch_impl(x, w, False)


def _fp8(shape: tuple[int, int]) -> torch.Tensor:
    return (0.5 * torch.randn(shape, device="cuda")).to(torch.float8_e4m3fn)


def _e8m0(shape: tuple[int, int]) -> torch.Tensor:
    return torch.full(shape, 127, dtype=torch.uint8, device="cuda")


class SharedExpert:
    """The shared expert's GEMMs, one weight copy per layer."""

    def __init__(self, impl: str, shapes: list[tuple[int, int]], layers: int):
        self.impl = impl
        self.shapes = shapes
        if impl == "mxfp8":
            self.weights = [
                [
                    (_fp8((n, k)), _e8m0((triton.cdiv(n, 32), k // 32)))
                    for n, k in shapes
                ]
                for _ in range(layers)
            ]
        else:
            self.weights = [
                [
                    (torch.randn(n, k, device="cuda", dtype=torch.bfloat16), None)
                    for n, k in shapes
                ]
                for _ in range(layers)
            ]

    def bind(self, num_tokens: int) -> Callable[[int], None]:
        if self.impl == "mxfp8":
            from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (
                rocm_mxfp8_block32_gemm,
            )

            inputs = [
                (_fp8((num_tokens, k)), _e8m0((num_tokens, k // 32)))
                for _, k in self.shapes
            ]

            def run(layer: int) -> None:
                for (x, xs), (w, ws) in zip(inputs, self.weights[layer], strict=True):
                    rocm_mxfp8_block32_gemm(x, xs, w, ws, torch.bfloat16)

        else:
            acts = [
                torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16)
                for _, k in self.shapes
            ]

            def run(layer: int) -> None:
                for x, (w, _) in zip(acts, self.weights[layer], strict=True):
                    torch.mm(x, w.T)

        return run


def _kernel_class(name: str) -> str:
    if "router_gemm" in name or (name.startswith("Cijk") and "_BSS" in name):
        return "router"
    if "block32" in name or name.startswith("Cijk"):
        return "shared"
    return "other"


def measure(
    router: Router,
    x: torch.Tensor,
    weights: list[torch.Tensor],
    shared: Callable[[int], None] | None,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream],
    rounds: int,
    replays: int,
) -> dict[str, float]:
    main, aux = streams

    def step() -> None:
        for layer, w in enumerate(weights):
            if shared is not None:
                aux.wait_stream(main)
                with torch.cuda.stream(aux):
                    shared(layer)
            router(x, w)
            if shared is not None:
                main.wait_stream(aux)

    # Compiles, and creates each stream's split-K arrival counters.
    with torch.cuda.stream(main):
        step()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        step()
    graph.replay()
    torch.accelerator.synchronize()

    calls = replays * len(weights)
    layer_us = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        layer_us.append(start.elapsed_time(end) * 1e3 / calls)

    totals: dict[str, float] = defaultdict(float)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.accelerator.synchronize()
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA:
            totals[_kernel_class(ev.name)] += ev.device_time
    del graph
    return {
        "layer_us": statistics.median(layer_us),
        "router_us": totals["router"] / calls,
        "shared_us": totals["shared"] / calls,
        "other_us": totals["other"] / calls,
    }


def variants(m: int, N: int, K: int, args) -> dict[str, Router]:
    out: dict[str, Router] = {"hipblaslt": _hipblaslt}
    if main_path(m, N, K) == "low_m":
        out["low_m"] = _low_m
    elif m <= args.low_m_max:
        out["low_m*"] = _low_m
    if split_k_mod is not None and (
        split_k_mod.platform_supported() or args.ignore_arch
    ):
        cfg = split_k_mod._config(m, N, K)
        if cfg is not None:
            out["split_k"] = _split_k(cfg)
        if m <= args.candidates_max_m:
            for i, c in enumerate(args.candidates):
                out[f"cand{i}"] = _split_k(split_k_mod._Config(*c))
    if args.dispatch:
        out["dispatch"] = _dispatch
    return out


def main_path(m: int, N: int, K: int) -> str:
    """What tier 6 runs on main for this batch size."""
    x = torch.empty(m, K, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    return "low_m" if low_m_mod.can_use_rocm_fp32_router_gemm(x, w) else "hipblaslt"


def pick(m: int, row: dict, base: str, args) -> str:
    """split_k where it beats main's path: under contention by --margin while
    the shared expert overlaps the gate (and without slowing the fork-join
    layer or the alone case by more than --tolerance), alone above that."""
    if "split_k" not in row:
        return base
    sk, mp = row["split_k"], row[base]
    if m > args.stream_threshold:
        better = sk["alone"]["router_us"] < (1 - args.margin) * mp["alone"]["router_us"]
        return "split_k" if better else base
    c_sk, c_mp = sk["contended"], mp["contended"]
    better = c_sk["router_us"] < (1 - args.margin) * c_mp["router_us"]
    layer_ok = c_sk["layer_us"] <= (1 + args.tolerance) * c_mp["layer_us"]
    alone_ok = (
        sk["alone"]["router_us"] <= (1 + args.tolerance) * mp["alone"]["router_us"]
    )
    return "split_k" if better and layer_ok and alone_ok else base


def _cell(row: dict, name: str, mode: str, key: str = "router_us") -> str:
    v = row.get(name, {}).get(mode, {}).get(key)
    return f"{v:6.2f}" if v is not None else "     -"


def report(N: int, K: int, rows: dict[int, dict], args) -> list[tuple[int, int]]:
    names = ["hipblaslt", "low_m", "split_k"]
    extra = sorted({n for r in rows.values() for n in r} - set(names) - {"low_m*"})
    print(f"\n=== N={N} K={K}: router GEMM us (alone / contended), per layer ===")
    head = f"{'M':>4} {'main':>9} |"
    for n in names + ["low_m*"] + extra:
        head += f" {n:>15} |"
    head += f" {'layer us main':>13} {'split_k':>7} | pick"
    print(head)
    picks = {}
    for m, row in sorted(rows.items()):
        base = main_path(m, N, K)
        line = f"{m:4d} {base:>9} |"
        for n in names + ["low_m*"] + extra:
            line += f" {_cell(row, n, 'alone')} {_cell(row, n, 'contended')}  |"
        line += f" {_cell(row, base, 'contended', 'layer_us'):>13}"
        line += f" {_cell(row, 'split_k', 'contended', 'layer_us'):>7} |"
        picks[m] = pick(m, row, base, args)
        print(line + f" {picks[m]}")
    runs: list[tuple[int, int]] = []
    for m in sorted(picks):
        if picks[m] != "split_k":
            continue
        if runs and runs[-1][1] == max(p for p in picks if p < m):
            runs[-1] = (runs[-1][0], m)
        else:
            runs.append((m, m))
    print(
        f"split-K should serve N={N} K={K} at M in "
        + (", ".join(f"[{a}, {b}]" for a, b in runs) or "no measured batch size")
        + f" (margin {args.margin:.0%}, tolerance {args.tolerance:.0%})"
    )
    return runs


def main(args) -> None:
    if args.m == "quick":
        ms = QUICK_M
    elif args.m == "all":
        ms = ALL_M
    else:
        ms = sorted({int(s) for s in args.m.split(",")})
    shared_shapes = [
        tuple(map(int, s.split("x"))) for s in args.shared_shapes.split(",")
    ]
    props = torch.cuda.get_device_properties(0)
    meta = {
        "device": props.name,
        "arch": getattr(props, "gcnArchName", "?"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "split_k_module": split_k_mod is not None,
        "layers": args.layers,
        "shared": f"{args.shared} {args.shared_shapes}",
        "stream_threshold": args.stream_threshold,
    }
    print(json.dumps(meta))
    shared = SharedExpert(args.shared, shared_shapes, args.layers)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    timing = (streams, args.rounds, args.replays)
    results: dict = {"meta": meta, "shapes": {}}
    for shape in args.shapes.split(","):
        N, K = map(int, shape.split("x"))
        copies = max(args.layers, -(-2 * args.llc_mb * 2**20 // (2 * N * K)))
        weights = [
            (0.02 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
            for _ in range(copies)
        ]
        rows: dict[int, dict] = {}
        for m in ms:
            x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
            bound = shared.bind(m)
            rows[m] = {}
            for name, fn in variants(m, N, K, args).items():
                rows[m][name] = {
                    "alone": measure(fn, x, weights, None, *timing),
                    "contended": measure(fn, x, weights[: args.layers], bound, *timing),
                }
        runs = report(N, K, rows, args)
        results["shapes"][shape] = {"rows": rows, "split_k_ranges": runs}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="384x5120,128x5120", help="NxK,...")
    parser.add_argument("--m", default="quick", help="quick, all, or M,M,...")
    parser.add_argument("--layers", type=int, default=72)
    parser.add_argument("--llc-mb", type=int, default=256, help="MI3xx MALL size")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--shared", choices=["mxfp8", "bf16"], default="mxfp8")
    parser.add_argument(
        "--shared-shapes",
        default="4608x5120,5120x2304",
        help="NxK of the shared expert's gate_up and down GEMMs (TP1 DSv4.1)",
    )
    parser.add_argument(
        "--stream-threshold",
        type=int,
        default=envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
        help="largest M whose shared expert overlaps the gate in serving",
    )
    parser.add_argument("--low-m-max", type=int, default=64)
    parser.add_argument(
        "--candidates",
        type=json.loads,
        default=LOW_M_CANDIDATES,
        help="JSON list of split-K configs to time; [] for none",
    )
    parser.add_argument("--candidates-max-m", type=int, default=64)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--ignore-arch", action="store_true", help="time split-K off gfx950 too"
    )
    parser.add_argument("--json", default=None)
    main(parser.parse_args())
