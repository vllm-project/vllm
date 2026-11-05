# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-seam latency of DeepSeek-V4.1's fused delayed mHC seam on ROCm (gfx950).

One seam (hc_mult 4, 20 Sinkhorn iterations) on each backend:

- ``triton``: AITER's two-launch Triton seam, what vLLM main runs
  (``rocm_aiter_ops.mhc_fused_post_pre_delayed_rms_norm``).
- ``hip``: AITER's single-launch HIP Mega-MHC (``aiter.ops.mega_mhc``), reading
  ``fn`` packed once by ``mega_mhc_shuffle_fn``.
- ``flydsl``: AITER's single-launch FlyDSL Mega-mHC
  (``aiter.ops.flydsl.mega_mhc_kernels``), which packs ``fn`` on first use.
- ``overlap_main`` (``post`` mode only): vLLM's post-collapse kernel, all that
  stays on the critical path when the gates run on the side stream (FULL
  graphs, at most 128 tokens). The side stream then runs a ``no_post`` seam.

Every backend is checked against a float64 reference before it is timed.
Timing is HIP-graph replay over rotating inputs and weights larger than the
256 MB MALL, warmed up and captured on one stream (both Mega-MHC kernels keep
per-stream scratch), reported as the median over rounds.

Example:
    python benchmarks/kernels/benchmark_mhc_seam_rocm.py --json mhc_seam.json
    python benchmarks/kernels/benchmark_mhc_seam_rocm.py \
        --checkpoint /path/to/DeepSeek-V4.1-Flash --layer 30 --sublayer ffn

"""

import argparse
import json
import math
import os
import statistics
from importlib import metadata

import torch
import torch.nn.functional as F

HC = 4
RMS_EPS = HC_EPS = NORM_EPS = 1e-6
POST_MULT = 2.0
SINKHORN = 20
DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 129, 160, 192, 224, 256)
DEFAULT_TOKENS += (384, 512, 1024, 2048, 4096)
ROTATE_BYTES = 512 * 1024**2
MAX_SLOTS = 160
INNER, REPLAYS, ROUNDS, WARMUP = 2, 10, 5, 5


def load_weights(args, hidden: int) -> dict[str, torch.Tensor]:
    if args.checkpoint is None:
        g = torch.Generator("cuda").manual_seed(0)
        return dict(
            fn=torch.randn(24, HC * hidden, device="cuda", generator=g) * 0.01,
            scale=torch.tensor([0.7, 1.3, 0.9], device="cuda"),
            base=torch.randn(24, device="cuda", generator=g) * 0.5,
            norm=(1 + 0.2 * torch.randn(hidden, device="cuda", generator=g)).to(
                torch.bfloat16
            ),
        )
    from safetensors import safe_open

    with open(os.path.join(args.checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix, sub = f"layers.{args.layer}.", args.sublayer
    names = {
        "fn": f"hc_{sub}_fn",
        "scale": f"hc_{sub}_scale",
        "base": f"hc_{sub}_base",
        "norm": f"{sub}_norm.weight",
    }
    weights = {}
    for key, name in names.items():
        path = os.path.join(args.checkpoint, weight_map[prefix + name])
        with safe_open(path, "pt", device="cuda") as f:
            weights[key] = f.get_tensor(prefix + name).float().contiguous()
    weights["norm"] = weights["norm"].to(torch.bfloat16)
    return weights


def make_inputs(tokens: int, hidden: int, mode: str, seed: int):
    g = torch.Generator("cuda").manual_seed(seed)
    residual = torch.randn(tokens, HC, hidden, device="cuda", generator=g) * 0.6
    inputs = dict(
        residual=residual.to(torch.bfloat16),
        pre_mix=torch.rand(tokens, HC, device="cuda", generator=g) + 0.1,
    )
    if mode == "post":
        sublayer = torch.randn(tokens, hidden, device="cuda", generator=g) * 0.3
        comb = torch.randn(tokens, HC, HC, device="cuda", generator=g)
        inputs.update(
            sublayer_out=sublayer.to(torch.bfloat16),
            post_layer_mix=torch.rand(tokens, HC, 1, device="cuda", generator=g) * 2,
            comb_res_mix=torch.softmax(comb, dim=-1),
            residual_out=torch.empty_like(inputs["residual"]),
        )
    return inputs


def _post_kwargs(x):
    keys = ("sublayer_out", "post_layer_mix", "comb_res_mix", "residual_out")
    return {k: x[k] for k in keys if k in x}


def run_triton(w, x):
    from vllm._aiter_ops import rocm_aiter_ops

    post = _post_kwargs(x)
    gates = rocm_aiter_ops.mhc_fused_post_pre_delayed_rms_norm(
        x["residual"],
        w["fn"],
        w["scale"],
        w["base"],
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN,
        x["pre_mix"],
        post.get("sublayer_out"),
        post.get("post_layer_mix"),
        post.get("comb_res_mix"),
        w["norm"],
        NORM_EPS,
        post.get("residual_out"),
    )
    return (post.get("residual_out", x["residual"]), *gates)


def run_hip(w, x):
    from aiter.ops.mega_mhc import mega_mhc

    return mega_mhc(
        x["residual"],
        w["fn_packed"],
        w["scale"],
        w["base"],
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN,
        x["pre_mix"],
        norm_weight=w["norm"],
        norm_eps=NORM_EPS,
        w_preshuffle_bf16=True,
        **_post_kwargs(x),
    )


def run_flydsl(w, x):
    from aiter.ops.flydsl.mega_mhc_kernels import flydsl_mega_mhc

    return flydsl_mega_mhc(
        x["residual"],
        w["fn"],
        w["scale"],
        w["base"],
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN,
        x["pre_mix"],
        norm_weight=w["norm"],
        norm_eps=NORM_EPS,
        **_post_kwargs(x),
    )


def run_overlap_main(w, x):
    from vllm.model_executor.kernels.mhc.triton import (
        mhc_post_collapse_rms_norm_triton,
    )

    residual, layer_input = mhc_post_collapse_rms_norm_triton(
        x["residual"],
        x["sublayer_out"],
        x["post_layer_mix"],
        x["comb_res_mix"],
        x["pre_mix"],
        w["norm"],
        NORM_EPS,
    )
    return residual, None, None, layer_input, None


BACKENDS = {
    "triton": run_triton,
    "hip": run_hip,
    "flydsl": run_flydsl,
    "overlap_main": run_overlap_main,
}


def reference_fp64(w, x):
    """The seam in float64, rounded to bf16 where the kernels round."""
    f64, n = torch.float64, HC
    residual = x["residual"]
    if "sublayer_out" in x:
        residual = (
            torch.einsum("tij,tih->tjh", x["comb_res_mix"].to(f64), residual.to(f64))
            + x["post_layer_mix"].to(f64) * x["sublayer_out"].to(f64).unsqueeze(1)
        ).to(torch.bfloat16)
    r = residual.flatten(1).to(f64)
    mixes = r @ w["fn"].to(f64).t() * torch.rsqrt(r.square().mean(-1, True) + RMS_EPS)
    scale, base = w["scale"].to(f64), w["base"].to(f64)
    pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + HC_EPS
    post = torch.sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n]) * POST_MULT
    comb = mixes[:, 2 * n :] * scale[2] + base[2 * n :]
    comb = torch.softmax(comb.view(-1, n, n), -1) + HC_EPS
    comb = comb / (comb.sum(-2, True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(-1, True) + HC_EPS)
        comb = comb / (comb.sum(-2, True) + HC_EPS)
    collapse = (x["pre_mix"].to(f64).unsqueeze(-1) * residual.to(f64)).sum(1)
    collapse = collapse.to(torch.bfloat16).to(f64)
    layer_input = F.rms_norm(collapse, collapse.shape[-1:], w["norm"].to(f64), NORM_EPS)
    return residual, post.unsqueeze(-1), comb, layer_input.to(torch.bfloat16), pre


def bf16_ulps(actual, expected) -> tuple[float, float]:
    """Largest difference in bf16 ulps, and the fraction of equal elements."""
    a, b = actual.float(), expected.float()
    magnitude = torch.maximum(a.abs(), b.abs())
    ulp = torch.ldexp(torch.ones_like(magnitude), torch.frexp(magnitude)[1] - 8)
    return float(((a - b).abs() / ulp).max()), float((a == b).float().mean())


def rel_err(actual, expected) -> float:
    a, b = actual.double(), expected.double()
    return float((a - b).norm() / b.norm())


def check(name, out, ref) -> dict:
    """Numerics against the float64 reference; raises if clearly wrong."""
    report = {}
    for i, label in ((0, "residual"), (3, "layer_input")):
        ulps, equal = bf16_ulps(out[i], ref[i])
        report[label] = dict(
            max_ulps=ulps, frac_equal=equal, rel_err=rel_err(out[i], ref[i])
        )
    for i, label in ((1, "post_mix"), (2, "comb_mix"), (4, "next_pre_mix")):
        if out[i] is not None:
            report[label] = dict(rel_err=rel_err(out[i].view_as(ref[i]), ref[i]))
    bad = [
        label
        for label, r in report.items()
        if r.get("max_ulps", 0) > 4 or r["rel_err"] > 1e-3
    ]
    if bad:
        raise AssertionError(f"{name}: {bad} disagree with the reference: {report}")
    return report


def time_graph(call, slots, stream) -> float:
    """Median over rounds of µs per call, replaying one graph over all slots."""
    with torch.cuda.stream(stream):
        for w, x in slots:
            call(w, x)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(INNER):
            for w, x in slots:
                call(w, x)
    torch.accelerator.synchronize()
    for _ in range(WARMUP):
        graph.replay()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(ROUNDS)]
    for start, end in events:
        start.record()
        for _ in range(REPLAYS):
            graph.replay()
        end.record()
    torch.accelerator.synchronize()
    per_call = REPLAYS * INNER * len(slots)
    return statistics.median(s.elapsed_time(e) * 1e3 / per_call for s, e in events)


def slot_bytes(tokens: int, hidden: int, mode: str) -> int:
    # residual + layer input (bf16), sublayer_out + residual_out with the post,
    # plus the fp32 fn (or its bf16 hi/lo pack, the same size).
    per_token = hidden * (20 if mode == "post" else 10)
    return tokens * per_token + 24 * HC * hidden * 4


def versions() -> dict:
    found = {}
    for dist in ("vllm", "amd-aiter", "flydsl", "torch", "triton"):
        try:
            found[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            found[dist] = None
    props = torch.cuda.get_device_properties(0)
    found.update(
        device=props.name,
        arch=props.gcnArchName,
        cus=props.multi_processor_count,
        hip=torch.version.hip,
    )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument(
        "--modes", nargs="+", default=["post", "no_post"], choices=["post", "no_post"]
    )
    parser.add_argument(
        "--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS)
    )
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--checkpoint", help="DeepSeek-V4.1 checkpoint directory")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--sublayer", choices=["attn", "ffn"], default="ffn")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()

    meta = versions()
    if not meta["arch"].startswith("gfx950"):
        raise SystemExit(f"gfx950 required, found {meta['arch']}")
    print(json.dumps(meta))
    from aiter.ops.mega_mhc import mega_mhc_shuffle_fn

    base_weights = load_weights(args, args.hidden)
    stream = torch.cuda.Stream()
    rows = []
    for t_index, tokens in enumerate(args.tokens):
        for mode in args.modes:
            backends = [
                b for b in args.backends if b != "overlap_main" or mode == "post"
            ]
            if t_index % 2:
                backends.reverse()
            count = math.ceil(ROTATE_BYTES / slot_bytes(tokens, args.hidden, mode))
            count = 1 if args.check_only else min(max(count, 2), MAX_SLOTS)
            slots = []
            for i in range(count):
                fn = base_weights["fn"].clone()
                w = dict(base_weights, fn=fn, fn_packed=mega_mhc_shuffle_fn(fn))
                slots.append((w, make_inputs(tokens, args.hidden, mode, seed=i)))
            ref = reference_fp64(*slots[0])
            row = dict(tokens=tokens, mode=mode, slots=count)
            for name in backends:
                call = BACKENDS[name]
                with torch.cuda.stream(stream):
                    first = call(*slots[0])
                    again = call(*slots[0])
                torch.accelerator.synchronize()
                row[f"{name}_numerics"] = check(name, first, ref)
                # The post modes share residual_out, so only the other outputs
                # are compared across the two calls.
                row[f"{name}_repeat_bitwise"] = all(
                    a is None or torch.equal(a, b) for a, b in zip(first[1:], again[1:])
                )
                if not args.check_only:
                    row[f"{name}_us"] = time_graph(call, slots, stream)
            rows.append(row)
            print(
                f"T={tokens:5d} {mode:8s} "
                + " ".join(
                    f"{name}={row.get(f'{name}_us', float('nan')):8.2f}us"
                    for name in args.backends
                    if f"{name}_numerics" in row
                ),
                flush=True,
            )
            del slots
            torch.accelerator.empty_cache()

    if not args.check_only:
        print("\nus per seam (median of rounds); ratios < 1 favour the numerator")
        header = "| T | mode | " + " | ".join(args.backends)
        print(header + " | hip/triton | flydsl/triton | hip/flydsl |")
        print("|" + " ---: |" * (len(args.backends) + 5))
        for row in rows:
            us = {b: row.get(f"{b}_us") for b in args.backends}
            cells = [f"{v:.2f}" if v else "-" for v in us.values()]

            def ratio(a, b, us=us):
                return f"{us[a] / us[b]:.3f}" if us.get(a) and us.get(b) else "-"

            print(
                f"| {row['tokens']} | {row['mode']} | "
                + " | ".join(cells)
                + f" | {ratio('hip', 'triton')} | {ratio('flydsl', 'triton')}"
                + f" | {ratio('hip', 'flydsl')} |"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(meta=meta, args=vars(args), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
