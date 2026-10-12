# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-gated, hot-cache CUDA-graph benchmark of complete BF16 chains."""

import argparse
import json
import random
import statistics
from functools import partial


def measure(functions, args, torch):
    graphs = {}
    outputs = {}
    for name, fn in functions.items():
        for _ in range(10):
            fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.inner):
                outputs[name] = fn()
        graphs[name] = graph

    trials = {name: [] for name in graphs}
    rng = random.Random(0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.trials):
        names = list(graphs)
        rng.shuffle(names)
        for name in names:
            graph = graphs[name]
            for _ in range(3):
                graph.replay()
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / (args.replays * args.inner)
            trials[name].append(us)
    return {
        name: {"median_us": statistics.median(times), "trials_us": times}
        for name, times in trials.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", default="1,4")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--cudnn", action="store_true")
    parser.add_argument("--apply-router-weight-on-input", action="store_true")
    parser.add_argument("--inner", type=int, default=32)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--json", type=str)
    args = parser.parse_args()
    ms = [int(m) for m in args.m_values.split(",")]
    if (
        min(ms + [args.inner, args.replays, args.trials, args.width, args.topk]) < 1
        or args.width > 8192
        or args.topk > 32
    ):
        parser.error("Require positive sizes, width <= 8192, and topk <= 32")

    import torch
    import torch.nn.functional as F

    from vllm.model_executor.layers.fused_moe.bf16_decode_fusions import (
        bf16_moe_weighted_sum,
        bf16_shared_expert_gate,
    )

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    from vllm.utils.torch_utils import set_random_seed

    set_random_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compute_capability": torch.cuda.get_device_capability(),
        "cache_mode": "hot",
        "args": vars(args),
        "cases": [],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print("CUDA-graph batch averages; speedups compare complete operation chains.")
    failed = False
    with torch.inference_mode():
        for m in ms:
            x = torch.randn(m, args.width, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn_like(x[:1]) / args.width**0.5
            shared = torch.randn_like(x)
            experts = torch.randn(
                m, args.topk, args.width, device="cuda", dtype=torch.bfloat16
            )
            weights = torch.randn(m, args.topk, device="cuda").softmax(-1)
            output = torch.empty_like(x)

            def shared_reference(x=x, weight=weight, shared=shared):
                return F.linear(x, weight).sigmoid() * shared

            def reduce_reference(experts=experts, weights=weights, output=output):
                values = experts
                if not args.apply_router_weight_on_input:
                    values = values * weights.bfloat16()[..., None]
                output.copy_(values.sum(dim=1), non_blocking=True)
                return output

            def reduce_fused(block_h, experts=experts, weights=weights, output=output):
                bf16_moe_weighted_sum(
                    experts,
                    weights,
                    output,
                    args.apply_router_weight_on_input,
                    block_h,
                )
                return output

            shared_functions = {"torch_chain": shared_reference}
            if args.cudnn:
                import flashinfer

                def cudnn_reference(x=x, weight=weight, shared=shared):
                    gate = flashinfer.mm_bf16(
                        x,
                        weight.t(),
                        bias=None,
                        pdl=False,
                        out_dtype=torch.bfloat16,
                        backend="cudnn",
                    )
                    return gate.sigmoid() * shared

                with flashinfer.autotune(True):
                    cudnn_reference()
                shared_functions["cudnn_chain"] = cudnn_reference
                report["flashinfer"] = flashinfer.__version__
            shared_functions.update(
                {
                    f"fused_h{block}": partial(
                        bf16_shared_expert_gate, x, weight, shared, block
                    )
                    for block in (256, 512, 1024, 4096)
                }
            )
            reduce_functions = {"torch_chain": reduce_reference}
            reduce_functions.update(
                {
                    f"fused_h{block}": partial(reduce_fused, block)
                    for block in (128, 256, 512)
                }
            )
            for operation, functions in (
                ("shared_gate", shared_functions),
                ("moe_reduce", reduce_functions),
            ):
                reference = functions["torch_chain"]().clone().float()
                checks = {}
                valid = {}
                for name, fn in functions.items():
                    actual = fn().float()
                    error = actual - reference
                    relative_l2 = (
                        error.norm() / reference.norm().clamp_min(1e-12)
                    ).item()
                    rtol, atol = (
                        (0.016, 0.001) if operation == "shared_gate" else (0.008, 1e-5)
                    )
                    passed = (
                        bool(torch.isfinite(actual).all())
                        and bool((error.abs() <= atol + rtol * reference.abs()).all())
                        and relative_l2 < 0.003
                    )
                    checks[name] = {
                        "passed": passed,
                        "max_abs_error": error.abs().max().item(),
                        "relative_l2_error": relative_l2,
                    }
                    if passed:
                        valid[name] = fn
                    else:
                        failed = True
                        print(f"FAIL {operation} M={m} {name}: {checks[name]}")
                results = measure(valid, args, torch)
                baseline = (
                    "cudnn_chain"
                    if operation == "shared_gate" and args.cudnn
                    else "torch_chain"
                )
                baseline_time = results.get(baseline, {}).get("median_us")
                for name, result in results.items():
                    if baseline_time is not None:
                        result["speedup"] = baseline_time / result["median_us"]
                    speedup = result.get("speedup", float("nan"))
                    print(
                        f"{operation:12s} M={m:2d} {name:14s} "
                        f"{result['median_us']:8.3f} us {speedup:6.3f}x"
                    )
                report["cases"].append(
                    {
                        "operation": operation,
                        "m": m,
                        "baseline": baseline,
                        "checks": checks,
                        "timings": results,
                    }
                )
    if args.json:
        from pathlib import Path

        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit("One or more candidates failed correctness; see checks above.")


if __name__ == "__main__":
    main()
