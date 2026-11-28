# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import importlib
import itertools
import json
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser

num_tokens_range = [2**i for i in range(0, 8, 2)]
num_experts_range = [16, 32, 64, 128, 256, 512]
topk_range = [3, 4, 8, 10]
configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


def torch_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "softmax",
):
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.float(), dim=-1)
    else:
        scores = torch.sigmoid(gating_output.float())
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


def get_benchmark(scoring_func, dtype_name="bfloat16"):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    from vllm.triton_utils import triton

    plot_name = f"fused-topk-perf-{scoring_func}"
    if dtype_name != "bfloat16":
        plot_name += f"-{dtype_name}"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "num_experts", "topk"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["torch", "vllm", "aiter_gating"],
            line_names=["Torch", "vLLM", "AITER topk_gating"],
            styles=[("blue", "-"), ("red", "-"), ("green", "-")],
            ylabel="us",
            plot_name=plot_name,
            args={},
        )
    )
    def benchmark(num_tokens, num_experts, topk, provider):
        dtype = getattr(torch, dtype_name)
        hidden_size = 1024
        renormalize = True
        hidden_states = torch.randn(
            (num_tokens, hidden_size), dtype=dtype, device="cuda"
        )
        gating_output = torch.randn(
            (num_tokens, num_experts), dtype=dtype, device="cuda"
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_topk(
                    gating_output=gating_output,
                    topk=topk,
                    renormalize=renormalize,
                    scoring_func=scoring_func,
                ),
                quantiles=quantiles,
            )
        elif provider == "aiter_gating":
            from vllm._aiter_ops import rocm_aiter_ops

            if scoring_func != "softmax" or not rocm_aiter_ops.is_topk_gating_enabled():
                return float("nan"), float("nan"), float("nan")
            topk_weights = torch.empty(
                (num_tokens, topk), dtype=torch.float32, device="cuda"
            )
            topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
            token_expert_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device="cuda"
            )
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rocm_aiter_ops.topk_gating(
                    topk_weights,
                    topk_ids,
                    token_expert_indices,
                    gating_output,
                    renormalize,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_topk(
                    hidden_states=hidden_states,
                    gating_output=gating_output,
                    topk=topk,
                    renormalize=renormalize,
                    scoring_func=scoring_func,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def build_chain(args: argparse.Namespace, num_tokens: int):
    """Build a fixed-input router GEMM -> top-k -> weight accumulation chain."""
    import vllm._custom_ops as ops

    gen = torch.Generator(device="cuda").manual_seed(0)
    gemm_dtype = getattr(torch, args.gemm_dtype)
    dtype = getattr(torch, args.dtype)
    hidden = torch.randn(
        num_tokens, args.hidden_size, dtype=gemm_dtype, device="cuda", generator=gen
    )
    routers = torch.randn(
        args.num_layers,
        args.hidden_size,
        args.num_experts,
        dtype=gemm_dtype,
        device="cuda",
        generator=gen,
    )
    routers.mul_(args.hidden_size**-0.5)
    router_weights = routers.unbind()
    logits = torch.empty(num_tokens, args.num_experts, dtype=gemm_dtype, device="cuda")
    gating = logits if dtype == gemm_dtype else torch.empty_like(logits, dtype=dtype)
    weights = torch.empty(num_tokens, args.topk, dtype=torch.float32, device="cuda")
    ids = torch.empty_like(weights, dtype=torch.int32)
    token_expert_indices = torch.empty_like(ids)
    acc = torch.zeros_like(weights)
    expected = torch.zeros_like(weights)
    topk_op = ops.topk_softmax if args.scoring_func == "softmax" else ops.topk_sigmoid

    for router in router_weights:
        reference_logits = (hidden @ router).to(dtype)
        reference_weights, _ = torch_topk(
            reference_logits, args.topk, True, args.scoring_func
        )
        expected.add_(reference_weights)

    def step():
        for router in router_weights:
            torch.mm(hidden, router, out=logits)
            if gating is not logits:
                gating.copy_(logits)
            topk_op(weights, ids, token_expert_indices, gating, True)
            acc.add_(weights)

    return step, acc, expected


def time_chain(step: Callable[[], None], iters: int) -> tuple[float, float]:
    """Return CUDA event interval and host enqueue time, in us per chain."""
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    host_start = time.perf_counter()
    for _ in range(iters):
        step()
    host_us = (time.perf_counter() - host_start) * 1e6 / iters
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / iters, host_us


def run_chain(args: argparse.Namespace):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("Chain mode requires an NVIDIA CUDA GPU.")

    # Adapted from Seungryong Kim's topk_chain_bench.py:
    # https://gist.github.com/srkim-business/ec5db57f7b78a9c9f06af65c891216c2
    from vllm import __version__

    lib = Path(
        importlib.import_module("vllm._moe_C_stable_libtorch").__file__
    ).resolve()
    with lib.open("rb") as source:
        lib_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    common = {
        "bench": "topk_chain",
        "run_id": uuid.uuid4().hex,
        "label": args.label,
        "round": args.round,
        "lib": str(lib),
        "lib_sha256": lib_sha256,
        "device": props.name,
        "cc": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": __version__,
        "command": [sys.executable, *sys.argv],
        "gating_dtype": args.dtype,
        "gemm_dtype": args.gemm_dtype,
        "scoring_func": args.scoring_func,
        "renormalize": True,
        "experts": args.num_experts,
        "topk": args.topk,
        "hidden": args.hidden_size,
        "layers": args.num_layers,
        "timer": "cuda_event",
        "cache_policy": "reuse",
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "warmup_ms": args.warmup_ms,
    }
    output = (
        Path(args.output_json)
        if args.output_json
        else (Path(args.save_path) / "topk_chain.jsonl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(common))
    with output.open("a") as out:
        for num_tokens in args.num_tokens:
            step, acc, expected = build_chain(args, num_tokens)
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            acc.zero_()
            step()
            torch.testing.assert_close(acc, expected, rtol=1e-4, atol=1e-4)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            acc.zero_()
            graph.replay()
            torch.testing.assert_close(acc, expected, rtol=1e-4, atol=1e-4)
            time_chain(graph.replay, 20)
            time_chain(step, 5)
            warmup_end = time.perf_counter() + args.warmup_ms / 1e3
            while time.perf_counter() < warmup_end:
                time_chain(graph.replay, 20)
            for repeat in range(args.repeats):
                for mode, run, iters in (
                    ("graph", graph.replay, args.replays),
                    ("eager", step, args.eager_iters),
                ):
                    gpu_us, host_us = time_chain(run, iters)
                    row = {
                        **common,
                        "tokens": num_tokens,
                        "exec": mode,
                        "repeat": repeat,
                        "iters": iters,
                        "us_per_chain": gpu_us,
                        "host_us_per_chain": host_us,
                        "us_per_layer": gpu_us / args.num_layers,
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(json.dumps(row))
            torch.accelerator.synchronize()
            del graph, run, step, acc, expected
    print(f"Raw samples appended to {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FlexibleArgumentParser(description="Benchmark the MoE topk kernel.")
    parser.add_argument("--mode", choices=["kernel", "chain"], default="kernel")
    parser.add_argument(
        "--scoring-func", choices=["softmax", "sigmoid"], default="softmax"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Gating input dtype (both benchmark modes).",
    )
    parser.add_argument("--save-path", type=str, default="./configs/fused_topk/")
    chain = parser.add_argument_group("chain mode")
    chain.add_argument("--num-tokens", type=int, nargs="+", default=[1, 16, 48])
    chain.add_argument("--num-experts", type=int, default=256)
    chain.add_argument("--topk", type=int, default=8)
    chain.add_argument("--hidden-size", type=int, default=4096)
    chain.add_argument("--num-layers", type=int, default=48)
    chain.add_argument(
        "--gemm-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Router GEMM dtype; a cast to --dtype is timed when different.",
    )
    chain.add_argument("--replays", type=int, default=200)
    chain.add_argument("--eager-iters", type=int, default=50)
    chain.add_argument("--repeats", type=int, default=5)
    chain.add_argument("--warmup-ms", type=int, default=2000)
    chain.add_argument(
        "--label", default="installed", help="Build label for comparison."
    )
    chain.add_argument("--round", type=int, default=0, help="Paired comparison round.")
    chain.add_argument(
        "--output-json",
        help="Append JSONL samples (default: SAVE_PATH/topk_chain.jsonl).",
    )
    args = parser.parse_args(argv)
    if args.mode == "chain":
        counts = [
            *args.num_tokens,
            args.num_experts,
            args.topk,
            args.hidden_size,
            args.num_layers,
            args.replays,
            args.eager_iters,
            args.repeats,
        ]
        if any(count <= 0 for count in counts):
            parser.error("Chain dimensions and iteration counts must be positive.")
        if args.topk > args.num_experts:
            parser.error("--topk must not exceed --num-experts.")
        if args.warmup_ms < 0 or args.round < 0:
            parser.error("--warmup-ms and --round must be nonnegative.")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "chain":
        run_chain(args)
    else:
        benchmark = get_benchmark(args.scoring_func, args.dtype)
        benchmark.run(print_data=True, save_path=args.save_path)
