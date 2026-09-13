# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import statistics

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

num_tokens_range = [2**i for i in range(0, 8, 2)]
num_experts_range = [16, 32, 64, 128, 256, 512]
topk_range = [3, 4]
configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))

A100_SMALL_TOPK_TOKENS = [1, 2, 4, 8, 16, 32, 64]


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


def get_benchmark(scoring_func):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "num_experts", "topk"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["torch", "vllm"],
            line_names=["Torch", "vLLM"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"fused-topk-perf-{scoring_func}",
            args={},
        )
    )
    def benchmark(num_tokens, num_experts, topk, provider):
        dtype = torch.bfloat16
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


def benchmark_a100_small_topk():
    """Compare the current fallback with the SM80 E=60 integrated path."""
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("--a100-small requires an SM80 GPU")

    def measure(fn):
        rounds = [
            triton.testing.do_bench(
                fn,
                warmup=100,
                rep=500,
                quantiles=[0.5, 0.2, 0.8],
            )
            for _ in range(5)
        ]
        return tuple(
            statistics.median(result[index] * 1000 for result in rounds)
            for index in range(3)
        )

    torch.manual_seed(20260827)
    for num_tokens in A100_SMALL_TOPK_TOKENS:
        hidden_states = torch.empty(
            (num_tokens, 1), dtype=torch.bfloat16, device="cuda"
        )
        gating_output = torch.randn(
            (num_tokens, 60), dtype=torch.bfloat16, device="cuda"
        )

        def fallback(num_tokens=num_tokens, gating_output=gating_output):
            weights = torch.empty((num_tokens, 4), dtype=torch.float32, device="cuda")
            ids = torch.empty((num_tokens, 4), dtype=torch.int32, device="cuda")
            source_rows = torch.empty_like(ids)
            ops.topk_softmax(
                weights,
                ids,
                source_rows,
                gating_output,
                False,
            )
            return weights, ids, source_rows

        def integrated(hidden_states=hidden_states, gating_output=gating_output):
            return fused_topk(hidden_states, gating_output, 4, False)

        reference = fallback()
        actual = integrated()
        torch.accelerator.synchronize()
        assert all(torch.equal(ref, out) for ref, out in zip(reference, actual))

        baseline = measure(fallback)
        optimized = measure(integrated)
        gain = (1 - optimized[0] / baseline[0]) * 100
        print(
            f"M={num_tokens:2d} baseline={baseline[0]:.3f} us "
            f"optimized={optimized[0]:.3f} us gain={gain:.2f}%"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the MoE topk kernel.")
    parser.add_argument("--scoring-func", type=str, default="softmax")
    parser.add_argument("--save-path", type=str, default="./configs/fused_topk/")
    parser.add_argument("--a100-small", action="store_true")
    args = parser.parse_args()

    if args.a100_small:
        benchmark_a100_small_topk()
        raise SystemExit

    # Get the benchmark function
    benchmark = get_benchmark(args.scoring_func)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
