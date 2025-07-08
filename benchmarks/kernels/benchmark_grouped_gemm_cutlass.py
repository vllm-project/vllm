# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.utils.benchmark as benchmark
from benchmark_shapes import WEIGHT_SHAPES_MOE

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    fused_topk,
)
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = [
    "nm-testing/Mixtral-8x7B-Instruct-v0.1",
    "nm-testing/deepseekv2-lite",
    "ibm-granite/granite-3.0-1b-a400m",
    "ibm-granite/granite-3.0-3b-a800m",
]
DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]

PER_ACT_TOKEN_OPTS = [False]
PER_OUT_CH_OPTS = [False]


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def bench_run(
    results: list[benchmark.Measurement],
    model: str,
    num_experts: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    mkn: tuple[int, int, int],
):
    label = "Quant Matmul"

    sub_label = (
        "{}, num_experts={}, topk={}, per_act_token={} per_out_ch={}, MKN=({})".format(
            model, num_experts, topk, per_act_token, per_out_ch, mkn
        )
    )

    print(f"Testing: {sub_label}")

    (m, k, n) = mkn

    dtype = torch.half

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((num_experts, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device="cuda", dtype=dtype) / 10

    _, a_scale = ops.scaled_fp8_quant(a)

    w1_q = torch.empty(
        (num_experts, 2 * n, k), device="cuda", dtype=torch.float8_e4m3fn
    )
    w2_q = torch.empty((num_experts, k, n), device="cuda", dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty((num_experts, 1, 1), device="cuda", dtype=torch.float32)
    w2_scale = torch.empty((num_experts, 1, 1), device="cuda", dtype=torch.float32)

    for expert in range(num_experts):
        w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(w1[expert])
        w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(w2[expert])

    score = torch.randn((m, num_experts), device="cuda", dtype=dtype)

    topk_weights, topk_ids, token_expert_indices = fused_topk(
        a, score, topk, renormalize=False
    )

    def run_triton_moe(
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a_scale: torch.Tensor,
        num_repeats: int,
    ):
        for _ in range(num_repeats):
            fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a_scale,
            )

    def run_cutlass_moe(
        a: torch.Tensor,
        a_scale: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        per_act_token: bool,
        num_repeats: int,
    ):
        for _ in range(num_repeats):
            cutlass_moe_fp8(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                w1_scale,
                w2_scale,
                per_act_token,
                a1_scale=None,
            )

    def run_cutlass_from_graph(
        a: torch.Tensor,
        a_scale: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
        ):
            return cutlass_moe_fp8(
                a,
                w1_q,
                w2_q,
                topk_weights,
                topk_ids,
                w1_scale,
                w2_scale,
                per_act_token,
                a1_scale=None,
            )

    def run_triton_from_graph(
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a_scale: torch.Tensor,
    ):
        with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
        ):
            return fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a_scale,
            )

    def replay_graph(graph, num_repeats):
        for _ in range(num_repeats):
            graph.replay()
        torch.cuda.synchronize()

    cutlass_stream = torch.cuda.Stream()
    cutlass_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cutlass_graph, stream=cutlass_stream):
        run_cutlass_from_graph(
            a,
            a_scale,
            w1_q,
            w2_q,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
        )
    torch.cuda.synchronize()

    triton_stream = torch.cuda.Stream()
    triton_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_graph, stream=triton_stream):
        run_triton_from_graph(
            a,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            a_scale,
        )
    torch.cuda.synchronize()

    min_run_time = 5
    num_warmup = 5
    num_runs = 25

    globals = {
        # Baseline params
        "w1": w1,
        "w2": w2,
        "score": score,
        "topk": topk,
        # Cutlass params
        "a_scale": a_scale,
        "w1_q": w1_q,
        "w2_q": w2_q,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "per_act_token": per_act_token,
        # cuda graph params
        "cutlass_graph": cutlass_graph,
        "triton_graph": triton_graph,
        # Gen params
        "a": a,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "num_runs": num_runs,
        # Kernels
        "run_triton_moe": run_triton_moe,
        "run_cutlass_moe": run_cutlass_moe,
        "replay_graph": replay_graph,
    }

    # Warmup
    run_triton_moe(
        a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        a_scale,
        num_warmup,
    )

    results.append(
        benchmark.Timer(
            stmt="run_triton_moe(a, w1_q, w2_q, topk_weights, topk_ids, w1_scale, w2_scale, a_scale, num_runs)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="triton_moe",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    # Warmup
    replay_graph(triton_graph, num_warmup)

    results.append(
        benchmark.Timer(
            stmt="replay_graph(triton_graph, num_runs)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="triton_moe_cuda_graphs",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    # Warmup
    run_cutlass_moe(
        a,
        a_scale,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        per_act_token,
        num_warmup,
    )

    results.append(
        benchmark.Timer(
            stmt="run_cutlass_moe(a, a_scale, w1_q, w2_q, w1_scale, w2_scale, topk_weights, topk_ids, per_act_token, num_runs)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="grouped_gemm_moe",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    # Warmup
    replay_graph(cutlass_graph, num_warmup)

    results.append(
        benchmark.Timer(
            stmt="replay_graph(cutlass_graph, num_runs)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="grouped_gemm_moe_cuda_graphs",
        ).blocked_autorange(min_run_time=min_run_time)
    )


def main(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    results: list[benchmark.Measurement] = []

    for model in args.models:
        for tp in args.tp_sizes:
            for layer in WEIGHT_SHAPES_MOE[model]:
                num_experts = layer[0]
                topk = layer[1]
                size_k = layer[2]
                size_n = layer[3] // tp

                if len(args.limit_k) > 0 and size_k not in args.limit_k:
                    continue

                if len(args.limit_n) > 0 and size_n not in args.limit_n:
                    continue

                for per_act_token in PER_ACT_TOKEN_OPTS:
                    for per_out_ch in PER_OUT_CH_OPTS:
                        for size_m in DEFAULT_BATCH_SIZES:
                            mkn = (size_m, size_k, size_n)
                            bench_run(
                                results,
                                model,
                                num_experts,
                                topk,
                                per_act_token,
                                per_out_ch,
                                mkn,
                            )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark Marlin across specified models/shapes/batches"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES_MOE.keys(),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=DEFAULT_TP_SIZES)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--limit-k", nargs="+", type=int, default=[])
    parser.add_argument("--limit-n", nargs="+", type=int, default=[])
    parser.add_argument("--limit-num-groups", nargs="+", type=int, default=[])
    parser.add_argument("--limit-per-act-token", nargs="+", type=int, default=[])
    parser.add_argument("--limit-per-out-ch", nargs="+", type=int, default=[])

    args = parser.parse_args()
    main(args)
