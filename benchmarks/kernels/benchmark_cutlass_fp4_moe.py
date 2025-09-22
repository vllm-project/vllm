# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the performance of the cutlass_moe_fp4 kernel vs the triton_moe
kernel. The cutlass_moe_fp4 kernel takes in fp4 quantized weights and 16-bit
activations. The triton_moe kernel takes in fp8 weights(tensor scaled to fp8)
and 16-bit activations.
"""

import nvtx
import torch
import torch.utils.benchmark as benchmark

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
from vllm.scalar_type import scalar_types
from vllm.utils import FlexibleArgumentParser

WEIGHT_SHAPES_MOE = {
    "nvidia/DeepSeek-R1-FP4": [
        [256, 8, 2048, 7168],
    ],
}

DEFAULT_MODELS = [
    "nvidia/DeepSeek-R1-FP4",
]

DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_TP_SIZES = [1]

PER_ACT_TOKEN_OPTS = [False]
PER_OUT_CH_OPTS = [False]
FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


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
    label = "NVFP4 Blockscaled CUTLASS MOE vs FP8 Tensor Scaled Triton"

    sub_label = (
        "{}, num_experts={}, topk={}, per_act_token={} per_out_ch={}, MKN=({})".format(
            model, num_experts, topk, per_act_token, per_out_ch, mkn
        )
    )

    print(f"Testing: {sub_label}")

    (m, k, n) = mkn

    dtype = torch.half
    device = "cuda"
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((num_experts, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device=device, dtype=dtype) / 10

    _, a_fp8_scale = ops.scaled_fp8_quant(a)

    w1_fp8q = torch.empty(
        (num_experts, 2 * n, k), device=device, dtype=torch.float8_e4m3fn
    )
    w2_fp8q = torch.empty((num_experts, k, n), device=device, dtype=torch.float8_e4m3fn)
    w1_fp8scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)
    w2_fp8scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)

    for expert in range(num_experts):
        w1_fp8q[expert], w1_fp8scale[expert] = ops.scaled_fp8_quant(w1[expert])
        w2_fp8q[expert], w2_fp8scale[expert] = ops.scaled_fp8_quant(w2[expert])

    w1_fp8q_notransp = w1_fp8q.clone()
    w2_fp8q_notransp = w2_fp8q.clone()
    w1_fp8q = w1_fp8q.transpose(1, 2)
    w2_fp8q = w2_fp8q.transpose(1, 2)

    score = torch.randn((m, num_experts), device=device, dtype=dtype)

    topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

    quant_blocksize = 16
    w1_blockscale = torch.empty(
        (num_experts, 2 * n, k // quant_blocksize),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w2_blockscale = torch.empty(
        (num_experts, k, n // quant_blocksize), device=device, dtype=torch.float8_e4m3fn
    )

    # n_b_scales = 2 * n if per_out_ch else 1
    # k_b_scales = k if per_out_ch else 1
    w1_fp4 = torch.empty((num_experts, 2 * n, k // 2), device=device, dtype=torch.uint8)
    w2_fp4 = torch.empty((num_experts, k, n // 2), device=device, dtype=torch.uint8)

    w1_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)
    w2_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)
    a1_gs = torch.ones((num_experts,), device=device, dtype=torch.float32)
    a2_gs = torch.ones((num_experts,), device=device, dtype=torch.float32)

    for expert in range(num_experts):
        w1_e = w1[expert]
        w2_e = w2[expert]
        w1_amax = torch.abs(w1_e).max().to(torch.float32)
        w2_amax = torch.abs(w2_e).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_fp4[expert], w1_blockscale[expert] = ops.scaled_fp4_quant(
            w1_e, w1_gs[expert]
        )

        w2_fp4[expert], w2_blockscale[expert] = ops.scaled_fp4_quant(
            w2_e, w2_gs[expert]
        )

    def run_triton_moe(
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a_fp8_scale: torch.Tensor,
        num_repeats: int,
    ):
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a_fp8_scale,
        )

        for _ in range(num_repeats):
            fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            )

    def run_cutlass_moe_fp4(
        a: torch.Tensor,
        w1_fp4: torch.Tensor,
        w2_fp4: torch.Tensor,
        w1_blockscale: torch.Tensor,
        w2_blockscale: torch.Tensor,
        w1_gs: torch.Tensor,
        w2_gs: torch.Tensor,
        a1_gs: torch.Tensor,
        a2_gs: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        m: int,
        n: int,
        k: int,
        e: int,
        device: torch.device,
        num_repeats: int,
    ):
        quant_config = nvfp4_moe_quant_config(
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
            g1_alphas=w1_gs,
            g2_alphas=w2_gs,
        )
        for _ in range(num_repeats):
            with nvtx.annotate("cutlass_moe_fp4", color="green"):
                cutlass_moe_fp4(
                    a=a,
                    w1_fp4=w1_fp4,
                    w2_fp4=w2_fp4,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    m=m,
                    n=n,
                    k=k,
                    e=num_experts,
                    quant_config=quant_config,
                )

    def run_cutlass_from_graph(
        a: torch.Tensor,
        a1_gscale: torch.Tensor,
        w1_fp4: torch.Tensor,
        w1_blockscale: torch.Tensor,
        w1_alphas: torch.Tensor,
        a2_gscale: torch.Tensor,
        w2_fp4: torch.Tensor,
        w2_blockscale: torch.Tensor,
        w2_alphas: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        m: int,
        n: int,
        k: int,
        e: int,
        device: torch.device,
    ):
        quant_config = nvfp4_moe_quant_config(
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
            g1_alphas=w1_gs,
            g2_alphas=w2_gs,
        )

        with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
        ):
            return cutlass_moe_fp4(
                a=a,
                w1_fp4=w1_fp4,
                w2_fp4=w2_fp4,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                m=m,
                n=n,
                k=k,
                e=num_experts,
                quant_config=quant_config,
            )

    def run_triton_from_graph(
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a_fp8_scale: torch.Tensor,
    ):
        with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
        ):
            quant_config = fp8_w8a8_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a_fp8_scale,
            )
            return fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            )

    def replay_graph(graph, num_repeats):
        for _ in range(num_repeats):
            graph.replay()
        torch.cuda.synchronize()

    cutlass_stream = torch.cuda.Stream()
    cutlass_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cutlass_graph, stream=cutlass_stream):
        run_cutlass_from_graph(
            a=a,
            a1_gscale=a1_gs,
            w1_fp4=w1_fp4,
            w1_blockscale=w1_blockscale,
            w1_alphas=w1_gs,
            a2_gscale=a2_gs,
            w2_fp4=w2_fp4,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_gs,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            m=m,
            n=n,
            k=k,
            e=num_experts,
            device=device,
        )
    torch.cuda.synchronize()

    triton_stream = torch.cuda.Stream()
    triton_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_graph, stream=triton_stream):
        run_triton_from_graph(
            a,
            w1_fp8q_notransp,
            w2_fp8q_notransp,
            topk_weights,
            topk_ids,
            w1_fp8scale,
            w2_fp8scale,
            a_fp8_scale,
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
        "w1_fp8q_notransp": w1_fp8q_notransp,
        "w2_fp8q_notransp": w2_fp8q_notransp,
        "w1_fp8scale": w1_fp8scale,
        "w2_fp8scale": w2_fp8scale,
        "a_fp8_scale": a_fp8_scale,
        # Cutlass params
        "a": a,
        "a1_gscale": a1_gs,
        "w1_fp4": w1_fp4,
        "w1_blockscale": w1_blockscale,
        "w1_alphas": w1_gs,
        "a2_gscale": a2_gs,
        "w2_fp4": w2_fp4,
        "w2_blockscale": w2_blockscale,
        "w2_alphas": w2_gs,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "m": m,
        "n": n,
        "k": k,
        "e": num_experts,
        "device": device,
        # cuda graph params
        "cutlass_graph": cutlass_graph,
        "triton_graph": triton_graph,
        # Gen params
        "num_runs": num_runs,
        # Kernels
        "run_triton_moe": run_triton_moe,
        "run_cutlass_moe_fp4": run_cutlass_moe_fp4,
        "replay_graph": replay_graph,
    }

    # Warmup
    run_triton_moe(
        a,
        w1_fp8q_notransp,
        w2_fp8q_notransp,
        topk_weights,
        topk_ids,
        w1_fp8scale,
        w2_fp8scale,
        a_fp8_scale,
        num_warmup,
    )

    results.append(
        benchmark.Timer(
            stmt="run_triton_moe(a, w1_fp8q_notransp, w2_fp8q_notransp, topk_weights, topk_ids, w1_fp8scale, w2_fp8scale, a_fp8_scale, num_runs)",  # noqa: E501
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

    run_cutlass_moe_fp4(
        a,
        w1_fp4,
        w2_fp4,
        w1_blockscale,
        w2_blockscale,
        w1_gs,
        w2_gs,
        a1_gs,
        a2_gs,
        topk_weights,
        topk_ids,
        m,
        n,
        k,
        num_experts,
        device,
        num_warmup,
    )

    results.append(
        benchmark.Timer(
            stmt="run_cutlass_moe_fp4(a, w1_fp4, w2_fp4, w1_blockscale, w2_blockscale, w1_alphas, w2_alphas, a1_gscale, a2_gscale, topk_weights, topk_ids, m, n, k, e, device, num_runs)",  # noqa: E501
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="cutlass_moe_fp4",
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
            description="cutlass_moe_fp4_cuda_graphs",
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
                        for size_m in args.batch_sizes:
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
        description="Benchmark NVFP4 CUTLASS MOE across specified models/shapes/batches"
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
