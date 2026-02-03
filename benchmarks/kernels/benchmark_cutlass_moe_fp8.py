# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the performance of the cutlass_moe_fp8 kernel vs the triton_moe
kernel. Both kernels take in fp8 quantized weights and 16-bit activations,
but use different quantization strategies and backends.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.workspace import init_workspace_manager

# Weight shapes for different models: [num_experts, topk, hidden_size,
# intermediate_size]
WEIGHT_SHAPES_MOE = {
    "mixtral-8x7b": [
        [8, 2, 4096, 14336],
    ],
    "deepseek-v2": [
        [160, 6, 5120, 12288],
    ],
    "custom-small": [
        [8, 2, 2048, 7168],
    ],
    "glm45-fp8": [
        [128, 8, 4096, 1408],
    ],
    "Llama-4-Maverick-17B-128E-Instruct-FP8": [
        [128, 1, 5120, 8192],
    ],
}

DEFAULT_MODELS = [
    "mixtral-8x7b",
]

DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_TP_SIZES = [1]

PER_ACT_TOKEN_OPTS = [False, True]
PER_OUT_CH_OPTS = [False, True]

FP8_DTYPE = current_platform.fp8_dtype()


def bench_run(
    results: list,
    model: str,
    num_experts: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    mkn: tuple[int, int, int],
):
    init_workspace_manager(torch.cuda.current_device())
    (m, k, n) = mkn

    dtype = torch.half
    device = "cuda"

    # Create input activations
    a = torch.randn((m, k), device=device, dtype=dtype) / 10

    # Create weights
    w1 = torch.randn((num_experts, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device=device, dtype=dtype) / 10

    # Create FP8 quantized weights and scales for both kernels
    w1_fp8q = torch.empty((num_experts, 2 * n, k), device=device, dtype=FP8_DTYPE)
    w2_fp8q = torch.empty((num_experts, k, n), device=device, dtype=FP8_DTYPE)

    # Create scales based on quantization strategy
    if per_out_ch:
        # Per-channel quantization
        w1_scale = torch.empty(
            (num_experts, 2 * n, 1), device=device, dtype=torch.float32
        )
        w2_scale = torch.empty((num_experts, k, 1), device=device, dtype=torch.float32)
    else:
        # Per-tensor quantization
        w1_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)
        w2_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)

    # Quantize weights
    for expert in range(num_experts):
        if per_out_ch:
            # Per-channel quantization - not yet implemented properly
            # For now, fall back to per-tensor quantization
            w1_fp8q[expert], w1_scale_temp = ops.scaled_fp8_quant(w1[expert])
            w2_fp8q[expert], w2_scale_temp = ops.scaled_fp8_quant(w2[expert])
            # Expand scalar scales to the expected per-channel shape
            w1_scale[expert] = w1_scale_temp.expand(2 * n, 1)
            w2_scale[expert] = w2_scale_temp.expand(k, 1)
        else:
            # Per-tensor quantization
            w1_fp8q[expert], w1_scale_temp = ops.scaled_fp8_quant(w1[expert])
            w2_fp8q[expert], w2_scale_temp = ops.scaled_fp8_quant(w2[expert])
            # Store scalar scales in [1, 1] tensors
            w1_scale[expert, 0, 0] = w1_scale_temp
            w2_scale[expert, 0, 0] = w2_scale_temp

    # Prepare weights for CUTLASS (no transpose needed)
    w1_fp8q_cutlass = w1_fp8q  # Keep original [E, 2N, K]
    w2_fp8q_cutlass = w2_fp8q  # Keep original [E, K, N]

    # Create router scores and get topk
    score = torch.randn((m, num_experts), device=device, dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

    # WORKAROUND: CUTLASS MoE FP8 has issues with per-token quantization
    # Force per-tensor quantization for all cases to match working e2e setup
    a1_scale = torch.full((), 1e-2, device=device, dtype=torch.float32)
    a2_scale = torch.full((), 1e-2, device=device, dtype=torch.float32)

    # Force per-tensor quantization for all cases
    per_act_token = False

    # Pre-create quantization config to avoid creating it inside CUDA graph
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=per_act_token,
        per_out_ch_quant=per_out_ch,
    )

    fn = mk.FusedMoEKernel.make_mk(
        MoEPrepareAndFinalizeNoEP(),
        CutlassExpertsFp8(
            moe_config=make_dummy_moe_config(
                num_experts=num_experts,
                hidden_dim=k,
                intermediate_size_per_partition=n,
                in_dtype=a.dtype,
            ),
            quant_config=quant_config,
        ),
    )

    # Create CUDA graphs for CUTLASS (match benchmark_moe.py pattern exactly)
    cutlass_stream = torch.cuda.Stream()
    cutlass_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cutlass_graph, stream=cutlass_stream):
        # Capture 10 invocations like benchmark_moe.py
        for _ in range(10):
            fn(
                a,
                w1_fp8q_cutlass,
                w2_fp8q_cutlass,
                topk_weights,
                topk_ids,
                activation="silu",
                global_num_experts=num_experts,
            )
    torch.cuda.synchronize()

    # Create CUDA graphs for Triton (match benchmark_moe.py pattern exactly)
    triton_stream = torch.cuda.Stream()
    triton_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_graph, stream=triton_stream):
        # Capture 10 invocations like benchmark_moe.py
        for _ in range(10):
            fused_experts(
                a,
                w1_fp8q,
                w2_fp8q,
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            )
    torch.cuda.synchronize()

    def bench_cuda_graph(graph, num_warmup=5, num_iters=100):
        """Benchmark CUDA graph using events like benchmark_moe.py"""
        # Warmup
        for _ in range(num_warmup):
            graph.replay()
        torch.cuda.synchronize()

        # Timing
        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)

        latencies = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

        # Divide by 10 since graph contains 10 calls
        return sum(latencies) / (num_iters * 10)

    # Benchmark parameters
    num_warmup = 5
    num_iters = 100

    # Benchmark only CUDA graphs (more reliable and faster)
    # Benchmark Triton MoE with CUDA graphs
    triton_graph_time = bench_cuda_graph(
        triton_graph, num_warmup=num_warmup, num_iters=num_iters
    )

    # Benchmark CUTLASS MoE with CUDA graphs
    cutlass_graph_time = bench_cuda_graph(
        cutlass_graph, num_warmup=num_warmup, num_iters=num_iters
    )

    # Convert ms to us and return results
    triton_time_us = triton_graph_time * 1000
    cutlass_time_us = cutlass_graph_time * 1000

    return {
        "batch_size": m,
        "triton_time_us": triton_time_us,
        "cutlass_time_us": cutlass_time_us,
    }


def main(args):
    # Initialize workspace manager (required for CUTLASS MoE kernels)
    device = torch.device("cuda:0")
    init_workspace_manager(device)

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    all_results = []

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

                for per_act_token in args.per_act_token_opts:
                    for per_out_ch in args.per_out_ch_opts:
                        print(
                            f"\n=== {model}, experts={num_experts}, topk={topk},"
                            f"per_act={per_act_token}, per_out_ch={per_out_ch} ==="
                        )

                        config_results = []
                        for size_m in args.batch_sizes:
                            mkn = (size_m, size_k, size_n)
                            result = bench_run(
                                [],  # Not used anymore
                                model,
                                num_experts,
                                topk,
                                per_act_token,
                                per_out_ch,
                                mkn,
                            )
                            if result:
                                config_results.append(result)

                        # Print results table for this configuration
                        if config_results:
                            print(
                                f"\n{'Batch Size':<12}"
                                f"{'Triton (us)':<15}"
                                f"{'CUTLASS (us)':<15}"
                            )
                            print("-" * 45)
                            for result in config_results:
                                print(
                                    f"{result['batch_size']:<12}"
                                    f"{result['triton_time_us']:<15.2f}"
                                    f"{result['cutlass_time_us']:<15.2f}"
                                )

                            all_results.extend(config_results)

    print(f"\nTotal benchmarks completed: {len(all_results)}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="""Benchmark CUTLASS FP8 MOE vs Triton FP8 FUSED MOE
         across specified models/shapes/batches

        Example usage:
        python benchmark_cutlass_moe_fp8.py  \
            --model "Llama-4-Maverick-17B-128E-Instruct-FP8"  \
            --tp-sizes 8 \
            --batch-size 2 4 8  \
            --per-act-token-opts false \
            --per-out-ch-opts false

        """
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
    parser.add_argument(
        "--per-act-token-opts",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Per-activation token quantization options (true/false)",
    )
    parser.add_argument(
        "--per-out-ch-opts",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Per-output channel quantization options (true/false)",
    )

    args = parser.parse_args()
    main(args)
