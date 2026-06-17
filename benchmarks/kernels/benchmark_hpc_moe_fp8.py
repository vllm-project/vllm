# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the performance of the HPC FP8 MoE backend (HPCExperts) against the
Triton and CUTLASS FP8 fused-MoE kernels. All kernels take fp8 quantized
weights and 16-bit activations with per-tensor FP8 quantization.

Example usage:
    python benchmark_hpc_moe_fp8.py \
        --models custom-small \
        --tp-sizes 1 \
        --batch-sizes 8 16 32
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.hpc_moe import HPCExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hpc import has_hpc
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
    "custom-small",
]

DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_TP_SIZES = [1]

FP8_DTYPE = current_platform.fp8_dtype()


def bench_run(
    model: str,
    num_experts: int,
    topk: int,
    mkn: tuple[int, int, int],
    bench_triton: bool,
    bench_cutlass: bool,
):
    init_workspace_manager(torch.accelerator.current_device_index())
    (m, k, n) = mkn

    dtype = torch.bfloat16
    device = "cuda"

    # Create input activations
    a = torch.randn((m, k), device=device, dtype=dtype) / 10

    # Create weights
    w1 = torch.randn((num_experts, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device=device, dtype=dtype) / 10

    # Create FP8 per-tensor quantized weights and scales
    w1_fp8q = torch.empty((num_experts, 2 * n, k), device=device, dtype=FP8_DTYPE)
    w2_fp8q = torch.empty((num_experts, k, n), device=device, dtype=FP8_DTYPE)
    w1_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)
    w2_scale = torch.empty((num_experts, 1, 1), device=device, dtype=torch.float32)

    for expert in range(num_experts):
        w1_fp8q[expert], w1_scale_temp = ops.scaled_fp8_quant(w1[expert])
        w2_fp8q[expert], w2_scale_temp = ops.scaled_fp8_quant(w2[expert])
        w1_scale[expert, 0, 0] = w1_scale_temp
        w2_scale[expert, 0, 0] = w2_scale_temp

    # Create router scores and get topk
    score = torch.randn((m, num_experts), device=device, dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

    # Force per-tensor activation quantization (matches the e2e setup).
    a1_scale = torch.full((), 1e-2, device=device, dtype=torch.float32)
    a2_scale = torch.full((), 1e-2, device=device, dtype=torch.float32)

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=n,
        in_dtype=a.dtype,
    )

    # ---- HPC quant config (single dq scale + inverse a2 scale, like FI). ----
    hpc_quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        g1_alphas=(w1_scale * a1_scale).squeeze(),
        w2_scale=w2_scale,
        g2_alphas=(w2_scale * a2_scale).squeeze(),
        a1_scale=a1_scale,
        a1_gscale=a1_scale,
        a2_scale=a2_scale,
        a2_gscale=1.0 / a2_scale,
        per_act_token_quant=False,
    )
    hpc_fn = mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        HPCExperts(
            moe_config=moe_config,
            quant_config=hpc_quant_config,
        ),
    )

    # ---- Standard per-tensor quant config for Triton / CUTLASS. ----
    std_quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=False,
    )

    def capture_graph(call_fn):
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            # Capture 10 invocations like benchmark_moe.py
            for _ in range(10):
                call_fn()
        torch.accelerator.synchronize()
        return graph

    # HPC graph
    hpc_graph = capture_graph(
        lambda: hpc_fn.apply(
            a,
            w1_fp8q,
            w2_fp8q,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
    )

    triton_graph = None
    if bench_triton:
        triton_graph = capture_graph(
            lambda: fused_experts(
                a,
                w1_fp8q,
                w2_fp8q,
                topk_weights,
                topk_ids,
                quant_config=std_quant_config,
            )
        )

    cutlass_graph = None
    if bench_cutlass:
        cutlass_fn = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=std_quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            CutlassExpertsFp8(
                moe_config=moe_config,
                quant_config=std_quant_config,
            ),
        )
        cutlass_graph = capture_graph(
            lambda: cutlass_fn.apply(
                a,
                w1_fp8q,
                w2_fp8q,
                topk_weights,
                topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
        )

    def bench_cuda_graph(graph, num_warmup=5, num_iters=100):
        """Benchmark CUDA graph using events like benchmark_moe.py"""
        for _ in range(num_warmup):
            graph.replay()
        torch.accelerator.synchronize()

        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)

        latencies = []
        for _ in range(num_iters):
            torch.accelerator.synchronize()
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

        # Divide by 10 since graph contains 10 calls
        return sum(latencies) / (num_iters * 10)

    num_warmup = 5
    num_iters = 100

    # us = ms * 1000
    hpc_time_us = (
        bench_cuda_graph(hpc_graph, num_warmup=num_warmup, num_iters=num_iters) * 1000
    )
    result = {"batch_size": m, "hpc_time_us": hpc_time_us}
    if triton_graph is not None:
        result["triton_time_us"] = (
            bench_cuda_graph(triton_graph, num_warmup=num_warmup, num_iters=num_iters)
            * 1000
        )
    if cutlass_graph is not None:
        result["cutlass_time_us"] = (
            bench_cuda_graph(cutlass_graph, num_warmup=num_warmup, num_iters=num_iters)
            * 1000
        )

    return result


def main(args):
    if not has_hpc():
        raise RuntimeError(
            "The hpc (hpc-ops) package is not available; cannot benchmark the "
            "HPC MoE backend."
        )

    # Initialize workspace manager (required for the modular MoE kernels)
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

                print(
                    f"\n=== {model}, experts={num_experts}, topk={topk}, "
                    f"tp={tp} (n={size_n}, k={size_k}) ==="
                )

                config_results = []
                for size_m in args.batch_sizes:
                    mkn = (size_m, size_k, size_n)
                    result = bench_run(
                        model,
                        num_experts,
                        topk,
                        mkn,
                        bench_triton=not args.no_triton,
                        bench_cutlass=not args.no_cutlass,
                    )
                    if result:
                        config_results.append(result)

                if config_results:
                    header = f"\n{'Batch Size':<12}{'HPC (us)':<15}"
                    if not args.no_triton:
                        header += f"{'Triton (us)':<15}"
                    if not args.no_cutlass:
                        header += f"{'CUTLASS (us)':<15}"
                    print(header)
                    print("-" * len(header))
                    for result in config_results:
                        line = (
                            f"{result['batch_size']:<12}{result['hpc_time_us']:<15.2f}"
                        )
                        if "triton_time_us" in result:
                            line += f"{result['triton_time_us']:<15.2f}"
                        if "cutlass_time_us" in result:
                            line += f"{result['cutlass_time_us']:<15.2f}"
                        print(line)

                    all_results.extend(config_results)

    print(f"\nTotal benchmarks completed: {len(all_results)}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="""Benchmark HPC FP8 MoE backend vs Triton / CUTLASS FP8
         fused MoE across specified models/shapes/batches.

        Example usage:
        python benchmark_hpc_moe_fp8.py \
            --models custom-small \
            --tp-sizes 1 \
            --batch-sizes 8 16 32
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
        "--no-triton",
        action="store_true",
        help="Skip the Triton fused-MoE baseline.",
    )
    parser.add_argument(
        "--no-cutlass",
        action="store_true",
        help="Skip the CUTLASS fused-MoE baseline.",
    )

    args = parser.parse_args()
    main(args)
