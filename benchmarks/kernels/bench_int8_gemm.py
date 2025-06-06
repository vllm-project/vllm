# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import copy
import itertools

import torch
from weight_shapes import WEIGHT_SHAPES

from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_int8_quant as vllm_scaled_int8_quant
from vllm.triton_utils import triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        x_log=False,
        line_arg="provider",
        line_vals=[
            "torch-bf16",
            # "int8-tensor-w-token-a",
            "int8-tensor-w-tensor-a",
            "int8-channel-w-token-a",
            # "int8-channel-w-tensor-a",
            # "int8-tensor-w-token-a-noquant",
            "int8-tensor-w-tensor-a-noquant",
            "int8-channel-w-token-a-noquant",
            # "int8-channel-w-tensor-a-noquant",
        ],
        line_names=[
            "torch-bf16",
            # "int8-tensor-w-token-a",
            "int8-tensor-w-tensor-a",
            "int8-channel-w-token-a",
            # "int8-channel-w-tensor-a",
            # "int8-tensor-w-token-a-noquant",
            "int8-tensor-w-tensor-a-noquant",
            "int8-channel-w-token-a-noquant",
            # "int8-channel-w-tensor-a-noquant",
        ],
        ylabel="TFLOP/s (larger is better)",
        plot_name="BF16 vs INT8 GEMMs",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((N, K), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if "torch-bf16" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b), quantiles=quantiles
        )

    elif "int8" in provider:
        # Weights are always quantized ahead of time
        if "noquant" in provider:
            # For "no quant", we don't measure the time for activations
            if "tensor-w-token-a" in provider:
                # Dynamic per-token quant for A, static per-tensor quant for B
                scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b, scale_b)
                assert scale_b_int8.numel() == 1
                a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a)

            elif "tensor-w-tensor-a" in provider:
                # Static per-tensor quantization with fixed scales for both A and B
                scale_a = torch.tensor([1.0], device=device, dtype=torch.float32)
                scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b, scale_b)
                assert scale_b_int8.numel() == 1
                a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a, scale_a)

            elif "channel-w-token-a" in provider:
                # Dynamic per-channel quantization for weights, per-token quant for A
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b)
                assert scale_b_int8.numel() == N
                a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a)

            elif "channel-w-tensor-a" in provider:
                # Dynamic per-channel quantization for weights, per-tensor quant for A
                scale_a = torch.tensor([1.0], device=device, dtype=torch.float32)
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b)
                assert scale_b_int8.numel() == N
                a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a, scale_a)

            def run_quant():
                return vllm_scaled_mm(a_int8, b_int8, scale_a_int8, scale_b_int8, dtype)

        else:
            # Quantize the activations during the GEMM call
            if "tensor-w-token-a" in provider:
                # Dynamic per-token quant for A, static per-tensor quant for B
                scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b, scale_b)
                assert scale_b_int8.numel() == 1

                def run_quant():
                    a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a)
                    return vllm_scaled_mm(
                        a_int8, b_int8, scale_a_int8, scale_b_int8, dtype
                    )

            elif "tensor-w-tensor-a" in provider:
                # Static per-tensor quantization with fixed scales for both A and B
                scale_a = torch.tensor([1.0], device=device, dtype=torch.float32)
                scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b, scale_b)
                assert scale_b_int8.numel() == 1

                def run_quant():
                    a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a, scale_a)
                    return vllm_scaled_mm(
                        a_int8, b_int8, scale_a_int8, scale_b_int8, dtype
                    )

            elif "channel-w-token-a" in provider:
                # Dynamic per-channel quant for weights, per-token quant for A
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b)
                assert scale_b_int8.numel() == N

                def run_quant():
                    a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a)
                    return vllm_scaled_mm(
                        a_int8, b_int8, scale_a_int8, scale_b_int8, dtype
                    )

            elif "channel-w-tensor-a" in provider:
                # Dynamic per-channel quant for weights, static per-tensor quant for A
                scale_a = torch.tensor([1.0], device=device, dtype=torch.float32)
                b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b)
                assert scale_b_int8.numel() == N

                def run_quant():
                    a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a, scale_a)
                    return vllm_scaled_mm(
                        a_int8, b_int8, scale_a_int8, scale_b_int8, dtype
                    )

        b_int8 = b_int8.t()

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: run_quant(), quantiles=quantiles
        )

    # Calculate TFLOP/s, two flops per multiply-add
    tflops = lambda ms: (2 * M * N * K) * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def prepare_shapes(args):
    KN_model_names = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        assert model in WEIGHT_SHAPES
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KN.append(model)
            KN_model_names.append(KN)
    return KN_model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        choices=[*WEIGHT_SHAPES.keys()],
        help="List of models to benchmark",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="List of tensor parallel sizes",
    )
    args = parser.parse_args()

    KN_model_names = prepare_shapes(args)
    for K, N, model_name in KN_model_names:
        print(f"{model_name}, N={N} K={K}, BF16 vs INT8 GEMMs TFLOP/s:")
        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path=f"bench_int8_res_n{N}_k{K}",
            N=N,
            K=K,
        )

    print("Benchmark finished!")
