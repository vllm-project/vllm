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

PROVIDER_CFGS = {
    "torch-bf16": dict(enabled=True),
    "int8-tensor-w-token-a": dict(
        w="tensor", a="token", no_a_quant=False, enabled=False
    ),
    "int8-tensor-w-tensor-a": dict(
        w="tensor", a="tensor", no_a_quant=False, enabled=True
    ),
    "int8-channel-w-token-a": dict(
        w="channel", a="token", no_a_quant=False, enabled=True
    ),
    "int8-channel-w-tensor-a": dict(
        w="channel", a="tensor", no_a_quant=False, enabled=False
    ),
    "int8-tensor-w-token-a-noquant": dict(
        w="tensor", a="token", no_a_quant=True, enabled=False
    ),
    "int8-tensor-w-tensor-a-noquant": dict(
        w="tensor", a="tensor", no_a_quant=True, enabled=True
    ),
    "int8-channel-w-token-a-noquant": dict(
        w="channel", a="token", no_a_quant=True, enabled=True
    ),
    "int8-channel-w-tensor-a-noquant": dict(
        w="channel", a="tensor", no_a_quant=True, enabled=False
    ),
}


def _quant_weight(b, w_type, device):
    if w_type == "tensor":
        scale_b = torch.ones(1, device=device, dtype=torch.float32)
        b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b, scale_b)
        assert scale_b_int8.numel() == 1
    else:  # channel
        b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b)
        assert scale_b_int8.numel() == b.shape[0]
    return b_int8.t(), scale_b_int8


def build_int8_runner(cfg, a, b, dtype, device):
    # quant before running the kernel
    b_int8, scale_b_int8 = _quant_weight(b, cfg["w"], device)

    scale_a_const = None
    if cfg["a"] == "tensor":
        scale_a_const = torch.ones(1, device=device, dtype=torch.float32)

    # no quant, create activation ahead
    if cfg["no_a_quant"]:
        if cfg["a"] == "tensor":
            a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a, scale_a_const)
        else:  # token
            a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a)

        def run_quant():
            return vllm_scaled_mm(a_int8, b_int8, scale_a_int8, scale_b_int8, dtype)

        return run_quant

    # dynamic quant, create activation inside
    if cfg["a"] == "tensor":

        def run_quant():
            a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a, scale_a_const)
            return vllm_scaled_mm(a_int8, b_int8, scale_a_int8, scale_b_int8, dtype)

    else:  # token

        def run_quant():
            a_int8, scale_a_int8, _ = vllm_scaled_int8_quant(a)
            return vllm_scaled_mm(a_int8, b_int8, scale_a_int8, scale_b_int8, dtype)

    return run_quant


_enabled = [k for k, v in PROVIDER_CFGS.items() if v.get("enabled")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=[k for k in _enabled],
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

    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b), quantiles=quantiles
        )
    else:
        cfg = PROVIDER_CFGS[provider]
        run_quant = build_int8_runner(cfg, a, b, dtype, device)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: run_quant(), quantiles=quantiles
        )

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


def prepare_shapes(args):
    KN_model_names = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
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
        choices=list(WEIGHT_SHAPES.keys()),
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

    for K, N, model in prepare_shapes(args):
        print(f"{model}, N={N} K={K}, BF16 vs INT8 GEMMs TFLOP/s:")
        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path=f"bench_int8_res_n{N}_k{K}",
            N=N,
            K=K,
        )

    print("Benchmark finished!")
