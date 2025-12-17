# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at).
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import copy
import itertools

import torch
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from weight_shapes import WEIGHT_SHAPES

from vllm._custom_ops import fusedQuantizeMx, matmul_mxf4_bf16_tn
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
from vllm.triton_utils import triton

PROVIDER_CFGS = {
    "torch-bf16": dict(enabled=True),
    "mxfp4": dict(no_a_quant=False, enabled=True),
    "mxfp4-noquant": dict(no_a_quant=True, enabled=True),
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return (
        deterministic_hadamard_matrix(group_size, dtype=dtype, device=device)
        * group_size**-0.5
    )


def _quant_weight_mxfp4(
    b: torch.Tensor, forward_hadamard_matrix: torch.Tensor, device: str
):
    weight_hf_e2m1, weight_hf_e8m0 = fusedQuantizeMx(
        b, forward_hadamard_matrix, method="abs_max"
    )
    weight_hf_scale_block = to_blocked(weight_hf_e8m0, backend="triton")
    return weight_hf_e2m1, weight_hf_scale_block


def build_mxfp4_runner(cfg, a, b, forward_hadamard_matrix, dtype, device):
    weight_hf_e2m1, weight_hf_scale_block = _quant_weight_mxfp4(
        b, forward_hadamard_matrix, device
    )
    alpha = torch.tensor([1.0], device="cuda")

    if cfg["no_a_quant"]:
        # Pre-quantize activation
        input_hf_e2m1, input_hf_e8m0 = fusedQuantizeMx(
            a, forward_hadamard_matrix, method="abs_max"
        )
        input_hf_scale_block = to_blocked(input_hf_e8m0, backend="triton")

        def run():
            return matmul_mxf4_bf16_tn(
                input_hf_e2m1,
                weight_hf_e2m1,
                input_hf_scale_block,
                weight_hf_scale_block,
                alpha,
            )

        return run

    # Quantize activation on-the-fly
    def run():
        input_hf_e2m1, input_hf_e8m0 = fusedQuantizeMx(
            a, forward_hadamard_matrix, method="abs_max"
        )
        input_hf_scale_block = to_blocked(input_hf_e8m0, backend="triton")
        return matmul_mxf4_bf16_tn(
            input_hf_e2m1,
            weight_hf_e2m1,
            input_hf_scale_block,
            weight_hf_scale_block,
            alpha,
        )

    return run


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[
            1,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            24576,
            32768,
        ],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="TFLOP/s (larger is better)",
        plot_name="BF16 vs MXFP4 GEMMs",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, had_size):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((N, K), device=device, dtype=dtype)
    forward_hadamard_matrix = get_hadamard_matrix(had_size, dtype, device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b), rep=200, quantiles=quantiles
        )
    else:
        cfg = PROVIDER_CFGS[provider]
        run_quant = build_mxfp4_runner(
            cfg, a, b, forward_hadamard_matrix, dtype, device
        )
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: run_quant(), rep=200, quantiles=quantiles
        )

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


def prepare_shapes(args):
    out = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            out.append(KN)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.3-70B-Instruct"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    for K, N, model in prepare_shapes(args):
        for had_size in [32, 64, 128]:
            print(f"{model}, N={N} K={K}, HAD={had_size}, BF16 vs MXFP4 GEMMs TFLOP/s:")
            benchmark.run(
                print_data=True,
                show_plots=True,
                save_path=f"bench_mxfp4_res_n{N}_k{K}",
                N=N,
                K=K,
                had_size=had_size,
            )

    print("Benchmark finished!")
