# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare SM100 block FP8 Linear with and without pre-quantization padding.

Includes input quantization and padding in GPU timing. CUDA graph replay excludes
Python launch overhead, so these measurements do not replace an eager serving run.
"""

import argparse
import os
import statistics
import subprocess

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=34816)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[7, 127, 128, 129, 130, 131, 256, 257, 8192, 8193, 8194, 8195],
    )
    args = parser.parse_args()
    if not current_platform.is_device_capability_family(100):
        parser.error("This benchmark requires an SM100-family GPU (e.g. B300).")
    if args.n <= 0 or args.n % 16 or args.k <= 0 or args.k % 128:
        parser.error("N must be a positive multiple of 16; K a multiple of 128.")
    if any(m <= 0 for m in args.m):
        parser.error("M must be positive.")

    print(
        f"GPU={current_platform.get_device_name()}, torch={torch.__version__}, "
        f"CUDA={torch.version.cuda}, dtype=BF16, N={args.n}, K={args.k}"
    )
    print(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    env_names = (
        "CUDA_VISIBLE_DEVICES",
        "VLLM_BATCH_INVARIANT",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_USE_DEEP_GEMM_E8M0",
    )
    print({name: os.environ.get(name) for name in env_names})
    print("CUPTI, CUDA graphs, cold L2; TFLOPS uses 2*M*N*K (unpadded work)")
    print("M,baseline_us,padded_us,speedup,baseline_tflops,padded_tflops")
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = torch.randn(args.n, args.k, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((args.n + 127) // 128, args.k // 128, device="cuda") * 0.1
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(args.n, args.k),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    with set_current_vllm_config(VllmConfig(compilation_config={"mode": 0})):
        kernel = CutlassFp8BlockScaledMMKernel(config)

    def run(x, w, s, padded):
        layer.weight, layer.weight_scale = w, s
        if padded:
            return kernel.apply_weights(layer, x)
        return Fp8BlockScaledMMLinearKernel.apply_weights(kernel, layer, x)

    for index, m in enumerate(args.m):
        x = torch.randn(m, args.k, device="cuda", dtype=torch.bfloat16) * 0.1
        expected = run(x, weight, scales, False)
        actual = run(x, weight, scales, True)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        timings = {}
        for padded in [False, True] if index % 2 == 0 else [True, False]:
            for _ in range(10):
                run(x, weight, scales, padded)
            torch.accelerator.synchronize()
            timings[padded] = (
                statistics.median(
                    bench_gpu_time_with_cupti(
                        run,
                        input_args=(x, weight, scales, padded),
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                    )
                )
                * 1000
            )
        base, pad = timings[False], timings[True]
        flops = 2 * m * args.n * args.k
        print(
            f"{m},{base:.3f},{pad:.3f},{base / pad:.3f},"
            f"{flops / (base * 1e6):.3f},{flops / (pad * 1e6):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
