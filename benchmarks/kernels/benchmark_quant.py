# SPDX-License-Identifier: Apache-2.0

import time

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser


@torch.inference_mode()
def main(num_tokens: int,
         hidden_size: int,
         static_scale: bool,
         quant_dtype: torch.dtype,
         dtype: torch.dtype,
         seed: int = 0,
         do_profile: bool = False,
         num_warmup_iters: int = 5,
         num_iters: int = 100) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device("cuda")

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    scale = torch.randn(1, 1, dtype=torch.float32) if static_scale else None

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if quant_dtype == torch.int8:
                ops.scaled_int8_quant(x, scale)
            else:
                ops.scaled_fp8_quant(x, scale)
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=num_warmup_iters, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=num_iters, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError(f"Unsupported dtype: {dt}")

    parser = FlexibleArgumentParser(
        description="Benchmark the quantization (fp8 or int8) kernel.")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--static-scale", action="store_true")
    parser.add_argument("--quant-dtype",
                        type=str,
                        choices=["fp8", "int8"],
                        default="int8")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters",
                        type=int,
                        default=100,
                        help="Number of benchmark iterations. "
                        "If --profile is set, this number is ignored")

    args = parser.parse_args()
    print(args)

    main(num_tokens=args.num_tokens,
         hidden_size=args.hidden_size,
         static_scale=args.static_scale,
         quant_dtype=to_torch_dtype(args.quant_dtype),
         dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
         seed=args.seed,
         do_profile=args.profile,
         num_warmup_iters=args.num_warmup_iters,
         num_iters=args.num_iters)
