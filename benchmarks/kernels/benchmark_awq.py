import argparse
import random
import time

import torch

from vllm import quantization_ops
from vllm.model_executor.layers.quantized_ops.awq import awq_matmul, unpack_int32

MAX_INT32 = 0x7fffffff
MIN_INT32 = -MAX_INT32 - 1


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_to_torch_dtype[args.dtype]
    pack_factor = 32 // args.bits

    x = torch.randn(args.m, args.k, dtype=dtype, device="cuda")
    w = torch.randint(MIN_INT32,
                      MAX_INT32, (args.k, args.n // pack_factor),
                      dtype=torch.int32,
                      device="cuda")
    qzeros = torch.randint(MIN_INT32,
                           MAX_INT32,
                           (args.k // args.group_size, args.n // pack_factor),
                           dtype=torch.int32,
                           device="cuda")
    scales = torch.randn(args.k // args.group_size,
                         args.n,
                         dtype=dtype,
                         device="cuda")

    if args.version == "triton":
        unpacked_qzeros = unpack_int32(qzeros, pack_factor)
        shifter = torch.tensor(
            [0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32, device="cuda") * 4

    def run(num_iters: int) -> None:
        for _ in range(num_iters):
            if args.version == "orig":
                quantization_ops.awq_gemm(x, w, scales, qzeros, pack_factor)
            elif args.version == "triton":
                awq_matmul(x,
                           w,
                           unpacked_qzeros,
                           scales,
                           pack_factor,
                           args.group_size,
                           shifter,
                           is_qzero_packed=False)
            else:
                raise ValueError(f"Invalid version: {args.version}")
        torch.cuda.synchronize()

    def run_benchmark(num_iters: int,
                      profile: bool = False,
                      gpu_time: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        if gpu_time:
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
                run(num_iters)
            return p.key_averages().total_average(
            ).self_cuda_time_total / num_iters
        else:
            run(num_iters)

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark(num_iters=3, profile=False) * 1000000
    # Benchmark.
    if args.gpu_time:
        assert not args.profile
        latency = run_benchmark(num_iters=100, gpu_time=True)
    elif args.profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Elapsed time: {latency:.3f} us")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["orig", "triton"],
                        default="triton")
    parser.add_argument("--m", "-m", type=int, default=8)
    parser.add_argument("--n", "-n", type=int, default=20480)
    parser.add_argument("--k", "-k", type=int, default=5120)
    parser.add_argument("--bits", type=int, choices=[4], default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gpu-time", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)
