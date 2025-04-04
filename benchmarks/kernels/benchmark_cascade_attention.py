# SPDX-License-Identifier: Apache-2.0

import time

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser
from vllm.v1.attention.backends.flash_attn import merge_attn_states_triton

NUM_HEADS = [(16, 2)]
HEAD_SIZES = [128, 192, 256]


@torch.inference_mode()
def main(
    version: str,
    num_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    prefix_output = torch.randn(num_tokens,
                                num_query_heads,
                                head_size,
                                dtype=dtype)
    suffix_output = torch.randn(num_tokens,
                                num_query_heads,
                                head_size,
                                dtype=dtype)
    prefix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)
    suffix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)

    output = torch.empty(num_tokens, num_query_heads, head_size, dtype=dtype)

    def run_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if version == "cuda":
                ops.merge_attn_states(
                    output,
                    prefix_output,
                    prefix_lse,
                    suffix_output,
                    suffix_lse,
                )
            elif version == "triton":
                merge_attn_states_triton(
                    output,
                    prefix_output,
                    prefix_lse,
                    suffix_output,
                    suffix_lse,
                )
            else:
                raise ValueError(f"Invalid version: {version}")
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["cuda", "triton"],
                        default="cuda")
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[32, 64, 128, 128, 192, 256],
                        default=128)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    main(
        version=args.version,
        num_tokens=args.num_tokens,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
    )
