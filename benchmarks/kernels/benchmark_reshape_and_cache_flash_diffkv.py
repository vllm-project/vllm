# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)

logger = init_logger(__name__)


def parse_head_sizes(value: str) -> tuple[int, int]:
    try:
        head_size_k, head_size_v = value.split(":", maxsplit=1)
        return int(head_size_k), int(head_size_v)
    except ValueError as err:
        raise ValueError(
            f"Invalid head size pair {value!r}; expected HEAD_SIZE_K:HEAD_SIZE_V."
        ) from err


@torch.inference_mode()
def run_benchmark(
    num_tokens: int,
    num_heads: int,
    head_size_k: int,
    head_size_v: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    num_iters: int,
    warmup_iters: int,
    implementation: str,
    benchmark_mode: str,
    device: str = "cuda",
) -> float:
    """Return latency in seconds for one DiffKV reshape-and-cache shape."""

    if kv_cache_dtype == "fp8" and not current_platform.has_device_capability(89):
        return float("nan")
    if implementation not in ("cuda", "triton"):
        raise ValueError(
            f"Unsupported implementation: {implementation}. "
            "Only 'cuda' and 'triton' are supported."
        )

    set_random_seed(42)
    torch.set_default_device(device)

    key = torch.randn(num_tokens, num_heads, head_size_k, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_heads, head_size_v, dtype=dtype, device=device)

    num_slots = block_size * num_blocks
    if num_tokens > num_slots:
        raise ValueError("num_tokens cannot exceed the total number of cache slots")
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.empty(
        (num_blocks, block_size, num_heads, head_size_k + head_size_v),
        dtype=cache_dtype,
        device=device,
    )

    k_scale = (key.abs().amax() / 64.0).to(torch.float32)
    v_scale = (value.abs().amax() / 64.0).to(torch.float32)

    if implementation == "cuda":
        function_under_test = lambda: ops.reshape_and_cache_flash_diffkv(
            key,  # noqa: F821
            value,  # noqa: F821
            kv_cache,  # noqa: F821
            slot_mapping,  # noqa: F821
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        function_under_test = lambda: triton_reshape_and_cache_flash_diffkv(
            key,  # noqa: F821
            value,  # noqa: F821
            kv_cache,  # noqa: F821
            slot_mapping,  # noqa: F821
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    def run_cuda_benchmark(n_iters: int) -> float:
        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            function_under_test()
            torch.accelerator.synchronize()
        end = time.perf_counter()
        return (end - start) / n_iters

    # Trigger CUDA extension dispatch and Triton compilation outside timing.
    run_cuda_benchmark(1)

    if benchmark_mode == "cudagraph":
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            function_under_test()
        torch.accelerator.synchronize()
        function_under_test = lambda: g.replay()

    run_cuda_benchmark(warmup_iters)
    lat = run_cuda_benchmark(num_iters)

    del key, value, kv_cache, slot_mapping
    torch.accelerator.empty_cache()

    return lat


def main(args):
    rows = []
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    for kv_cache_dtype in args.kv_cache_dtype:
        for head_size_k, head_size_v in args.head_sizes:
            for num_heads in args.num_heads:
                for num_tokens in args.num_tokens:
                    cuda_lat = run_benchmark(
                        num_tokens=num_tokens,
                        num_heads=num_heads,
                        head_size_k=head_size_k,
                        head_size_v=head_size_v,
                        block_size=args.block_size,
                        num_blocks=args.num_blocks,
                        dtype=dtype,
                        kv_cache_dtype=kv_cache_dtype,
                        num_iters=args.iters,
                        warmup_iters=args.warmup_iters,
                        implementation="cuda",
                        benchmark_mode=args.mode,
                        device="cuda",
                    )
                    triton_lat = run_benchmark(
                        num_tokens=num_tokens,
                        num_heads=num_heads,
                        head_size_k=head_size_k,
                        head_size_v=head_size_v,
                        block_size=args.block_size,
                        num_blocks=args.num_blocks,
                        dtype=dtype,
                        kv_cache_dtype=kv_cache_dtype,
                        num_iters=args.iters,
                        warmup_iters=args.warmup_iters,
                        implementation="triton",
                        benchmark_mode=args.mode,
                        device="cuda",
                    )
                    speedup = triton_lat / cuda_lat
                    rows.append(
                        [
                            num_tokens,
                            num_heads,
                            f"{head_size_k}:{head_size_v}",
                            args.dtype,
                            kv_cache_dtype,
                            f"{cuda_lat * 1e6:.3f}",
                            f"{triton_lat * 1e6:.3f}",
                            f"{speedup:.3f}x",
                        ]
                    )

    print(f"Benchmark results for DiffKV reshape-and-cache ({args.mode}):")
    print(
        tabulate(
            rows,
            headers=[
                "num_tokens",
                "num_heads",
                "head_sizes",
                "dtype",
                "kv_cache_dtype",
                "cuda (us)",
                "triton (us)",
                "speedup",
            ],
        )
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 128, 512],
    )
    parser.add_argument("--num-heads", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--head-sizes",
        type=parse_head_sizes,
        nargs="+",
        default=[(64, 128), (80, 96), (128, 128)],
        help="K/V head-size pairs formatted as HEAD_SIZE_K:HEAD_SIZE_V.",
    )
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        nargs="+",
        choices=["auto", "fp8"],
        default=["auto", "fp8"],
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cudagraph", "no_graph"],
        default="cudagraph",
    )

    args = parser.parse_args()

    main(args)
