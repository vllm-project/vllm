# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: Unfused (C++/Triton) vs Fused RoPE + FP8 per-tensor KV cache write
Run from vllm root:
    python benchmarks/kernels/benchmark_fused_rope_fp8_kvcache.py
"""

import time

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

# shapes: (label, num_tokens)
SHAPES = [
    # decode (9 cases)
    ("decode",   1),
    ("decode",   2),
    ("decode",   4),
    ("decode",   16),
    ("decode",   32),
    ("decode",   64),
    ("decode",   128),
    ("decode",   256),
    ("decode",   512),
    # prefill (7 cases)
    ("prefill",  128),
    ("prefill",  256),
    ("prefill",  512),
    ("prefill",  1024),
    ("prefill",  2048),
    ("prefill",  4096),
    ("prefill",  8192),
]



@torch.inference_mode()
def run_benchmark(
    group: str,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    num_iters: int,
    implementation: str,
    benchmark_mode: str,
    device: str = "cuda",
) -> float:
    """Return latency (seconds) for given configuration."""
    torch.set_default_device(device)

    # Setup inputs
    query = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=dtype, device=device
    )
    cos_sin_cache = torch.randn(
        32768, head_size, dtype=torch.float32, device=device
    )
    positions = torch.randint(
        0, 4096, (num_tokens,), dtype=torch.long, device=device
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 4

    # Slot mapping setup based on decode vs prefill
    if "decode" in group.lower():
        total_slots = num_blocks * block_size
        slot_mapping = torch.randperm(
            total_slots, device=device
        )[:num_tokens].to(torch.long)
    else:
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    # FP8 caches
    key_cache   = torch.zeros(num_blocks, block_size, num_kv_heads, head_size,
                              dtype=torch.uint8, device=device)
    value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size,
                              dtype=torch.uint8, device=device)

    k_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    is_neox = True

    # Define implementations matching production attention backend updates
    if implementation == "fused":
        def function_under_test(
            q=query, k=key, v=value, kc=key_cache, vc=value_cache,
            sm=slot_mapping, pos=positions, csc=cos_sin_cache,
            ks=k_scale, vs=v_scale
        ):
            # 1. Rotate key and write key/value to FP8 cache in one fused kernel
            ops.fused_rope_fp8_kvcache(
                k, v,
                kc, vc,
                sm, pos,
                csc,
                ks, vs,
                is_neox,
                True,  # flash_layout
            )
    elif implementation == "unfused-cpp":
        def function_under_test(
            q=query, k=key, v=value, kc=key_cache, vc=value_cache,
            sm=slot_mapping, pos=positions, csc=cos_sin_cache,
            ks=k_scale, vs=v_scale
        ):
            # 1. Rotate key only
            ops.rotary_embedding(
                pos,
                k.view(num_tokens, -1),
                None,
                head_size,
                csc,
                is_neox,
            )
            # 2. Write key/value to FP8 cache using C++ custom op
            ops.reshape_and_cache_flash(
                k, v,
                kc, vc,
                sm, "fp8",
                ks, vs,
            )
    elif implementation == "unfused-triton":
        def function_under_test(
            q=query, k=key, v=value, kc=key_cache, vc=value_cache,
            sm=slot_mapping, pos=positions, csc=cos_sin_cache,
            ks=k_scale, vs=v_scale
        ):
            # 1. Rotate key only
            ops.rotary_embedding(
                pos,
                k.view(num_tokens, -1),
                None,
                head_size,
                csc,
                is_neox,
            )
            # 2. Write key/value to FP8 cache using Triton kernel
            triton_reshape_and_cache_flash(
                k, v,
                kc, vc,
                sm, "fp8",
                ks, vs,
            )
    else:
        raise ValueError(f"Unknown implementation: {implementation}")


    # Measure pure GPU time using CUDA events (do_bench)
    # If cudagraph mode is selected, we wrap the function in a CUDA Graph first.
    if benchmark_mode == "cudagraph":
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            function_under_test()
        torch.cuda.synchronize()
        target_fn = lambda: g.replay()
    else:
        target_fn = function_under_test

    # 1. Measure pure GPU time using Triton do_bench (CUDA events)
    from vllm.triton_utils import triton
    lat_do_bench = triton.testing.do_bench(target_fn) / 1000.0

    # 2. Measure wall-clock time with standard python timers and explicit synchronization
    # Warmup
    for _ in range(10):
        target_fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        target_fn()
    torch.cuda.synchronize()
    lat_wall = (time.perf_counter() - t0) / num_iters

    # Free memory
    del query, key, value, key_cache, value_cache, slot_mapping
    torch.cuda.empty_cache()

    return lat_do_bench, lat_wall


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark Fused RoPE + FP8 KV Cache Operator"
    )
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cudagraph", "no_graph"],
        default="cudagraph",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("\n" + "=" * 105)
    print(
        f"  RoPE + FP8 KV Cache Update Benchmark  |  "
        f"Mode: {args.mode.upper()}  |  dtype: {args.dtype}"
    )
    print("=" * 105)

    headers = [
        "Group", "Tokens", "Unfused Triton (GPU/Wall us)", "Unfused C++ (GPU/Wall us)",
        "Fused C++ (GPU/Wall us)", "GPU Speedup vs C++", "Wall Speedup vs C++"
    ]
    rows = []

    for group, num_tokens in SHAPES:
        # 1. Unfused Triton
        t_triton_gpu, t_triton_wall = run_benchmark(
            group, num_tokens, args.num_heads, args.num_kv_heads,
            args.head_size, args.block_size, dtype, args.iters,
            "unfused-triton", args.mode
        )
        t_triton_gpu *= 1e6
        t_triton_wall *= 1e6

        # 2. Unfused C++
        t_cpp_gpu, t_cpp_wall = run_benchmark(
            group, num_tokens, args.num_heads, args.num_kv_heads,
            args.head_size, args.block_size, dtype, args.iters,
            "unfused-cpp", args.mode
        )
        t_cpp_gpu *= 1e6
        t_cpp_wall *= 1e6

        # 3. Fused C++
        t_fused_gpu, t_fused_wall = run_benchmark(
            group, num_tokens, args.num_heads, args.num_kv_heads,
            args.head_size, args.block_size, dtype, args.iters,
            "fused", args.mode
        )
        t_fused_gpu *= 1e6
        t_fused_wall *= 1e6

        speedup_gpu = t_cpp_gpu / t_fused_gpu
        speedup_wall = t_cpp_wall / t_fused_wall

        rows.append([
            group,
            num_tokens,
            f"{t_triton_gpu:.2f} / {t_triton_wall:.2f}",
            f"{t_cpp_gpu:.2f} / {t_cpp_wall:.2f}",
            f"{t_fused_gpu:.2f} / {t_fused_wall:.2f}",
            f"{speedup_gpu:.2f}x",
            f"{speedup_wall:.2f}x"
        ])

        # Print current row immediately to show progress
        print(
            f"Running {group} {num_tokens} tokens... "
            f"Fused: {t_fused_gpu:.2f}/{t_fused_wall:.2f} us | "
            f"C++: {t_cpp_gpu:.2f}/{t_cpp_wall:.2f} us | "
            f"Triton: {t_triton_gpu:.2f}/{t_triton_wall:.2f} us"
        )

    print("\n" + "=" * 105)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
