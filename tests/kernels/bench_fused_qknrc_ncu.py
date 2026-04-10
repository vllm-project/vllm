# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NCU profiling harness for fused_qk_norm_rope_cache_quant (v1 warp-per-head).

10 token-count levels (dense at low end, sparse at high end):
  1, 4, 16, 64, 128, 512, 1024, 2048, 5000, 10000

Usage — profile a single token count:
    ncu --set full --target-processes all \
        -o profile_T128 \
        python tests/kernels/bench_fused_qknrc_ncu.py --tokens 128

Profile all 10 levels (one invocation each):
    ncu --set full --target-processes all \
        --kernel-name "fused_kernel" \
        -o profile_all \
        python tests/kernels/bench_fused_qknrc_ncu.py

Specify a custom list:
    python tests/kernels/bench_fused_qknrc_ncu.py --tokens 1 32 256 4096

Options:
    --fp8           Use FP8 E4M3 KV cache (default: BF16)
    --gptj          Use GPT-J RoPE layout (default: NeoX)
    --heads-q N     Number of Q heads (default: 32)
    --heads-kv N    Number of KV heads (default: 8)
    --head-dim N    Head dimension (default: 128)
    --repeat N      Kernel invocations per token count (default: 1)
"""

import argparse
import sys

import torch

import vllm._custom_ops as ops  # noqa: F401


DEFAULT_LEVELS = [1, 4, 16, 64, 128, 512, 1024, 2048, 5000, 10000]


def make_cos_sin_cache(max_pos: int, head_dim: int, dtype: torch.dtype):
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype).cuda()


def run_kernel(
    num_tokens: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_size: int,
    is_neox: bool,
    is_fp8: bool,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    repeat: int,
):
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    max_pos = cos_sin_cache.size(0)
    dtype = q_weight.dtype

    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.long, device="cuda"
    )
    slot_mapping = torch.arange(
        num_tokens, dtype=torch.long, device="cuda"
    )

    query = torch.randn(
        num_tokens, num_heads_q, head_dim, dtype=dtype, device="cuda"
    )
    key = torch.randn(
        num_tokens, num_heads_kv, head_dim, dtype=dtype, device="cuda"
    )
    value = torch.randn(
        num_tokens, num_heads_kv, head_dim, dtype=dtype, device="cuda"
    )

    for _ in range(repeat):
        torch.ops._C.fused_qk_norm_rope_cache_quant(
            query, key, value, k_cache, v_cache,
            q_weight, k_weight, cos_sin_cache,
            positions, slot_mapping,
            1.0, 1.0, 1e-6,
            num_heads_q, num_heads_kv, head_dim, block_size,
            is_neox, is_fp8,
        )

    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="NCU profiling harness for fused_qk_norm_rope_cache_quant"
    )
    parser.add_argument(
        "--tokens", type=int, nargs="*", default=None,
        help="Token counts to profile (default: 10 predefined levels)",
    )
    parser.add_argument("--fp8", action="store_true", help="FP8 KV cache")
    parser.add_argument("--gptj", action="store_true", help="GPT-J RoPE")
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Invocations per token count (NCU replays internally, 1 is fine)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    token_counts = sorted(args.tokens) if args.tokens else DEFAULT_LEVELS
    is_neox = not args.gptj
    is_fp8 = args.fp8
    num_heads_q = args.heads_q
    num_heads_kv = args.heads_kv
    head_dim = args.head_dim
    block_size = args.block_size
    dtype = torch.bfloat16

    max_pos = max(token_counts) + 1024
    max_tokens = max(token_counts)
    num_blocks = (max_tokens + block_size - 1) // block_size + 64

    gpu = torch.cuda.get_device_name(0)
    gqa_ratio = num_heads_q // num_heads_kv
    warps = min(2 + gqa_ratio, 5)
    rope_str = "NeoX" if is_neox else "GPT-J"
    cache_str = "FP8" if is_fp8 else "BF16"

    print(f"GPU: {gpu}")
    print(f"Config: Q={num_heads_q} KV={num_heads_kv} hd={head_dim} "
          f"GQA={gqa_ratio} warps={warps} rope={rope_str} cache={cache_str}")
    print(f"Token levels ({len(token_counts)}): {token_counts}")
    print(f"Repeat per level: {args.repeat}")
    print()

    cos_sin_cache = make_cos_sin_cache(max_pos, head_dim, dtype)
    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda").abs() + 0.1
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda").abs() + 0.1

    cache_dtype = torch.uint8 if is_fp8 else dtype
    k_cache = torch.zeros(
        num_blocks, block_size, num_heads_kv, head_dim,
        dtype=cache_dtype, device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)

    for T in token_counts:
        print(f">>> T={T}", flush=True)
        run_kernel(
            T, num_heads_q, num_heads_kv, head_dim, block_size,
            is_neox, is_fp8,
            cos_sin_cache, k_cache, v_cache, q_weight, k_weight,
            args.repeat,
        )
        print(f"    done", flush=True)

    print("\nAll levels complete.")


if __name__ == "__main__":
    main()
