# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for Triton Unified Attention kernel.

Compares performance between:
- Triton Unified Attention (from vllm/attention/ops/triton_unified_attention.py)
- FlashAttention (via flash_attn package)

Usage:
    python benchmark_triton_unified_attention.py
    python benchmark_triton_unified_attention.py --batch-sizes 1 4 8
    python benchmark_triton_unified_attention.py --output-csv results.csv
"""

import csv
import os
import random
from datetime import datetime

import torch

from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

# Try to import flash_attn for comparison
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.warning(
        "flash_attn not available. Only Triton Unified Attention will be benchmarked."
    )


def create_kv_cache_with_random(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random KV cache tensors in the format expected by unified_attention.

    Returns K and V caches with shape: [num_blocks, block_size, num_kv_heads, head_size]
    """
    scale = 1.0 / (head_size**0.5)
    k_cache = torch.empty(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    v_cache = torch.empty(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    k_cache.uniform_(-scale, scale)
    v_cache.uniform_(-scale, scale)
    return k_cache, v_cache


def time_fn(fn, warmup: int = 10, trials: int = 20):
    """Time a function with warmup and multiple trials."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(trials):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return sum(times) / len(times), torch.std(torch.tensor(times))


@torch.inference_mode()
def benchmark_attention(
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    query_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    use_alibi: bool = False,
    sliding_window: int = -1,
    softcap: float = 0.0,
    warmup: int = 10,
    trials: int = 20,
    seed: int = 0,
    device: str = "cuda",
) -> dict:
    """Benchmark Triton Unified Attention vs FlashAttention.

    Args:
        dtype: Data type for tensors
        batch_size: Number of sequences in batch
        seq_len: Total sequence length (context + query)
        query_len: Number of query tokens per sequence
        num_query_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_size: Dimension of each head
        block_size: KV cache block size
        use_alibi: Whether to use ALiBi position encoding
        sliding_window: Sliding window size (-1 for no window)
        softcap: Softcap value for attention logits (0.0 for no cap)
        warmup: Number of warmup iterations
        trials: Number of timed trials
        seed: Random seed
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    scale = float(1.0 / (head_size**0.5))
    context_len = seq_len - query_len

    # Ensure valid configuration
    assert num_query_heads % num_kv_heads == 0, (
        "num_query_heads must be divisible by num_kv_heads"
    )
    assert context_len >= 0, "context_len must be non-negative"

    # Total query tokens across all sequences
    total_query_tokens = batch_size * query_len

    # Create query tensor [total_tokens, num_query_heads, head_size]
    query = torch.empty(
        total_query_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    query.uniform_(-scale, scale)

    # Create output tensor
    output_triton = torch.empty_like(query)

    # Create cumulative sequence lengths for queries
    # For simplicity, assume uniform query_len per sequence
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * query_len, query_len, dtype=torch.int32, device=device
    )

    # Sequence lengths (total length = context + query)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    # Create KV cache
    max_num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * max_num_blocks_per_seq * 2  # Extra blocks for randomness

    k_cache, v_cache = create_kv_cache_with_random(
        num_blocks, block_size, num_kv_heads, head_size, dtype, device
    )

    # Create block tables [batch_size, max_num_blocks_per_seq]
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        for j in range(max_num_blocks_per_seq):
            block_tables[i, j] = random.randint(0, num_blocks - 1)

    # ALiBi slopes
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float32, device=device)

    # Window size for sliding window attention
    window_size = (sliding_window, sliding_window) if sliding_window > 0 else (-1, -1)

    # Benchmark Triton Unified Attention
    def run_triton_unified():
        unified_attention(
            q=query,
            k=k_cache,
            v=v_cache,
            out=output_triton,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            seqused_k=seq_lens,
            max_seqlen_k=seq_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=softcap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=alibi_slopes,
        )

    triton_mean, triton_std = time_fn(run_triton_unified, warmup=warmup, trials=trials)

    # Benchmark FlashAttention if available
    flash_mean = None
    flash_std = None
    speedup = None

    if FLASH_ATTN_AVAILABLE and not use_alibi:
        # For FlashAttention, we need contiguous K, V tensors
        # Create K, V for flash attention [total_kv_tokens, num_kv_heads, head_size]
        total_kv_tokens = batch_size * seq_len
        k_flash = torch.empty(
            total_kv_tokens, num_kv_heads, head_size, dtype=dtype, device=device
        )
        v_flash = torch.empty(
            total_kv_tokens, num_kv_heads, head_size, dtype=dtype, device=device
        )
        k_flash.uniform_(-scale, scale)
        v_flash.uniform_(-scale, scale)

        # Cumulative sequence lengths for K/V
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
        )

        # The `query` tensor is already in the correct format for flash_attn_varlen_func
        def run_flash_attn():
            flash_attn_varlen_func(
                q=query,
                k=k_flash,
                v=v_flash,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=query_len,
                max_seqlen_k=seq_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
            )

        try:
            flash_mean, flash_std = time_fn(
                run_flash_attn, warmup=warmup, trials=trials
            )
            speedup = (flash_mean - triton_mean) / flash_mean
        except Exception as e:
            logger.warning("FlashAttention benchmark failed: %s", e)
            flash_mean = None
            flash_std = None

    # Print results
    result_str = (
        f"  batch={batch_size:4d}  seq_len={seq_len:6d}  query_len={query_len:4d}  "
        f"triton={triton_mean:8.3f}ms (std={triton_std.item():6.3f})"
    )
    if flash_mean is not None:
        result_str += (
            f"  flash={flash_mean:8.3f}ms (std={flash_std.item():6.3f})  "
            f"speedup={speedup:+.1%}"
        )
    print(result_str)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "query_len": query_len,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "block_size": block_size,
        "dtype": str(dtype),
        "use_alibi": use_alibi,
        "sliding_window": sliding_window,
        "softcap": softcap,
        "triton_mean_ms": triton_mean,
        "triton_std_ms": triton_std.item(),
        "flash_mean_ms": flash_mean,
        "flash_std_ms": flash_std.item() if flash_std is not None else None,
        "speedup_vs_flash": speedup,
    }


def write_results_to_csv(results: list[dict], filename: str | None = None):
    """Write benchmark results to CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"triton_unified_attention_benchmark_{timestamp}.csv"

    fieldnames = [
        "batch_size",
        "seq_len",
        "query_len",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "block_size",
        "dtype",
        "use_alibi",
        "sliding_window",
        "softcap",
        "triton_mean_ms",
        "triton_std_ms",
        "flash_mean_ms",
        "flash_std_ms",
        "speedup_vs_flash",
    ]

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"\nResults written to {filename}")


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark Triton Unified Attention vs FlashAttention"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--query-lens",
        type=int,
        nargs="+",
        default=[1, 64, 256],
        help="Query lengths to benchmark (decode=1, prefill>1)",
    )
    parser.add_argument(
        "--num-query-heads",
        type=int,
        default=32,
        help="Number of query heads",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of KV heads",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=128,
        choices=[64, 80, 96, 112, 128, 192, 256],
        help="Head dimension",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        choices=[16, 32, 64],
        help="KV cache block size",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument(
        "--use-alibi",
        action="store_true",
        help="Use ALiBi position encoding",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=-1,
        help="Sliding window size (-1 for no window)",
    )
    parser.add_argument(
        "--softcap",
        type=float,
        default=0.0,
        help="Softcap value for attention logits (0.0 for no cap)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of timed trials",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV filename (auto-generated if not specified)",
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Only benchmark decode (query_len=1)",
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Only benchmark prefill (query_len>1)",
    )

    args = parser.parse_args()

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Filter query lengths based on mode
    query_lens = args.query_lens
    if args.decode_only:
        query_lens = [1]
    elif args.prefill_only:
        query_lens = [ql for ql in query_lens if ql > 1]

    print("=" * 80)
    print("Triton Unified Attention Benchmark")
    print("=" * 80)
    print("Configuration:")
    print(f"  dtype: {dtype}")
    print(f"  num_query_heads: {args.num_query_heads}")
    print(f"  num_kv_heads: {args.num_kv_heads}")
    print(f"  head_size: {args.head_size}")
    print(f"  block_size: {args.block_size}")
    print(f"  use_alibi: {args.use_alibi}")
    print(f"  sliding_window: {args.sliding_window}")
    print(f"  softcap: {args.softcap}")
    print(f"  FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    print("=" * 80)

    all_results = []

    for query_len in query_lens:
        mode = "decode" if query_len == 1 else f"prefill (query_len={query_len})"
        print(f"\n--- Mode: {mode} ---")

        for seq_len in args.seq_lens:
            # Skip invalid configurations
            if query_len > seq_len:
                continue

            for batch_size in args.batch_sizes:
                try:
                    result = benchmark_attention(
                        dtype=dtype,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        query_len=query_len,
                        num_query_heads=args.num_query_heads,
                        num_kv_heads=args.num_kv_heads,
                        head_size=args.head_size,
                        block_size=args.block_size,
                        use_alibi=args.use_alibi,
                        sliding_window=args.sliding_window,
                        softcap=args.softcap,
                        warmup=args.warmup,
                        trials=args.trials,
                        seed=args.seed,
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(
                        "Benchmark failed for batch=%d, seq_len=%d, query_len=%d: %s",
                        batch_size,
                        seq_len,
                        query_len,
                        e,
                    )

    # Write results to CSV
    if all_results:
        write_results_to_csv(all_results, args.output_csv)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
