# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark AWQ GEMV split-k selection and generate optimized configs.

This script benchmarks the AWQ GEMV HIP kernel with different split-k
values for the linear layer shapes of a given model, then saves the
optimal (K, N) -> split_k mapping to a JSON config file.

Usage:
    # Tune for a specific model and save config
    python benchmark_awq_gemv.py --model Qwen/Qwen3-4B-AWQ --tune

    # Tune for another model, merging into existing config
    python benchmark_awq_gemv.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --tune

    # Just benchmark (show results without saving)
    python benchmark_awq_gemv.py --model Qwen/Qwen3-4B-AWQ

    # Benchmark specific shapes
    python benchmark_awq_gemv.py --shapes 2560,2560 2560,19456 4096,4096
"""

import argparse
import json
import os
import sys

import torch

from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser


def get_device_config_filename() -> str:
    """Get the config filename for the current device."""
    device_name = current_platform.get_device_name().replace(" ", "_")
    return f"device_name={device_name}.json"


def get_default_save_dir() -> str:
    """Get the default directory for saving AWQ GEMV configs."""
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "vllm",
        "model_executor",
        "layers",
        "quantization",
        "awq_gemv_configs",
    )


def extract_awq_shapes(
    model: str,
    group_size: int = 128,
    tp_size: int = 1,
    trust_remote_code: bool = False,
) -> list[tuple[int, int]]:
    """Extract unique (K, N) shapes for AWQ-quantized linear layers.

    Args:
        model: HuggingFace model name/path
        group_size: AWQ quantization group size
        tp_size: Tensor parallel size
        trust_remote_code: Whether to trust remote code

    Returns:
        List of unique (K, N) tuples, sorted by (N, K)
    """
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    from vllm.transformers_utils.config import get_config

    config = get_config(model=model, trust_remote_code=trust_remote_code)

    # Determine which modules are excluded from quantization.
    # The quant config JSON often has modules_to_not_convert=None;
    # vLLM auto-detects it from safetensors metadata at runtime.
    raw_quant_config = getattr(config, "quantization_config", {})
    if isinstance(raw_quant_config, dict):
        awq_config = AWQConfig.from_config(raw_quant_config)
        awq_config.maybe_update_config(model)
        modules_to_not_convert = awq_config.modules_to_not_convert
    else:
        modules_to_not_convert = []

    # Extract model dimensions
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    vocab_size = config.vocab_size

    # Apply tensor parallelism
    assert num_attention_heads % tp_size == 0, (
        f"num_attention_heads ({num_attention_heads}) must be divisible "
        f"by tp_size ({tp_size})"
    )
    assert num_key_value_heads % tp_size == 0 or num_key_value_heads < tp_size, (
        f"num_key_value_heads ({num_key_value_heads}) must be divisible "
        f"by tp_size ({tp_size}) or less than tp_size"
    )
    assert intermediate_size % tp_size == 0, (
        f"intermediate_size ({intermediate_size}) must be divisible "
        f"by tp_size ({tp_size})"
    )

    tp_heads = num_attention_heads // tp_size
    tp_kv_heads = max(1, num_key_value_heads // tp_size)
    tp_intermediate = intermediate_size // tp_size

    shapes = set()

    # QKV projection: K=hidden_size, N=(heads + 2*kv_heads) * head_dim
    qkv_n = (tp_heads + 2 * tp_kv_heads) * head_dim
    shapes.add((hidden_size, qkv_n))

    # Output projection: K=num_heads*head_dim, N=hidden_size
    o_k = tp_heads * head_dim
    shapes.add((o_k, hidden_size))

    # Gate+Up projection (merged): K=hidden_size, N=2*intermediate_size
    shapes.add((hidden_size, 2 * tp_intermediate))

    # Down projection: K=intermediate_size, N=hidden_size
    shapes.add((tp_intermediate, hidden_size))

    # LM head: only include if it's actually AWQ-quantized.
    # Exclude if: weight-tied to embeddings, or in modules_to_not_convert.
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
    lm_head_excluded = tie_word_embeddings or any(
        "lm_head" in m for m in modules_to_not_convert
    )
    if not lm_head_excluded and vocab_size % 8 == 0:
        shapes.add((hidden_size, vocab_size))

    # Filter to shapes where HIP GEMV is used
    filtered = []
    for K, N in shapes:
        if N % 8 != 0:
            continue
        if K % group_size != 0:
            continue
        if N < 1500:
            # HIP GEMV is skipped for small N (Triton is more efficient)
            continue
        filtered.append((K, N))

    filtered.sort(key=lambda x: (x[1], x[0]))  # Sort by (N, K)
    return filtered


def compute_padding(K: int, group_size: int, split_k: int) -> int:
    """Compute padded K for a given split_k target.

    Returns the padded K value (may equal K if already divisible).
    """
    num_groups = K // group_size
    if num_groups % split_k == 0:
        return K
    padded_groups = ((num_groups + split_k - 1) // split_k) * split_k
    return padded_groups * group_size


# L2 cache size to flush (8 MB should cover RDNA GPUs)
_L2_FLUSH_SIZE = 8 * 1024 * 1024


def _flush_l2_cache(flush_buf: torch.Tensor) -> None:
    """Flush the GPU L2 cache by writing to a large buffer."""
    flush_buf.zero_()


def benchmark_shape(
    K: int,
    N: int,
    group_size: int,
    split_k: int,
    num_warmup: int = 50,
    num_iters: int = 200,
) -> float:
    """Benchmark AWQ GEMV for a specific shape and split_k.

    Clears the L2 cache between each iteration to get realistic
    latencies that don't benefit from cached data.

    Returns:
        Average kernel time in microseconds.
    """
    from vllm._custom_ops import awq_gemv_hip

    padded_K = compute_padding(K, group_size, split_k)
    num_groups = padded_K // group_size

    # Verify divisibility
    assert num_groups % split_k == 0, (
        f"num_groups ({num_groups}) not divisible by split_k ({split_k}) "
        f"for K={K}, padded_K={padded_K}"
    )

    # Create synthetic AWQ tensors
    act = torch.randn(padded_K, dtype=torch.float16, device="cuda")
    qweight = torch.randint(
        0, 2**31 - 1, (padded_K, N // 8), dtype=torch.int32, device="cuda"
    )
    scales = torch.randn(num_groups, N, dtype=torch.float16, device="cuda")
    qzeros = torch.randint(
        0, 2**31 - 1, (num_groups, N // 8), dtype=torch.int32, device="cuda"
    )

    # L2 cache flush buffer
    flush_buf = torch.empty(_L2_FLUSH_SIZE // 4, dtype=torch.float32, device="cuda")

    # Warmup (with cache flushing too, to stabilize clocks)
    for _ in range(num_warmup):
        _flush_l2_cache(flush_buf)
        awq_gemv_hip(act, qweight, scales, qzeros, split_k)
    torch.cuda.synchronize()

    # Benchmark with CUDA events and L2 cache clearing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(num_iters):
        _flush_l2_cache(flush_buf)
        torch.cuda.synchronize()

        start_event.record()
        awq_gemv_hip(act, qweight, scales, qzeros, split_k)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event) * 1000)  # us

    # Remove outliers: drop top/bottom 5%
    latencies.sort()
    trim = max(1, len(latencies) // 20)
    trimmed = latencies[trim:-trim]

    return sum(trimmed) / len(trimmed)


def check_against_existing(
    results: dict[tuple[int, int], tuple[int, float]],
    existing_config: dict[str, int],
    all_timings: dict[tuple[int, int], dict[int, float]],
) -> list[str]:
    """Check new results against existing config entries.

    Args:
        results: dict mapping (K, N) -> (best_split_k, best_time_us)
        existing_config: existing JSON config mapping "K,N" -> split_k
        all_timings: dict mapping (K, N) -> {split_k: time_us}

    Returns:
        List of warning messages for contradictions.
    """
    warnings = []

    for (K, N), (best_sk, best_time) in sorted(results.items()):
        key = f"{K},{N}"
        existing_sk = existing_config.get(key)
        if existing_sk is not None and existing_sk != best_sk:
            existing_time = all_timings.get((K, N), {}).get(existing_sk)
            time_str = ""
            if existing_time is not None:
                delta_pct = (
                    (existing_time - best_time) / best_time * 100
                    if best_time > 0
                    else 0
                )
                time_str = (
                    f"\n    Existing split_k={existing_sk} @ "
                    f"{existing_time:.1f}us vs best split_k={best_sk} @ "
                    f"{best_time:.1f}us (delta={delta_pct:+.1f}%)"
                )
            warnings.append(
                f"  K={K}, N={N}: benchmark found split_k={best_sk} but "
                f"existing config says split_k={existing_sk}{time_str}"
            )

    return warnings


def load_existing_config(config_path: str) -> dict[str, int] | None:
    """Load existing config file if it exists."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return None


def main(args: argparse.Namespace):
    print("AWQ GEMV Split-K Benchmark")
    print(f"Device: {current_platform.get_device_name()}")
    print()

    # Determine shapes to benchmark
    if args.shapes:
        shapes = []
        for s in args.shapes:
            k, n = s.split(",")
            shapes.append((int(k), int(n)))
        shapes.sort(key=lambda x: (x[1], x[0]))
    else:
        if not args.model:
            print("Error: either --model or --shapes must be specified")
            sys.exit(1)
        print(f"Extracting shapes from model: {args.model}")
        shapes = extract_awq_shapes(
            args.model,
            group_size=args.group_size,
            tp_size=args.tp_size,
            trust_remote_code=args.trust_remote_code,
        )

    if not shapes:
        print("No shapes to benchmark (all filtered out)")
        sys.exit(0)

    print(f"Shapes to benchmark ({len(shapes)}):")
    for K, N in shapes:
        num_groups = K // args.group_size
        print(f"  K={K}, N={N} (num_groups={num_groups})")
    print()

    split_k_values = list(range(1, 21))  # 1-20, all supported by HIP kernel

    # Benchmark all shapes x split_k values
    # Keyed on (K, N) tuples
    all_timings: dict[tuple[int, int], dict[int, float]] = {}
    results: dict[tuple[int, int], tuple[int, float]] = {}

    for K, N in shapes:
        num_groups = K // args.group_size
        print(f"Benchmarking K={K}, N={N} (num_groups={num_groups}):")

        shape_timings: dict[int, float] = {}
        best_sk = 1
        best_time = float("inf")

        for sk in split_k_values:
            # Check if this split_k is feasible (with padding)
            padded_K = compute_padding(K, args.group_size, sk)
            overhead = (padded_K - K) / K if K > 0 else 0

            if overhead > 0.15:
                print(
                    f"  split_k={sk:2d}: SKIPPED "
                    f"(padding overhead {overhead:.0%} > 15%)"
                )
                continue

            try:
                time_us = benchmark_shape(
                    K,
                    N,
                    args.group_size,
                    sk,
                    num_warmup=args.num_warmup,
                    num_iters=args.num_iters,
                )
                shape_timings[sk] = time_us

                pad_str = f" (padded K={padded_K})" if padded_K != K else ""
                marker = ""
                if time_us < best_time:
                    best_time = time_us
                    best_sk = sk
                    marker = " <-- best"

                print(f"  split_k={sk:2d}: {time_us:8.1f} us{pad_str}{marker}")
            except Exception as e:
                print(f"  split_k={sk:2d}: ERROR - {e}")

        all_timings[(K, N)] = shape_timings
        results[(K, N)] = (best_sk, best_time)
        print(f"  ==> Best: split_k={best_sk} @ {best_time:.1f} us")
        print()

    # Print summary table
    print("=" * 60)
    print("Summary:")
    print(f"{'K':>8} {'N':>8} {'Best SK':>8} {'Time (us)':>10}")
    print("-" * 60)
    for K, N in shapes:
        sk, time = results[(K, N)]
        print(f"{K:>8} {N:>8} {sk:>8} {time:>10.1f}")
    print("=" * 60)
    print()

    # Check against existing config
    config_path = os.path.join(
        args.save_dir or get_default_save_dir(),
        get_device_config_filename(),
    )
    existing_config = load_existing_config(config_path)

    has_warnings = False
    if existing_config is not None:
        existing_warnings = check_against_existing(
            results, existing_config, all_timings
        )
        if existing_warnings:
            has_warnings = True
            print("WARNING: Results contradict existing config:")
            for w in existing_warnings:
                print(w)
            print()

    if not args.tune:
        print("Run with --tune to save results to config file.")
        return

    # Build new config: merge with existing
    new_config: dict[str, int] = {}
    if existing_config is not None:
        new_config.update(existing_config)

    # Add/overwrite with new results
    for (K, N), (best_sk, _best_time) in results.items():
        new_config[f"{K},{N}"] = best_sk

    print("Derived config:")
    print(json.dumps(new_config, indent=4))
    print()

    if has_warnings and not args.force:
        response = input(
            "Contradictions detected (see warnings above). Save anyway? [y/N] "
        )
        if response.lower() != "y":
            print("Aborted.")
            return

    # Save config
    save_dir = args.save_dir or get_default_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, get_device_config_filename())

    print(f"Saving config to {config_path}")
    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=4)
        f.write("\n")

    print("Done!")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark AWQ GEMV split-k selection")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name to extract shapes from",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="+",
        default=None,
        help="Explicit K,N shapes to benchmark (e.g., 2560,2560 2560,19456)",
    )
    parser.add_argument(
        "--tp-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size for shape computation (default: 1)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="AWQ quantization group size (default: 128)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=200,
        help="Number of benchmark iterations (default: 200)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Save results to config file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Save even when contradictions are detected",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save config (default: built-in awq_gemv_configs/)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model config",
    )
    args = parser.parse_args()

    main(args)
