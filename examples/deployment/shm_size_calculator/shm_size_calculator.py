#!/usr/bin/env python3
"""
Calculate recommended /dev/shm size for vLLM deployment.

This tool estimates the shared memory required for inter-process communication
(IPC) between GPU workers during tensor parallel inference. It accounts for
NCCL communication buffers, PyTorch tensor descriptors, KV cache metadata,
and batch exchange buffers.

Usage:
    python examples/deployment/shm_size_calculator.py \
        --model-params 30 \
        --tensor-parallel-size 4 \
        --max-concurrent-requests 20 \
        --max-seq-length 8192

    # With all optional parameters
    python examples/deployment/shm_size_calculator.py \
        --model-params 70 \
        --tensor-parallel-size 8 \
        --max-concurrent-requests 32 \
        --max-seq-length 16384 \
        --hidden-size 8192 \
        --num-layers 80 \
        --num-attention-heads 64

Or use interactive mode:
    python examples/deployment/shm_size_calculator.py
"""

import argparse
import json
import math
import sys


# Architecture lookup tables for popular models
MODEL_PROFILES = {
    "mistral-7b": {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32},
    "mistral-8x7b": {"hidden_size": 4096, "num_layers": 58, "num_attention_heads": 32},
    "mistral-8x22b": {"hidden_size": 6144, "num_layers": 72, "num_attention_heads": 48},
    "mistral-large": {"hidden_size": 12288, "num_layers": 72, "num_attention_heads": 96},
    "mistral-medium": {"hidden_size": 5120, "num_layers": 40, "num_attention_heads": 32},
    "llama-3.1-8b": {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32},
    "llama-3.1-70b": {"hidden_size": 8192, "num_layers": 80, "num_attention_heads": 64},
    "llama-3.1-405b": {"hidden_size": 16384, "num_layers": 126, "num_attention_heads": 128},
    "deepseek-v3": {"hidden_size": 7168, "num_layers": 61, "num_attention_heads": 128},
    "qwen-2.5-72b": {"hidden_size": 8192, "num_layers": 80, "num_attention_heads": 64},
    "gemma-2-27b": {"hidden_size": 4608, "num_layers": 46, "num_attention_heads": 32},
    "phi-4": {"hidden_size": 6144, "num_layers": 44, "num_attention_heads": 48},
}


DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}


def calculate_shm(
    model_size_params,
    tensor_parallel_size,
    max_concurrent_requests,
    max_seq_length,
    dtype="bf16",
    hidden_size=None,
    num_layers=None,
    num_attention_heads=None,
):
    """
    Calculate recommended /dev/shm size.

    Args:
        model_size_params: Model size in billions of parameters
        tensor_parallel_size: Number of GPUs (tensor parallelism degree)
        max_concurrent_requests: Max concurrent requests
        max_seq_length: Maximum sequence length
        dtype: Weight precision (bf16, fp16, fp8, int8, int4)
        hidden_size: Model hidden size (auto-estimated if None)
        num_layers: Number of transformer layers (auto-estimated if None)
        num_attention_heads: Number of attention heads (auto-estimated if None)

    Returns:
        dict with breakdown and recommended size in GiB
    """
    # Validate inputs
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if max_concurrent_requests < 1:
        raise ValueError("max_concurrent_requests must be >= 1")
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be >= 1")
    if model_size_params <= 0:
        raise ValueError("model_size_params must be > 0")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported: {', '.join(DTYPE_BYTES.keys())}")

    # Auto-estimate missing architecture params
    if hidden_size is None:
        if model_size_params <= 7:
            hidden_size = 4096
        elif model_size_params <= 30:
            hidden_size = 5120
        elif model_size_params <= 70:
            hidden_size = 8192
        else:
            hidden_size = 12288

    if num_layers is None:
        num_layers = hidden_size // 128

    if num_attention_heads is None:
        num_attention_heads = max(8, (hidden_size // 128) * 8)

    head_dim = hidden_size // num_attention_heads

    # ---- 1. NCCL buffers ----
    # Ring buffers per GPU pair. Each pair needs ~1% of model weight size
    # as communication buffer in shared memory.
    bytes_per_param = DTYPE_BYTES[dtype]
    model_size_bytes = model_size_params * 1e9 * bytes_per_param
    buffers_per_pair = model_size_bytes * 0.01
    num_pairs = (tensor_parallel_size * (tensor_parallel_size - 1)) // 2
    nccl_buffers_bytes = buffers_per_pair * num_pairs

    # ---- 2. PyTorch tensor descriptors ----
    # Each GPU worker shares tensor metadata (pointers, shapes, dtype).
    # ~1000 descriptors per layer per worker, ~256 bytes each.
    pt_descriptors_bytes = num_layers * 1000 * 256 * tensor_parallel_size

    # ---- 3. KV cache metadata (block tables, pagination) ----
    # Block size typically 16 tokens. Each block table entry = 8 bytes.
    block_size = 16
    blocks_per_request = math.ceil(max_seq_length / block_size)
    kv_metadata_bytes = (
        max_concurrent_requests * blocks_per_request * 8 * tensor_parallel_size
    )

    # ---- 4. MMAP / batch exchange buffers ----
    # Token batch exchange between scheduler and workers.
    mmap_bytes = (
        max_concurrent_requests * max_seq_length * 2 * tensor_parallel_size
    )

    # ---- Total ----
    total_bytes = nccl_buffers_bytes + pt_descriptors_bytes + kv_metadata_bytes + mmap_bytes

    # Add 30% safety margin
    total_bytes_with_margin = total_bytes * 1.3

    # Round up to nearest GiB
    recommended_gib = math.ceil(total_bytes_with_margin / (1024 ** 3))

    return {
        "nccl_buffers_gb": round(nccl_buffers_bytes / (1024 ** 3), 2),
        "pt_descriptors_gb": round(pt_descriptors_bytes / (1024 ** 3), 2),
        "kv_metadata_gb": round(kv_metadata_bytes / (1024 ** 3), 2),
        "mmap_batch_gb": round(mmap_bytes / (1024 ** 3), 2),
        "total_raw_gb": round(total_bytes / (1024 ** 3), 2),
        "total_with_margin_gb": round(total_bytes_with_margin / (1024 ** 3), 2),
        "recommended_gb": recommended_gib,
        "details": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "head_dim": head_dim,
            "model_size_bytes": model_size_bytes,
            "buffers_per_pair_gb": round(buffers_per_pair / (1024 ** 3), 4),
            "num_nccl_pairs": num_pairs,
            "block_size": block_size,
            "blocks_per_request": blocks_per_request,
            "safety_margin": "30%",
        },
    }


def format_table(result):
    """Format the result as a human-readable table."""
    lines = [
        "=" * 60,
        "  vLLM /dev/shm Size Calculator",
        "=" * 60,
        "",
        "  Component                   Size (GiB)",
        "-" * 60,
        f"  NCCL communication buffers  {result['nccl_buffers_gb']:>10.2f}",
        f"  PyTorch tensor descriptors  {result['pt_descriptors_gb']:>10.2f}",
        f"  KV cache metadata           {result['kv_metadata_gb']:>10.2f}",
        f"  MMAP batch buffers          {result['mmap_batch_gb']:>10.2f}",
        "-" * 60,
        f"  Total (raw)                 {result['total_raw_gb']:>10.2f}",
        f"  Total (+30% margin)         {result['total_with_margin_gb']:>10.2f}",
        "=" * 60,
        f"  RECOMMENDED SHM: {result['recommended_gb']} GiB",
        "=" * 60,
        "",
    ]

    d = result["details"]
    lines.append("  Configuration:")
    lines.append(f"    - Model size:              {d['model_size_bytes'] / 1e9:.0f}B params ({d['dtype'].upper()})")
    lines.append(f"    - Hidden size:             {d['hidden_size']}")
    lines.append(f"    - Num layers:              {d['num_layers']}")
    lines.append(f"    - Num attention heads:     {d['num_attention_heads']}")
    lines.append(f"    - Head dimension:          {d['head_dim']}")
    lines.append(f"    - Block size:              {d['block_size']} tokens")
    lines.append(f"    - Safety margin:           {d['safety_margin']}")
    lines.append("")
    lines.append("  Note: This is an estimate for IPC buffers only.")
    lines.append("  Model weights reside in GPU VRAM, not /dev/shm.")
    lines.append("")

    return "\n".join(lines)


def print_usage():
    """Print usage information."""
    print("""
Usage:
    python shm_size_calculator.py --model-params N [options]

Required:
    --model-params N          Model size in billions of parameters (e.g., 30, 70)

Optional:
    --tensor-parallel-size N  Number of GPUs (default: 4)
    --max-concurrent-requests N  Max concurrent requests (default: 20)
    --max-seq-length N        Max sequence length (default: 8192)
    --hidden-size N           Model hidden size (auto-estimated if omitted)
    --num-layers N            Number of transformer layers (auto-estimated if omitted)
    --num-attention-heads N   Number of attention heads (auto-estimated if omitted)
    --model-profile NAME      Use a built-in model profile (see list with --model-profiles)
    --json                    Output as JSON instead of formatted table
    --help                    Show this help message

Examples:
    python shm_size_calculator.py --model-params 30 --tensor-parallel-size 4
    python shm_size_calculator.py --model-profile mistral-medium --json
    python shm_size_calculator.py --model-params 70 --tp 8 --requests 32 --seq 16384
""")


def print_model_profiles():
    """Print available built-in model profiles."""
    print("\nAvailable model profiles:")
    for name, profile in sorted(MODEL_PROFILES.items()):
        print(f"  {name:25s} hidden={profile['hidden_size']:<6d}  "
              f"layers={profile['num_layers']:<5d}  "
              f"heads={profile['num_attention_heads']}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate recommended /dev/shm size for vLLM tensor parallel deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-params",
        type=float,
        help="Model size in billions of parameters (e.g., 30, 70)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism (default: 4)",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        "--requests",
        type=int,
        default=20,
        help="Maximum concurrent requests (default: 20)",
    )
    parser.add_argument(
        "--max-seq-length",
        "--seq",
        type=int,
        default=8192,
        help="Maximum sequence length (default: 8192)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=list(DTYPE_BYTES.keys()),
        help="Weight precision (default: bf16)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Model hidden size (auto-estimated if omitted)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers (auto-estimated if omitted)",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=None,
        help="Number of attention heads (auto-estimated if omitted)",
    )
    parser.add_argument(
        "--model-profile",
        type=str,
        choices=list(MODEL_PROFILES.keys()),
        default=None,
        help="Use a built-in model profile",
    )
    parser.add_argument(
        "--model-profiles",
        action="store_true",
        help="List all available model profiles",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted table",
    )

    args = parser.parse_args()

    # Show available profiles if requested
    if args.model_profiles:
        print_model_profiles()
        sys.exit(0)

    # Validate required args
    if args.model_params is None and args.model_profile is None:
        print("Error: --model-params is required (or use --model-profile to use a built-in profile).")
        print_usage()
        sys.exit(1)

    # Auto-estimate model params from profile if not explicitly provided
    if args.model_params is None:
        # Approximate parameter count from architecture
        profile = MODEL_PROFILES[args.model_profile]
        hs = profile["hidden_size"]
        nl = profile["num_layers"]
        # Rough estimation: params ≈ layers × 2 × hidden² × 4 / 1e9 (FFN dominant)
        args.model_params = nl * 2 * hs * hs * 4 / 1e9
        # Override other fields from the profile
        args.hidden_size = profile["hidden_size"]
        args.num_layers = profile["num_layers"]
        args.num_attention_heads = profile["num_attention_heads"]
    elif args.model_profile:
        # Apply model profile for architecture params if not manually set
        profile = MODEL_PROFILES[args.model_profile]
        if args.hidden_size is None:
            args.hidden_size = profile["hidden_size"]
        if args.num_layers is None:
            args.num_layers = profile["num_layers"]
        if args.num_attention_heads is None:
            args.num_attention_heads = profile["num_attention_heads"]

    # Calculate
    try:
        result = calculate_shm(
            model_size_params=args.model_params,
            tensor_parallel_size=args.tensor_parallel_size,
            max_concurrent_requests=args.max_concurrent_requests,
            max_seq_length=args.max_seq_length,
            dtype=args.dtype,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Add metadata to details
    result["details"]["model_params"] = args.model_params
    result["details"]["tp_size"] = args.tensor_parallel_size
    result["details"]["max_concurrent_requests"] = args.max_concurrent_requests
    result["details"]["max_seq_length"] = args.max_seq_length
    result["details"]["dtype"] = args.dtype

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_table(result))


if __name__ == "__main__":
    main()
