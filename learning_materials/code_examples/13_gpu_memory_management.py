"""
Example 13: GPU Memory Management

Shows how to monitor and manage GPU memory usage.

Usage:
    python 13_gpu_memory_management.py
"""

import torch
from vllm import LLM, SamplingParams


def get_gpu_memory_info():
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def main():
    """Demo GPU memory management."""
    print("=== GPU Memory Management ===\n")

    # Check initial memory
    print("Initial GPU memory:")
    info = get_gpu_memory_info()
    if info["available"]:
        print(f"  Allocated: {info['allocated_gb']:.2f} GB")
        print(f"  Reserved: {info['reserved_gb']:.2f} GB")
        print(f"  Total: {info['total_gb']:.2f} GB")
    else:
        print("  CUDA not available")

    # Load model with memory control
    print("\nLoading model with 0.8 GPU memory utilization...")
    llm = LLM(
        model="facebook/opt-125m",
        trust_remote_code=True,
        gpu_memory_utilization=0.8  # Use 80% of GPU memory
    )

    # Check memory after loading
    print("\nAfter model loading:")
    info = get_gpu_memory_info()
    if info["available"]:
        print(f"  Allocated: {info['allocated_gb']:.2f} GB")
        print(f"  Reserved: {info['reserved_gb']:.2f} GB")
        print(f"  Usage: {(info['allocated_gb'] / info['total_gb']) * 100:.1f}%")

    # Run inference
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    llm.generate(["Test prompt"], sampling_params)

    print("\nAfter inference:")
    info = get_gpu_memory_info()
    if info["available"]:
        print(f"  Allocated: {info['allocated_gb']:.2f} GB")


if __name__ == "__main__":
    main()
