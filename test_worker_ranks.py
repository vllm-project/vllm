#!/usr/bin/env python3
"""
Simple test script to verify worker rank printing functionality.

Usage:
    python test_worker_ranks.py --tensor-parallel-size 2
    python test_worker_ranks.py --tensor-parallel-size 2 --pipeline-parallel-size 2
"""

import argparse
import torch
from vllm import LLM


def main():
    parser = argparse.ArgumentParser(description="Test worker rank printing")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small")
    
    args = parser.parse_args()
    
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size
    
    print(f"Configuration:")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"  Pipeline Parallel Size: {args.pipeline_parallel_size}")
    print(f"  Data Parallel Size: {args.data_parallel_size}")
    print(f"  Total GPUs Required: {total_gpus}")
    print(f"  Model: {args.model}")
    
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"  Available GPUs: {available_gpus}")
        if available_gpus < total_gpus:
            print(f"  WARNING: Not enough GPUs! Need {total_gpus}, have {available_gpus}")
    
    print("\nInitializing LLM with worker rank printing enabled...")
    
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            data_parallel_size=args.data_parallel_size,
            max_model_len=512,
            dtype="auto",
            trust_remote_code=True,
            print_worker_ranks=True,  # Enable rank printing
        )
        
        print("✓ LLM initialized successfully!")
        print("✓ Worker rank information should have been printed above.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 