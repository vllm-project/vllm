#!/usr/bin/env python3
# python HTTP/test_dist_ranks.py --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 1
# python HTTP/test_dist_ranks.py --tensor-parallel-size 2 --pipeline-parallel-size 2 --token-parallel-size 2 --enable-tknp
"""
Test script to instantiate a vLLM LLM with distributed configuration
and print rank information from worker processes using the new print_worker_ranks flag.

Usage:
    python test_distributed_ranks_with_flag.py --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 1
    python test_distributed_ranks_with_flag.py --tensor-parallel-size 2 --pipeline-parallel-size 2 --data-parallel-size 1
    python test_distributed_ranks_with_flag.py --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 2
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM distributed configuration and print rank information from workers"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for pipeline parallelism (default: 1)"
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for data parallelism (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-small",
        help="Model to load (default: microsoft/DialoGPT-small)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model length (default: 512)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype (default: auto)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model"
    )
    parser.add_argument(
        "--token-parallel-size",
        type=int,
        default=1,
        help="Number of token parallel groups (default: 1)"
    )
    parser.add_argument(
        "--enable-token-parallel",
        action="store_true",
        help="Enable token parallelism"
    )
    
    args = parser.parse_args()
    
    # Calculate total GPUs needed
    total_gpus = args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size
    
    print(f"Configuration:")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"  Pipeline Parallel Size: {args.pipeline_parallel_size}")
    print(f"  Data Parallel Size: {args.data_parallel_size}")
    print(f"  Token Parallel Size: {args.token_parallel_size}")
    print(f"  Total GPUs Required: {total_gpus}")
    print(f"  Model: {args.model}")
    print(f"  Print Worker Ranks: ENABLED")
    
    # Check if we have enough GPUs
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"  Available GPUs: {available_gpus}")
        if available_gpus < total_gpus:
            print(f"  WARNING: Not enough GPUs available! Need {total_gpus}, have {available_gpus}")
            print(f"  This will likely cause an error during initialization.")
    else:
        print("  WARNING: CUDA not available!")
    
    print("\nInitializing vLLM LLM with worker rank printing enabled...")
    print("=" * 80)
    
    try:
        # Initialize the LLM with distributed configuration and worker rank printing
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            data_parallel_size=args.data_parallel_size,
            token_parallel_size=args.token_parallel_size,
            enable_token_parallel=args.enable_token_parallel,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            disable_log_stats=True,  # Disable stats logging for cleaner output
            print_worker_ranks=True,  # This is the new flag!
            distributed_executor_backend="ray",
        )
        
        print("=" * 80)
        print("✓ LLM initialized successfully!")
        
        # Check if distributed environment is initialized in main process
        print(f"\nMain Process Distributed Environment Status:")
        print(f"  torch.distributed.is_initialized(): {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"  torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")
            print(f"  torch.distributed.get_rank(): {torch.distributed.get_rank()}")
        else:
            print("  (This is expected - main process has no distributed environment)")
        
        # For multiprocessing setups, explain what happened
        if total_gpus > 1:
            print(f"\n✓ Worker rank information should have been printed above by each of the {total_gpus} worker processes.")
            print("Each worker process printed its own rank information during initialization.")
        else:
            print("\nNote: With only 1 GPU, no worker processes were created.")
            print("The distributed environment is not initialized for single-GPU setups.")
        
    except Exception as e:
        print(f"✗ Error during initialization: {e}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)
    
    #sleep for 10 seconds
    time.sleep(2)
    print("\n✓ Test completed successfully!")
    print("\nThe rank information shown above was printed by the worker processes themselves,")
    print("demonstrating that the distributed environment is initialized and working correctly")
    print("within each worker process.")


if __name__ == "__main__":
    main()