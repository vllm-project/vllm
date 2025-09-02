#!/usr/bin/env python3
"""
Manual distributed environment setup for testing TokenParallelLinear classes.

This script sets up the distributed environment without using vLLM's LLM class,
allowing you to test the TokenParallelLinear classes in isolation.

Usage:
    # Run with torchrun for multiple processes
    torchrun --nproc_per_node=2 TKNP/test_TKNP_classes.py --tensor-parallel-size=1 --pipeline-parallel-size=1 --token-parallel-size=2
    torchrun --nproc_per_node=2 TKNP/test_TKNP_classes.py --token-parallel-size=2
    # Or run single process for debugging
    python test_TKNP_classes.py --token_parallel_size=1
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional

# Add vllm to path if needed
# sys.path.insert(0, '/home/grads/s/sls7161/Documents/MLSystems/vllm-distributed')

from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config, get_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    ensure_model_parallel_initialized
)
from vllm.distributed.parallel_state import (
    get_tp_group, get_pp_group, get_dp_group, get_tknp_group,
    is_tknp_initialized, get_tknp_rank, get_tknp_world_size, initialize_model_parallel
)
from vllm.model_executor.layers.token_parallel_linear import (
    TokenParallelQKVLinear,
    TokenParallelRowLinear,
    create_token_parallel_qkv_linear,
    create_token_parallel_row_linear
)

def setup_distributed_environment():
    """Initialize the distributed environment using torchrun environment variables."""
    # Get distributed info from torchrun environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Process {rank}: Initializing distributed environment")
    print(f"  world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    # Initialize distributed environment
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl"
    )
    
    return world_size, rank, local_rank

def setup_model_parallel_groups(args):
    """Setup model parallel groups."""
    print(f"Process {dist.get_rank()}: Setting up model parallel groups")
    
    # Initialize model parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_parallel_size,
    )

def setup_vllm_config(args):
    """Setup vLLM configuration with token parallelism."""
    print(f"Process {dist.get_rank()}: Setting up vLLM configuration")
    
    # Create parallel configuration
    parallel_config = ParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=1,  # Required when token parallelism is enabled
        token_parallel_size=args.token_parallel_size,
        enable_token_parallel= args.token_parallel_size > 1,
    )
    
    # Create the full vLLM configuration
    vllm_config = VllmConfig(parallel_config=parallel_config)
    
    print(f"  Configuration created:")
    print(f"    Tensor parallel size: {parallel_config.tensor_parallel_size}")
    print(f"    Pipeline parallel size: {parallel_config.pipeline_parallel_size}")
    print(f"    Data parallel size: {parallel_config.data_parallel_size}")
    print(f"    Token parallel enabled: {parallel_config.enable_token_parallel}")
    print(f"    Token parallel size: {parallel_config.token_parallel_size}")
    print(f"    World size: {parallel_config.world_size}")
    
    return vllm_config

def test_process_groups():
    """Test that all process groups are properly initialized and working."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"\nProcess {rank}: Testing process groups")
    print("=" * 50)
    
    # Test tensor parallel group
    tp_group = get_tp_group()
    print(f"Process {rank}: TP group - rank {tp_group.rank_in_group}/{tp_group.world_size}")
    
    # Test pipeline parallel group  
    pp_group = get_pp_group()
    print(f"Process {rank}: PP group - rank {pp_group.rank_in_group}/{pp_group.world_size}")
    
    # Test data parallel group
    dp_group = get_dp_group()
    print(f"Process {rank}: DP group - rank {dp_group.rank_in_group}/{dp_group.world_size}")
    
    # Test token parallel group if enabled
    if is_tknp_initialized():
        tknp_group = get_tknp_group()
        tknp_rank = get_tknp_rank()
        tknp_world_size = get_tknp_world_size()
        print(f"Process {rank}: TKNP group - rank {tknp_rank}/{tknp_world_size}")
        
    else:
        print(f"Process {rank}: Token parallelism not enabled or not initialized")

def test_token_parallel_qkv_linear():
    """Test TokenParallelQKVLinear layer."""
    print(f"\n=== Testing TokenParallelQKVLinear on Rank {dist.get_rank()} ===")
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    head_size = 64
    num_heads = 12
    num_kv_heads = 12
    
    # Create test input
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Is token parallel initialized: {is_tknp_initialized()}")
    if is_tknp_initialized():
        print(f"Token parallel rank: {get_tknp_rank()}")
        print(f"Token parallel world size: {get_tknp_world_size()}")
    
    # Create TokenParallelQKVLinear layer
    try:
        qkv_layer = create_token_parallel_qkv_linear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )
        
        print(f"QKV layer created: {qkv_layer}")
        print(f"Is root rank: {qkv_layer.is_root_rank}")
        
        # Forward pass
        with torch.no_grad():
            output = qkv_layer(x)
            print(f"QKV output shape: {output.shape}")
            print(f"QKV output dtype: {output.dtype}")
            print(f"QKV output device: {output.device}")
            
        print("✓ TokenParallelQKVLinear test passed")
        
    except Exception as e:
        print(f"✗ TokenParallelQKVLinear test failed: {e}")
        import traceback
        traceback.print_exc()


def test_token_parallel_row_linear():
    """Test TokenParallelRowLinear layer."""
    print(f"\n=== Testing TokenParallelRowLinear on Rank {dist.get_rank()} ===")
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    input_size = 768
    output_size = 768
    
    # Create test input
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    x = torch.randn(batch_size, seq_len, input_size, dtype=torch.float16, device=device)
    
    print(f"Input shape: {x.shape}")
    
    # Create TokenParallelRowLinear layer
    try:
        row_layer = create_token_parallel_row_linear(
            input_size=input_size,
            output_size=output_size,
            bias=True,
            reduce_results=True,
        )
        
        print(f"Row layer created: {row_layer}")
        print(f"Is root rank: {row_layer.is_root_rank}")
        
        # Forward pass
        with torch.no_grad():
            output = row_layer(x)
            print(f"Row output shape: {output.shape}")
            print(f"Row output dtype: {output.dtype}")
            print(f"Row output device: {output.device}")
            
        print("✓ TokenParallelRowLinear test passed")
        
    except Exception as e:
        print(f"✗ TokenParallelRowLinear test failed: {e}")
        import traceback
        traceback.print_exc()


def synchronize_and_cleanup():
    """Synchronize all processes and cleanup."""
    if dist.is_initialized():
        dist.barrier()
        print(f"Rank {dist.get_rank()}: All tests completed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test token parallel process group setup")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of tensor parallel processes (default: 1)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--token-parallel-size", type=int, default=1,
                        help="Number of token parallel processes (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Setup distributed environment
    world_size, rank, local_rank = setup_distributed_environment()
    
    # Set CUDA device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Process {rank}: Set CUDA device to {local_rank}")
    
    # Setup vLLM configuration
    vllm_config = setup_vllm_config(args)
    
    with set_current_vllm_config(vllm_config):
        setup_model_parallel_groups(args)
        # Ensure vllm config is set correctly
        # current_vllm_config = get_current_vllm_config()
        # assert current_vllm_config is not None, "Current vLLM config is None"
        # print(f"Current vLLM config token parallel size: {current_vllm_config.parallel_config.token_parallel_size}")

        test_process_groups()
        
        # build token parallel classes
        tknp_qkv_linear = create_token_parallel_qkv_linear(
            hidden_size=8192,
            head_size=128,
            total_num_heads=8192 // 128,
            total_num_kv_heads=8,
            bias=False,
        )

        print(f"Rank {dist.get_rank()}, TokenParallelQKVLinear: {tknp_qkv_linear}")
