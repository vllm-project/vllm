#!/usr/bin/env python3
# torchrun --nproc-per-node=8 HTTP/test_process_groups_only.py --tensor-parallel-size 4 --pipeline-parallel-size 1 --token-parallel-size 2 --enable-token-parallel
"""
Simple script to test token parallel process group setup without full LLM instantiation.
This is useful for verifying the distributed infrastructure before testing the full attention implementation.
"""

import argparse
import os
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import (
    init_distributed_environment, 
    initialize_model_parallel,
    get_tp_group,
    get_pp_group, 
    get_dp_group,
    get_tknp_group,
    is_tknp_initialized,
    get_tknp_rank,
    get_tknp_world_size,
)
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test token parallel process group setup")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Number of tensor parallel processes (default: 2)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--token-parallel-size", type=int, default=2,
                        help="Number of token parallel processes (default: 2)")
    parser.add_argument("--enable-token-parallel", action="store_true",
                        help="Enable token parallelism")
    return parser.parse_args()


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


def setup_vllm_config(args):
    """Setup vLLM configuration with token parallelism."""
    print(f"Process {dist.get_rank()}: Setting up vLLM configuration")
    
    # Create parallel configuration
    parallel_config = ParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=1,  # Required when token parallelism is enabled
        token_parallel_size=args.token_parallel_size if args.enable_token_parallel else 1,
        enable_token_parallel=args.enable_token_parallel,
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


def setup_model_parallel_groups(args):
    """Setup model parallel groups."""
    print(f"Process {dist.get_rank()}: Setting up model parallel groups")
    
    # Initialize model parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_parallel_size,
    )


def test_process_groups(args):
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
    if args.enable_token_parallel and is_tknp_initialized():
        tknp_group = get_tknp_group()
        tknp_rank = get_tknp_rank()
        tknp_world_size = get_tknp_world_size()
        print(f"Process {rank}: TKNP group - rank {tknp_rank}/{tknp_world_size}")
        
        # Test token parallel communication with a simple all-reduce
        test_tensor = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        print(f"Process {rank}: Before TKNP all-reduce: {test_tensor.item()}")
        
        # Perform all-reduce within token parallel group
        reduced_tensor = tknp_group.all_reduce(test_tensor)
        print(f"Process {rank}: After TKNP all-reduce: {reduced_tensor.item()}")
        
    else:
        print(f"Process {rank}: Token parallelism not enabled or not initialized")
    
    # Test tensor parallel communication
    print(f"\nProcess {rank}: Testing TP group communication")
    test_tensor = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    print(f"Process {rank}: Before TP all-reduce: {test_tensor.item()}")
    
    reduced_tensor = tp_group.all_reduce(test_tensor)
    print(f"Process {rank}: After TP all-reduce: {reduced_tensor.item()}")
    
    print(f"Process {rank}: All process group tests completed!")


def main():
    args = parse_args()
    
    # Set up distributed environment
    world_size, rank, local_rank = setup_distributed_environment()
    
    # Set CUDA device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Process {rank}: Set CUDA device to {local_rank}")
    
    # Setup vLLM configuration
    vllm_config = setup_vllm_config(args)
    
    # Set the vLLM configuration and setup model parallel groups
    with set_current_vllm_config(vllm_config):
        setup_model_parallel_groups(args)
        
        # Test process groups
        test_process_groups(args)
    
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("\nAll processes completed successfully!")


if __name__ == "__main__":
    main() 