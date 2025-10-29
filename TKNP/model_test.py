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

from transformers import LlamaConfig
from vllm.config import ModelConfig, CacheConfig
from vllm.model_executor.models.llama_vllm import LlamaForCausalLM

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
    
    model_config = ModelConfig(
        model= args.model_name,  # model path or name
        tokenizer=args.model_name,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype=torch.float16,
        seed=0,
        max_model_len=8192,
        enforce_eager=True,
        # Add other model config parameters as needed
    )
    
    # Create parallel configuration
    parallel_config = ParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=1,  # Required when token parallelism is enabled
        token_parallel_size=args.token_parallel_size,
        enable_token_parallel= args.token_parallel_size > 1,
    )
    
    # Create the full vLLM configuration
    vllm_config = VllmConfig(model_config=model_config, parallel_config=parallel_config)

    print(f"  Configuration created:")
    print(f"    Tensor parallel size: {parallel_config.tensor_parallel_size}")
    print(f"    Pipeline parallel size: {parallel_config.pipeline_parallel_size}")
    print(f"    Data parallel size: {parallel_config.data_parallel_size}")
    print(f"    Token parallel enabled: {parallel_config.enable_token_parallel}")
    print(f"    Token parallel size: {parallel_config.token_parallel_size}")
    print(f"    World size: {parallel_config.world_size}")
    
    return vllm_config

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
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model name (default: meta-llama/Llama-3.2-1B)")

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
    hf_config = LlamaConfig.from_pretrained(args.model_name)
    
    # print(hf_config)
    print(vllm_config)
    
    with set_current_vllm_config(vllm_config):
        setup_model_parallel_groups(args)
        # Ensure vllm config is set correctly
        # current_vllm_config = get_current_vllm_config()
        # assert current_vllm_config is not None, "Current vLLM config is None"
        # print(f"Current vLLM config token parallel size: {current_vllm_config.parallel_config.token_parallel_size}")

        model = LlamaForCausalLM(vllm_config=vllm_config)
        model = model.cuda()
        print(model)
        
        # try a forward pass with dummy input
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len)).cuda()
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, positions=positions)
            
        print(outputs)