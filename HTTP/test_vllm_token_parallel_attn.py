import argparse
import torch.distributed as dist
import os
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel
)


def parse_args():
    """Parse command line arguments for distributed vLLM inference."""
    parser = argparse.ArgumentParser(description="Distributed vLLM inference with torchrun")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of tensor parallel processes (default: 4)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Number of data parallel processes (default: 1)")
    parser.add_argument("--token-parallel-size", type=int, default=1,
                        help="Number of token parallel processes (default: 1)")
    parser.add_argument("--enable-token-parallel", action="store_true",
                        help="Enable token parallelism")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model length (default: 32768)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    
    return parser.parse_args()

def setup_distributed_environment():
    """Initialize the distributed environment using torchrun environment variables."""
    # Get distributed info from torchrun environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Initializing distributed environment: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
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
    """Setup model parallel groups with token parallelism support."""
    print(f"Setting up model parallel groups:")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  Pipeline parallel size: {args.pipeline_parallel_size}")
    print(f"  Token parallel enabled: {args.enable_token_parallel}")
    print(f"  Token parallel size: {args.token_parallel_size}")
    
    # Initialize model parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_parallel_size,
    )

def test_token_parallel_attention(args):
    """Test token parallel attention by creating an LLM instance."""
    print(f"Creating LLM instance with model: {args.model}")
    
    try:
        # Create LLM instance with token parallelism configuration
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
        }
        
        # Configure token parallelism if enabled
        if args.enable_token_parallel:
            llm_kwargs.update({
                "token_parallel_size": args.token_parallel_size,
                "enable_token_parallel": True,
                "data_parallel_size": 1,  # Required when token parallelism is enabled
            })
            print(f"Token parallelism enabled: token_parallel_size={args.token_parallel_size}")
        
        llm = LLM(**llm_kwargs)
        
        print("LLM instance created successfully!")
        
        # Test with a simple prompt
        prompts = ["Hello, how are you today?"]
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=100
        )
        
        print("Testing generation...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Print results
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            
        return True
        
    except Exception as e:
        print(f"Error creating LLM or generating: {e}")
        return False

if __name__ == "__main__":
    args = parse_args()

    # Initialize the distributed environment
    world_size, rank, local_rank = setup_distributed_environment()
    
    # Set CUDA device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Set CUDA device to: {local_rank}")

    # Initialize the model parallel groups
    setup_model_parallel_groups(args)
    
    # Test token parallel attention implementation
    success = test_token_parallel_attention(args)
    
    if success:
        print("Token parallel attention test completed successfully!")
    else:
        print("Token parallel attention test failed!")
    
    # Clean barrier to ensure all processes complete
    if dist.is_initialized():
        dist.barrier()
        print(f"Process {rank} completed successfully!")