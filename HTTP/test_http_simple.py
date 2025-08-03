# Simple distributed test without token parallelism
# torchrun --nproc-per-node=4 HTTP/test_http_simple.py --tensor-parallel-size 4 --pipeline-parallel-size 1

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simple test for distributed vLLM inference without token parallelism.
This helps verify that the basic distributed setup is working before
adding token parallelism complexity.
"""

import argparse
import torch.distributed as dist

from vllm import LLM, SamplingParams


def parse_args():
    """Parse command line arguments for distributed vLLM inference."""
    parser = argparse.ArgumentParser(description="Simple distributed vLLM inference test")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of tensor parallel processes (default: 4)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model length (default: 32768)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Create prompts, the same across all ranks
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create sampling parameters, the same across all ranks
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create LLM without token parallelism
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        distributed_executor_backend="external_launcher",
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    if dist.get_rank() == 0:
        print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}")

    outputs = llm.generate(prompts, sampling_params)

    # all ranks will have the same outputs
    if dist.get_rank() == 0:
        print("-" * 50)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n")
            print("-" * 50)


if __name__ == "__main__":
    main() 