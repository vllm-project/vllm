# Example usage:
# Without token parallelism: torchrun --nproc-per-node=2 TKNP/test_torchrun.py --tensor-parallel-size 1 --pipeline-parallel-size 1 --enable-token-parallel --token-parallel-size 2
# With token parallelism: torchrun --nproc-per-node=8 TKNP/test_torchrun.py --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --enable-token-parallel --token-parallel-size 2
# General tests: torchrun --nproc-per-node=1 TKNP/test_torchrun.py --tensor-parallel-size 1

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""
experimental support for tensor-parallel + token-parallel inference with torchrun,
see https://github.com/vllm-project/vllm/issues/11400 for
the motivation and use case for this example.
run the script with `torchrun --nproc-per-node=2 torchrun_example.py`,
the argument 2 should match the `tensor_parallel_size` below.
see `tests/distributed/test_torchrun_example.py` for the unit test.
"""

import argparse
import torch.distributed as dist

from vllm import LLM, SamplingParams


def parse_args():
    """Parse command line arguments for distributed vLLM inference."""
    parser = argparse.ArgumentParser(description="Distributed vLLM inference with torchrun")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of tensor parallel processes (default: 1)")
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


def main():
    args = parse_args()

    # Create prompts, the same across all ranks
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is very",
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is very",
    ]

    # Create sampling parameters, the same across all ranks
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Use `distributed_executor_backend="external_launcher"` so that
    # this llm engine/instance only creates one worker.
    # it is important to set an explicit seed to make sure that
    # all ranks have the same random seed, so that sampling can be
    # deterministic across ranks.
    
    # Prepare LLM kwargs - only include token parallel args if enabled
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "distributed_executor_backend": "external_launcher",
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "enforce_eager": True,
    }
    
    # Only add token parallel configs if token parallelism is enabled
    if args.enable_token_parallel:
        if args.token_parallel_size <= 1:
            raise ValueError("Token parallelism requires token_parallel_size > 1")
        llm_kwargs["enable_token_parallel"] = True
        llm_kwargs["token_parallel_size"] = args.token_parallel_size
    
    llm = LLM(**llm_kwargs)

    if dist.get_rank() == 0:
        if args.enable_token_parallel:
            print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}, data_parallel_size={args.data_parallel_size}, token_parallel_size={args.token_parallel_size}, enable_token_parallel={args.enable_token_parallel}")
        else:
            print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}, data_parallel_size={args.data_parallel_size}")

    outputs = llm.generate(prompts, sampling_params)

    # all ranks will have the same outputs
    if dist.get_rank() == 0:
        print("-" * 50)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n")
            print("-" * 50)
            
    # destroy the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Further tips:

1. to communicate control messages across all ranks, use the cpu group,
a PyTorch ProcessGroup with GLOO backend.

```python
from vllm.distributed.parallel_state import get_world_group
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)
if torch_rank == 0:
    # do something for rank 0, e.g. saving the results to disk.
```

2. to communicate data across all ranks, use the model's device group,
a PyTorch ProcessGroup with NCCL backend.
```python
from vllm.distributed.parallel_state import get_world_group
device_group = get_world_group().device_group
```

3. to access the model directly in every rank, use the following code:
```python
llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
```
"""


