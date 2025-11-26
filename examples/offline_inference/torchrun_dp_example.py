# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
experimental support for data-parallel inference with torchrun
Note the data load balancing and distribution is done out of the vllm engine,
no internal lb supported in external_launcher mode.

To run this example:
```bash
$ torchrun --nproc-per-node=2 examples/offline_inference/torchrun_dp_example.py
```

With custom parallelism settings:
```bash
$ torchrun --nproc-per-node=8 examples/offline_inference/torchrun_dp_example.py \
    --tp-size=2 --pp-size=1 --dp-size=4 --enable-ep
```
"""

import argparse

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data-parallel inference with torchrun"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size (default: 1)",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=2,
        help="Data parallel size (default: 2)",
    )
    parser.add_argument(
        "--enable-ep",
        action="store_true",
        help="Enable expert parallel (default: False)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-mini-MoE-instruct",
        help="Model name or path (default: microsoft/Phi-mini-MoE-instruct)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model length (default: 4096)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization (default: 0.6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    return parser.parse_args()


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

# Use `distributed_executor_backend="external_launcher"` so that
# this llm engine/instance only creates one worker.
# it is important to set an explicit seed to make sure that
# all ranks have the same random seed, so that sampling can be
# deterministic across ranks.
llm = LLM(
    model=args.model,
    tensor_parallel_size=args.tp_size,
    data_parallel_size=args.dp_size,
    pipeline_parallel_size=args.pp_size,
    enable_expert_parallel=args.enable_ep,
    distributed_executor_backend="external_launcher",
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    seed=args.seed,
)

dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
dp_size = llm.llm_engine.vllm_config.parallel_config.data_parallel_size

prompts = [
    f"{idx}.{prompt}" for idx, prompt in enumerate(prompts) if idx % dp_size == dp_rank
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(
        f"DP Rank: {dp_rank} Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n"
    )

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
