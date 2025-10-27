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
"""

from vllm import LLM, SamplingParams

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
    model="microsoft/Phi-mini-MoE-instruct",
    tensor_parallel_size=1,
    data_parallel_size=2,
    pipeline_parallel_size=1,
    enable_expert_parallel=False,
    distributed_executor_backend="external_launcher",
    max_model_len=4096,
    gpu_memory_utilization=0.6,
    seed=1,
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
