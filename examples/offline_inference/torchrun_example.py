"""
experimental support for tensor-parallel inference with torchrun,
see https://github.com/vllm-project/vllm/issues/11400 for
the motivation and use case for this example.
run the script with `torchrun --nproc-per-node=2 torchrun_example.py`,
the argument 2 should match the `tensor_parallel_size` below.
see `tests/distributed/test_torchrun_example.py` for the unit test.
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
llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=2,
    distributed_executor_backend="external_launcher",
)

outputs = llm.generate(prompts, sampling_params)

# all ranks will have the same outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")
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
