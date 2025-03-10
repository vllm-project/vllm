(serving-llamaindex)=

# LlamaIndex

vLLM is also available via [LlamaIndex](https://github.com/run-llama/llama_index) .

To install LlamaIndex, run

```console
pip install llama-index-llms-vllm -q
```

To run inference on a single or multiple GPUs, use `Vllm` class from `llamaindex`.

```python
from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
)
```

Please refer to this [Tutorial](https://docs.llamaindex.ai/en/latest/examples/llm/vllm/) for more details.
