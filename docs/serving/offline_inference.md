# Offline Inference

Offline inference is possible in your own code using vLLM's [`LLM`][vllm.LLM] class.

For example, the following code downloads the [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m) model from HuggingFace
and runs it in vLLM using the default configuration.

```python
from vllm import LLM

# Initialize the vLLM engine.
llm = LLM(model="facebook/opt-125m")
```

After initializing the `LLM` instance, use the available APIs to perform model inference.
The available APIs depend on the model type:

- [Generative models](../models/generative_models.md) output logprobs which are sampled from to obtain the final output text.
- [Pooling models](../models/pooling_models.md) output their hidden states directly.

!!! info
    [API Reference](../api/README.md#offline-inference)

## Ray Data LLM API

Ray Data LLM is an alternative offline inference API that uses vLLM as the underlying engine.
This API adds several batteries-included capabilities that simplify large-scale, GPU-efficient inference:

- Streaming execution processes datasets that exceed aggregate cluster memory.
- Automatic sharding, load balancing, and autoscaling distribute work across a Ray cluster with built-in fault tolerance.
- Continuous batching keeps vLLM replicas saturated and maximizes GPU utilization.
- Transparent support for tensor and pipeline parallelism enables efficient multi-GPU inference.
- Reading and writing to most popular file formats and cloud object storage.
- Scaling up the workload without code changes.

??? code

    ```python
    import ray  # Requires ray>=2.44.1
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

    config = vLLMEngineProcessorConfig(model_source="unsloth/Llama-3.2-1B-Instruct")
    processor = build_llm_processor(
        config,
        preprocess=lambda row: {
            "messages": [
                {"role": "system", "content": "You are a bot that completes unfinished haikus."},
                {"role": "user", "content": row["item"]},
            ],
            "sampling_params": {"temperature": 0.3, "max_tokens": 250},
        },
        postprocess=lambda row: {"answer": row["generated_text"]},
    )

    ds = ray.data.from_items(["An old silent pond..."])
    ds = processor(ds)
    ds.write_parquet("local:///tmp/data/")
    ```

For more information about the Ray Data LLM API, see the [Ray Data LLM documentation](https://docs.ray.io/en/latest/data/working-with-llms.html).
