---
title: Offline Inference
---

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
    [API Reference][offline-inference-api]

### Ray Data LLM API

Ray Data LLM is an alternative offline inference API that uses vLLM as the underlying engine.
This API adds several batteries-included capabilities that simplify large-scale, GPU-efficient inference:

- Streaming execution processes datasets that exceed aggregate cluster memory.
- Automatic sharding, load balancing, and autoscaling distribute work across a Ray cluster with built-in fault tolerance.
- Continuous batching keeps vLLM replicas saturated and maximizes GPU utilization.
- Transparent support for tensor and pipeline parallelism enables efficient multi-GPU inference.

The following example shows how to run batched inference with Ray Data and vLLM:
<gh-file:examples/offline_inference/batch_llm_inference.py>

For more information about the Ray Data LLM API, see the [Ray Data LLM documentation](https://docs.ray.io/en/latest/data/working-with-llms.html).
