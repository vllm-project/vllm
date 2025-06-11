---
title: Offline Inference
---
[](){ #offline-inference }

You can run vLLM in your own code on a list of prompts.

The offline API is based on the [LLM][vllm.LLM] class.
To initialize the vLLM engine, create a new instance of `LLM` and specify the model to run.

For example, the following code downloads the [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m) model from HuggingFace
and runs it in vLLM using the default configuration.

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
```

After initializing the `LLM` instance, you can perform model inference using various APIs.
The available APIs depend on the type of model that is being run:

- [Generative models][generative-models] output logprobs which are sampled from to obtain the final output text.
- [Pooling models][pooling-models] output their hidden states directly.

Please refer to the above pages for more details about each API.

!!! info
    [API Reference][offline-inference-api]
