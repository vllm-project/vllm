# Frequently Asked Questions

> Q: How can I serve multiple models on a single port using the OpenAI API?

A: Assuming that you're referring to using OpenAI compatible server to serve multiple models at once, that is not currently supported, you can run multiple instances of the server (each serving a different model) at the same time, and have another layer to route the incoming request to the correct server accordingly.

---

> Q: Which model to use for offline inference embedding?

A: You can try [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) and [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5);
more are listed [here](../models/supported_models.md).

By extracting hidden states, vLLM can automatically convert text generation models like [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B),
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) into embedding models,
but they are expected to be inferior to models that are specifically trained on embedding tasks.

---

> Q: Can the output of a prompt vary across runs in vLLM?

A: Yes, it can. vLLM does not guarantee stable log probabilities (logprobs) for the output tokens. Variations in logprobs may occur due to
numerical instability in Torch operations or non-deterministic behavior in batched Torch operations when batching changes. For more details,
see the [Numerical Accuracy section](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations).

In vLLM, the same requests might be batched differently due to factors such as other concurrent requests,
changes in batch size, or batch expansion in speculative decoding. These batching variations, combined with numerical instability of Torch operations,
can lead to slightly different logit/logprob values at each step. Such differences can accumulate, potentially resulting in
different tokens being sampled. Once a different token is sampled, further divergence is likely.

## Mitigation Strategies

- For improved stability and reduced variance, use `float32`. Note that this will require more memory.
- If using `bfloat16`, switching to `float16` can also help.
- Using request seeds can aid in achieving more stable generation for temperature > 0, but discrepancies due to precision differences may still occur.

---

> Q: How do you load weights from CPU?

A: vLLM supports loading model weights from CPU using the `pt_load_map_location` parameter. This parameter controls where PyTorch checkpoints are loaded to and is especially useful when:

- You have model weights stored on CPU and want to load them directly
- You need to manage memory usage by loading weights to CPU first
- You want to load from specific device mappings

## Usage Examples

### Command Line Interface

```bash
# Load weights from CPU
vllm serve meta-llama/Llama-2-7b-hf --pt-load-map-location cpu

# Load from specific device mapping (e.g., CUDA device 1 to device 0)
vllm serve meta-llama/Llama-2-7b-hf --pt-load-map-location '{"cuda:1": "cuda:0"}'
```

### Python API

```python
from vllm import LLM

# Load weights from CPU
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    pt_load_map_location="cpu"
)

# Load with device mapping
llm = LLM(
    model="meta-llama/Llama-2-7b-hf", 
    pt_load_map_location={"cuda:1": "cuda:0"}
)
```

The `pt_load_map_location` parameter accepts the same values as PyTorch's [`torch.load(map_location=...)`](https://pytorch.org/docs/stable/generated/torch.load.html) parameter:

- `"cpu"` - Load all weights to CPU
- `"cuda"` - Load all weights to CUDA (equivalent to `{"": "cuda"}`)
- `{"cuda:1": "cuda:0"}` - Map weights from CUDA device 1 to device 0
- Custom device mappings as needed

Note: This parameter defaults to `"cpu"` and primarily affects PyTorch `.pt`/`.bin` checkpoint files. For optimal performance on GPU inference, weights will be moved to the target device after loading.
