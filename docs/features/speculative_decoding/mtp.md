# Multi-Token Prediction

[Multi-Token Prediction](https://arxiv.org/pdf/2404.19737) is a speculative decoding model optimization whereby models are trained to predict not only the next token in the sequence, but `N` look-ahead tokens in parallel.

Currently vLLM supports MTP on a per-model architecture basis, meaning that your base model must already have pre-trained MTP heads in the checkpoint and you must specify the model architecture in the speculative decoding method. As of now, the following model architectures (speculation methods) are supported for MTP:

- `qwen3_next_mtp`
- `deepseek_mtp`
- `mimo_mtp`

```{autodata} vllm.config.speculative.MTP_MODEL_TYPES
:annotation:

## Qwen MTP Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    tensor_parallel_size=2,
    speculative_config={
        "num_speculative_tokens": 3,
        "method": "mtp",
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
