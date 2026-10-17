# DSpark Draft Models

DSpark generates a block of draft features in parallel, then samples proposals
sequentially with a Markov head. The target verifies the proposals using vLLM's
existing speculative decoding machinery.

## Liquid LFM2.5-VL

Use [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) with
its [DSpark drafter](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark).
The checkpoint's trained block size is nine; set `num_speculative_tokens` to 9.
The target handles image inputs, and the drafter consumes the target's language
features. See the [multimodal input guide](../multimodal_inputs.md) for image
request formats.

The following example uses the validated single-GPU, BF16, eager configuration
with Model Runner V2:

```python
import os

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from vllm import LLM, SamplingParams

llm = LLM(
    model="LiquidAI/LFM2.5-VL-3B",
    dtype="bfloat16",
    max_model_len=8192,
    enforce_eager=True,
    async_scheduling=False,
    enable_prefix_caching=False,
    speculative_config={
        "method": "dspark",
        "model": "LiquidAI/LFM2.5-VL-3B-DSpark",
        "num_speculative_tokens": 9,
        "enable_adaptive_verification": False,
    },
)

outputs = llm.generate(
    ["Explain why the sky is blue in one sentence."],
    SamplingParams(temperature=0, max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

Auxiliary hidden-state capture across pipeline-parallel ranks is not supported
for this target. CUDA graphs, async scheduling, adaptive verification, and prefix
caching require separate validation for this model pair. Performance depends on
input/output lengths and request concurrency. Low-precision verification can
produce different greedy token selections when candidate scores are close.
