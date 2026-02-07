# Speculating using Suffix Decoding

The following code configures vLLM to use speculative decoding where proposals are generated using Suffix Decoding ([technical report](https://arxiv.org/abs/2411.04975)).

Like n-gram, Suffix Decoding can generate draft tokens by pattern-matching using the last `n` generated tokens. Unlike n-gram, Suffix Decoding (1) can pattern-match against both the prompt and previous generations, (2) uses frequency counts to propose the most likely continuations, and (3) speculates an adaptive number of tokens for each request at each iteration to get better acceptance rates.

Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

!!! tip "Install Arctic Inference"
    Suffix Decoding requires [Arctic Inference](https://github.com/snowflakedb/ArcticInference). You can install it with `pip install arctic-inference`.

!!! tip "Suffix Decoding Speculative Tokens"
    Suffix Decoding will speculate a dynamic number of tokens for each request at each decoding step, so the `num_speculative_tokens` configuration specifies the *maximum* number of speculative tokens. It is suggested to use a high number such as `16` or `32` (default).

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```