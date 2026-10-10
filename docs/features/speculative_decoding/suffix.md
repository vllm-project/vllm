# Suffix Decoding

The following code configures vLLM to use speculative decoding where proposals are generated using Suffix Decoding ([technical report](https://arxiv.org/abs/2411.04975)).

Like n-gram, Suffix Decoding can generate draft tokens by pattern-matching using the last `n` generated tokens. Unlike n-gram, Suffix Decoding (1) can pattern-match against both the prompt and previous generations, (2) uses frequency counts to propose the most likely continuations, and (3) speculates an adaptive number of tokens for each request at each iteration to get better acceptance rates.

Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

!!! tip "Install Arctic Inference (V1 model runner only)"
    On the V1 model runner, Suffix Decoding requires [Arctic Inference](https://github.com/snowflakedb/ArcticInference). You can install it with `pip install arctic-inference`. Model Runner V2 (the default) runs suffix decoding on the GPU and does not need it; see [below](#model-runner-v2).

!!! tip "Suffix Decoding Speculative Tokens"
    Suffix Decoding will speculate a dynamic number of tokens for each request at each decoding step, so the `num_speculative_tokens` configuration specifies the *maximum* number of speculative tokens. It is suggested to use a high number such as `16` or `32` (default).

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
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

## Model Runner V2

On Model Runner V2, `method: "suffix"` runs on the GPU, works with async scheduling and does not need Arctic Inference. It builds on [`ngram_gpu`](n_gram.md):

- Each request is matched against its own prompt and output using the longest matching suffix, up to `suffix_decoding_max_tree_depth` tokens, rather than a fixed n-gram window.
- Responses of finished requests are kept in a GPU corpus (`suffix_decoding_corpus_tokens`, 1M tokens by default, oldest first out; `suffix_decoding_max_cached_requests: 0` disables it). Each step, every request looks up candidate positions in the corpus by its last 2 and last 4 tokens and extends them to the longest suffix match; the continuation is chosen by how often each next token follows those matches.

The longer match wins. The draft always has `num_speculative_tokens` tokens; slots without a match are filled with the last sampled token, and `suffix_decoding_max_spec_factor` and `suffix_decoding_min_token_prob` are not used. Without adaptive verification (below), Model Runner V2 verifies every slot, so prefer a moderate `num_speculative_tokens` such as `8`.

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 8,
        "enable_adaptive_verification": True,
    },
)
```

!!! tip "Use adaptive verification at higher concurrency"
    Verifying every draft slot costs compute once the GPU is no longer memory-bound, so without trimming suffix decoding can be slower than no speculation at high concurrency. With [adaptive verification](adaptive_verification.md), the speculator reports a per-slot acceptance estimate (learned online by draft source, match length and position) and the scheduler verifies only the slots worth their cost. It requires full CUDA graphs and an attention backend that replays variable-length decode batches (for example `TRITON_ATTN`, or FlashAttention 3).
