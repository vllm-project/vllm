# Suffix Decoding

The following code configures vLLM to use speculative decoding where proposals are generated using Suffix Decoding ([technical report](https://arxiv.org/abs/2411.04975)).

Like n-gram, Suffix Decoding generates draft tokens by pattern-matching on the last generated tokens. Unlike n-gram, Suffix Decoding (1) can pattern-match against both the request's own prompt and output and the responses of previous requests, and (2) uses frequency counts to propose the most likely continuations.

Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 8,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## How it works

Suffix decoding runs on the GPU and works with async scheduling. It builds on [`ngram_gpu`](n_gram.md):

- Each request is matched against its own prompt and output using the longest matching suffix, up to `suffix_decoding_max_tree_depth` tokens, rather than a fixed n-gram window.
- Responses of finished requests are kept in a GPU corpus (`suffix_decoding_corpus_tokens`, 1M tokens by default, oldest first out; `suffix_decoding_max_cached_requests: 0` disables it). Each step, every request looks up candidate positions in the corpus by its last 2 and last 4 tokens and extends them to the longest suffix match (so corpus matching needs `suffix_decoding_max_tree_depth` of at least 2); the continuation is chosen by how often each next token follows those matches.

The longer match wins. The draft always has `num_speculative_tokens` tokens; slots without a match are filled with the last sampled token. Without adaptive verification (below), every slot is verified, so keep `num_speculative_tokens` moderate (the default is `8`). `suffix_decoding_max_spec_factor` and `suffix_decoding_min_token_prob` are deprecated and have no effect.

Seeded sampling isn't reproducible with suffix decoding: drafts come from previously finished requests, and with adaptive verification the number of verified slots depends on the batch. For reproducible seeded sampling, set `suffix_decoding_corpus_tokens: 0` and leave adaptive verification off.

!!! warning "The corpus is shared"
    The corpus holds responses from all requests, whatever their `cache_salt` or LoRA adapter, and any request can draft from it. Drafts never change the output distribution, but how many are accepted shows in decode speed. Where requests must not influence each other, set `suffix_decoding_corpus_tokens: 0`.

## Adaptive verification

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
