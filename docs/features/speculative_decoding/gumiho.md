# Gumiho Draft Models

Gumiho ([Li et al., ICML 2025](https://arxiv.org/abs/2503.10135);
[reference code](https://github.com/AMD-AIG-AIMA/Gumiho)) is a hybrid
speculative decoding drafter that combines an EAGLE-style transformer draft
head with a set of parallel MLP heads. The first two
speculative tokens are produced autoregressively by the transformer head, and
every additional speculative token is produced in parallel by a dedicated MLP
head conditioned on the embeddings and hidden states of the first two draft
tokens. This trades a small amount of acceptance rate for a meaningful
reduction in draft latency when `num_speculative_tokens > 2`.

The verifier path (target model forward, rejection sampler) is the standard
V1 speculative decoding path; only the drafter is different.

## Gumiho Drafter Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "method": "gumiho",
        "model": "amd/Gumiho-llama3-8b",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

The drafter checkpoint typically contains weights for several MLP heads (one
per supported speculative token beyond the second). vLLM allocates only the
heads that fit within the requested `num_speculative_tokens` and skips the
rest of the weights with a one-shot warning.

## Configuration Notes

- Gumiho uses an EAGLE-like proposer under the hood. As with `mlp_speculator`,
  `draft_tensor_parallel_size` is forced to `1` because the released
  drafter is a single-replica model.
- `num_speculative_tokens` controls how many MLP heads are activated:
  - `num_speculative_tokens == 1` or `2`: only the transformer draft head is
    used (no MLP heads required).
  - `num_speculative_tokens > 2`: the transformer head produces the first two
    tokens and the remaining `num_speculative_tokens - 2` tokens are produced
    in parallel by the MLP heads.
- Gumiho currently only supports greedy draft sampling. When
  `draft_sample_method` is set to `probabilistic`, the proposer falls back to
  the regular sequential EAGLE-style path so it remains correct but no longer
  benefits from parallel MLP drafting.

## Pre-Trained Gumiho Draft Models

- [amd/Gumiho-llama3-8b](https://huggingface.co/amd/Gumiho-llama3-8b)
  (Llama-3 8B Instruct backbone)

!!! note
    Some Gumiho checkpoint repositories ship only the weights without a
    `config.json`. In that case, download the checkpoint and add a
    `config.json` whose top-level `model_type` is `gumiho` and whose `model`
    field contains the underlying Llama config; see the [original paper /
    code release](https://github.com/AMD-AIG-AIMA/Gumiho) for the exact
    structure used during training.
