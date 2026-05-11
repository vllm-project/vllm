# Score Mode: Perplexity and KLD Evaluation

This page describes how to use vLLM's score mode for efficient model evaluation
via perplexity (PPL) and Kullback-Leibler divergence (KLD) computation.

## Overview

Score mode enables GPU-side extraction of log-probabilities for specific target
tokens, avoiding the overhead of transferring full vocabulary logprobs to CPU.
This makes sliding-window perplexity and KLD calculations significantly faster
than extracting all logprobs and post-processing on CPU.

Three new `SamplingParams` fields control this behavior:

| Parameter | Type | Description |
|-----------|------|-------------|
| `score_mode` | `bool` | Extract only target token logprobs on GPU (for PPL). Requires `prompt_logprobs` to be set. |
| `return_prompt_logits` | `bool` | Return raw logits for all prompt positions (for generating reference logits). |
| `kld_mode` | `bool` | Compute KL divergence on GPU against reference logits. Mutually exclusive with `return_prompt_logits`. |

Three new `TokensPrompt` fields pass per-request data:

| Field | Type | Description |
|-------|------|-------------|
| `target_token_ids` | `list[int]` | Target tokens for score mode (typically `prompt_token_ids[1:]`). |
| `reference_logits_path` | `str` | Path to safetensors file with reference logits (KLD mode). |
| `reference_logits_key` | `str` | Key within the safetensors file for this window's reference logits. |

!!! note
    Score mode is supported on the v1 engine only.

## Perplexity Calculation

Perplexity measures how well a model predicts a sequence. Lower values indicate
better predictions. Score mode computes per-token logprobs on GPU for specified
target tokens, which are then aggregated into a perplexity score.

### Quick Start

```python
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Tokenize your text
tokens = llm.llm_engine.tokenizer.encode(
    "The quick brown fox jumps over the lazy dog.",
    add_special_tokens=False,
)

prompt: TokensPrompt = {
    "prompt_token_ids": tokens,
    "target_token_ids": tokens[1:],  # predict each next token
}

sampling_params = SamplingParams(
    prompt_logprobs=1,
    max_tokens=1,
    score_mode=True,
)

outputs = llm.generate([prompt], sampling_params=sampling_params)
```

For a complete sliding-window implementation compatible with EXL3's evaluation
methodology, see the example script:

[examples/offline_inference/score_mode_perplexity.py](../../examples/offline_inference/score_mode_perplexity.py)

## KL Divergence Calculation

KLD measures how much a quantized model's predictions diverge from a reference
(typically full-precision) model. Lower values indicate the quantization
preserves more of the original model's behavior.

### Two-Phase Workflow

KLD calculation uses a two-phase approach so only one model is loaded at a time:

1. **Phase 1 -- Generate reference logits**: Run the reference model with
   `return_prompt_logits=True` to obtain raw logits, then save them to a
   safetensors file.

2. **Phase 2 -- Compute KLD**: Run the test model with `kld_mode=True` and
   pass the reference logits path in the prompt. All KL math is computed on GPU.

### Quick Start

```python
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

# Phase 1: Get reference logits
ref_llm = LLM(model="/path/to/reference_model")

prompt: TokensPrompt = {
    "prompt_token_ids": tokens,
    "target_token_ids": tokens[1:],
}

ref_params = SamplingParams(
    prompt_logprobs=1,
    max_tokens=1,
    return_prompt_logits=True,
)

outputs = ref_llm.generate([prompt], sampling_params=ref_params)
ref_logits = outputs[0].prompt_logits  # [num_positions, vocab_size]

# Save ref_logits to safetensors, then unload reference model...

# Phase 2: Compute KLD
test_llm = LLM(model="/path/to/quantized_model")

prompt_kld: TokensPrompt = {
    "prompt_token_ids": tokens,
    "target_token_ids": tokens[1:],
    "reference_logits_path": "/path/to/ref_logits.safetensors",
    "reference_logits_key": "logits_0",
}

kld_params = SamplingParams(
    prompt_logprobs=1,
    max_tokens=1,
    kld_mode=True,
)

outputs = test_llm.generate([prompt_kld], sampling_params=kld_params)
kld_sum, kld_count = outputs[0].kld_result
mean_kld = kld_sum / kld_count
```

For a complete implementation that handles both phases, sliding windows, and
reference logits caching, see the example script:

[examples/offline_inference/score_mode_kld.py](../../examples/offline_inference/score_mode_kld.py)

## Constraints

- `score_mode` requires `prompt_logprobs` to be set.
- `kld_mode` and `return_prompt_logits` are mutually exclusive.
- `kld_mode` requires `reference_logits_path` and `reference_logits_key` in
  the prompt.
- Prefix caching should be disabled (`enable_prefix_caching=False`) for
  accurate evaluation results.
