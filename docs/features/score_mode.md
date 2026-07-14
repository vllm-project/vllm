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

## Determinism

Scoring is only meaningful if the same command produces the same score every
time. Several parts of the compiled stack select kernels by *timing*
candidates, so run-to-run timing noise can change which kernel wins, changing
floating-point reduction order and thus logits (and therefore PPL/KLD)
between otherwise identical runs:

- `combo_kernels` / `benchmark_combo_kernel` (enabled by default in
  `CompilationConfig` on torch >= 2.9)
- Inductor runtime Triton autotuning (`triton.autotune_pointwise`, on by
  default): multiple candidate configs per generated kernel are benchmarked
  at first call in every process
- `max_autotune` / `coordinate_descent_tuning` (only active for static
  `compile_sizes`)
- FlashInfer warmup autotuning (`enable_flashinfer_autotune`, enabled by
  default at optimization level >= 1 on SM90+): kernel tactics picked by
  per-process benchmarking
- Inductor on-device benchmarking for reduction config selection
  (split/persistent reductions), which changes summation order; disabled
  via `TORCHINDUCTOR_DETERMINISTIC=1` (PyTorch's dedicated run-to-run
  determinism mode)

The example scripts disable all of these by default by passing:

```python
llm = LLM(
    model=...,
    compilation_config={
        "inductor_compile_config": {
            "combo_kernels": False,
            "benchmark_combo_kernel": False,
            "triton.autotune_pointwise": False,
            "max_autotune": False,
            "coordinate_descent_tuning": False,
            "benchmark_fusion": False,
        },
    },
    enable_flashinfer_autotune=False,
)
```

with `TORCHINDUCTOR_DETERMINISTIC=1`, `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0`
and `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0`. Kernel selection
then uses fixed heuristics instead of timing; `torch.compile` speed is
otherwise retained. Pass `--no-deterministic` to opt out.

If bit-exact reproducibility is required and the compiled path still shows
run-to-run variation on your stack, `enforce_eager=True` (or
`TORCH_COMPILE_DISABLE=1`) is the guaranteed-deterministic baseline: one
fixed set of kernels, no compilation, no selection of any kind.

!!! note
    The first run after changing any compilation-affecting configuration
    compiles fresh; subsequent runs load from the compile cache. Always
    compare scores between runs that both hit a warm cache (i.e. discard
    the first run after a config change or vLLM rebuild).

Notes:

- Kernel rounding noise is orders of magnitude smaller than quantization
  effects, but KLD is sensitive enough to detect it. Reference logits and
  test runs must be produced under the same kernel configuration; never mix
  references generated with combo kernels enabled and disabled.
- Determinism is guaranteed run-to-run on the same GPU/driver/PyTorch
  version. Different hardware or software versions generate different
  kernels and yield slightly different (internally consistent) baselines.
- For a fully eager fallback, set `TORCH_COMPILE_DISABLE=1` (slower, and a
  slightly different baseline than compiled mode).

## Constraints

- `score_mode` requires `prompt_logprobs` to be set.
- `kld_mode` and `return_prompt_logits` are mutually exclusive.
- `kld_mode` requires `reference_logits_path` and `reference_logits_key` in
  the prompt.
- Prefix caching should be disabled (`enable_prefix_caching=False`) for
  accurate evaluation results.
