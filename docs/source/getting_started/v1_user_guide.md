# vLLM V1 User Guide

## Why vLLM V1?

vLLM V0 successfully supported a wide range of models and hardware, but as new features were developed independently, the system grew increasingly complex. This complexity made it harder to integrate new capabilities and introduced technical debt, revealing the need for a more streamlined and unified design.

Building on V0’s success, vLLM V1 retains the stable and proven components from V0
(such as the models, GPU kernels, and utilities). At the same time, it significantly
re-architects the core systems—covering the scheduler, KV cache manager, worker,
sampler, and API server—to provide a cohesive, maintainable framework that better
accommodates continued growth and innovation.

Specifically, V1 aims to:

- Provide a **simple, modular, and easy-to-hack codebase**.
- Ensure **high performance** with near-zero CPU overhead.
- **Combine key optimizations** into a unified architecture.
- Require **zero configs** by enabling features/optimizations by default.

For more detailed please refer to the vLLM V1 blog post [vLLM V1: A Major
Upgrade to vLLM’s Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (published Jan 27, 2025)

## Semantic Changes and Deprecated Features

### Logprobs

vLLM V1 introduces support for returning log probabilities (logprobs) for both
sampled tokens and the prompt.
However, there are some important semantic
differences compared to V0:

**Prompt Logprobs Without Prefix Caching**

In vLLM V1, if you request prompt logprobs (using the `prompt_logprobs=true` flag),
prefix caching is not available. This means that if you want logprobs for the prompt,
you must disable prefix caching (e.g. by starting the server with `--no-enable-prefix-caching`).
The team is working to support prompt logprobs with caching.

**Pre-Post-Processing Calculation**

Logprobs in V1 are now computed immediately
after the model’s raw output (i.e.
before applying any logits post-processing such as temperature scaling or penalty
adjustments). As a result, the returned logprobs do not reflect the final adjusted
probabilities that might be used during sampling.

In other words, if your sampling pipeline applies penalties or scaling, those
adjustments will affect token selection but not be visible in the logprobs output.

The team is working in progress to include these post-sampling
adjustments in future updates.

### Deprecated Features

As part of the major architectural rework in vLLM V1, several legacy features have been removed to simplify the codebase and improve efficiency.

**Deprecated sampling features**

- **best_of**: The sampling parameter best_of—which in V0 enabled
  generating multiple candidate outputs per request and then selecting the best
  one—has been deprecated in V1.
- **Per-Request Logits Processors**: In V0, users could pass custom
  processing functions to adjust logits on a per-request basis. In vLLM V1 this
  mechanism is deprecated. Instead, the design is moving toward supporting global
  logits processors—a feature the team is actively working on for future releases.

**Deprecated KV Cache features**

- KV Cache Swapping
- KV Cache Offloading

## Unsupported or Unoptimized Features in vLLM V1

vLLM V1 is a major rewrite designed for improved throughput, architectural
simplicity, and enhanced distributed inference. Although many features have been
re‐implemented or optimized compared to earlier versions, some functionalities
remain either unsupported or not yet fully optimized:

### Unoptimized Features

- **LoRA**: LoRA works for V1 on the main branch, but its performance is
  inferior to that of V0. The team is actively working on improving its
  performance
(e.g., see [PR #13096](https://github.com/vllm-project/vllm/pull/13096)).

- **Spec Decode**: Currently, only ngram-based spec decode is supported in V1. There
  will be follow-up work to support other types of spec decode.

### Unsupported Features

- **FP8 KV Cache**: While vLLM V1 introduces new FP8 kernels for model weight quantization, support for an FP8 key–value cache is not yet available. Users must continue using FP16 (or other supported precisions) for the KV cache.

- **Structured Generation Fallback**: For structured output tasks, V1 currently
  supports only the `xgrammar:no_fallback` mode.
  Details about the structured generation can be found
  [here](https://docs.vllm.ai/en/latest/features/structured_outputs.html).

## Unsupported Models

vLLM V1 excludes models tagged with `SupportsV0Only` while we develop support for
other types. The following categories are currently unsupported, but we plan to
support them eventually.

**Embedding Models**
- vLLM V1 does not yet include a `PoolingModelRunner` to support embedding/pooling
  models.  
  *Example*: `BAAI/bge-m3`

**Mamba Models**  
- Models using selective state-space mechanisms (instead of standard transformer attention) are not yet supported.  
  *Examples*:  
  - Pure Mamba models (e.g., `BAAI/mamba-large`)  
  - Hybrid Mamba-Transformer models (e.g., `ibm-ai-platform/Bamba-9B`)

**Encoder-Decoder Models**  
- vLLM V1 is currently optimized for decoder-only transformers. Models requiring
  cross-attention between separate encoder and decoder (e.g.,
  `facebook/bart-large-cnn`) are not yet supported.

For a complete list of supported models, see the [list of supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).
