# vLLM V1 User Guide

## Why vLLM V1?
Previous blog post [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)


## Semantic Changes and Deprecated Features
### Logprobs
- vLLM V1 now supports both sample logprobs and prompt logprobs. Currently, the [implementation](https://github.com/vllm-project/vllm/pull/9880) has the following **limitations and semantic changes**:
  - V1 prompt logprobs do not support prefix caching.
  - V1 logprobs are computed before logits post-processing, so penalty
  adjustments and temperature scaling are not applied.
- The team is actively working on implementing logprobs that include post-sampling adjustments.

### Deprecated Features:

#### Current deprecated sampling features
The team is working on supporting these features globally in the server.

- best_of
- per request logits processors

#### Deprecated KV Cache features
- KV Cache swapping
- KV Cache offloading

## Unsupported features

- **LoRA**: LoRA works for V1 on the main branch, but its performance is inferior to that
  of V0.
  The team is actively working on improving the performance [PR](https://github.com/vllm-project/vllm/pull/13096).

- **Spec Decode other than ngram**: currently, only ngram spec decode is supported in V1
  after this [PR](https://github.com/vllm-project/vllm/pull/12193).

- **Quantization**: For V1, when the CUDA graph is enabled, it defaults to the
  piecewiseCUDA graphintroduced in this[PR](https://github.com/vllm-project/vllm/pull/10058); consequently,FP8 and other quantizations are not supported.

- **FP8 KV Cache**: FP8 KV Cache is not yet supported in V1.

- **Structured Generation Fallback**: Only `xgrammar:no_fallback` is supported.
  Details about the structured generation can be found [here](https://docs.vllm.ai/en/latest/features/structured_outputs.html).



## Unsupported Models

vLLM V1 excludes models tagged with `SupportsV0Only` while we develop support for
other types. For a complete list of supported models, see our [documentation](https://docs.vllm.ai/en/latest/models/supported_models.html). The
following categories are currently unsupported, but we plan to
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

For a complete list of supported models, see
[our documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).

## FAQ
