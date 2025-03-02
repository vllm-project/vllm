# vLLM V1 User Guide

## Why vLLM V1?
Previous blog post [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/V1-alpha-release.html)

## Semantic changes and deprecated features

### Logprobs
- vLLM V1 now supports both sample logprobs and prompt logprobs, as introduced in this [PR](https://github.com/vllm-project/vllm/pull/9880).
- **Current Limitations**:
  - V1 prompt logprobs do not support prefix caching.
  - V1 logprobs are computed before logits post-processing, so penalty
  adjustments and temperature scaling are not applied.
- The team is actively working on implementing logprobs that include post-sampling adjustments.

### The following features has been deprecated in V1:

#### Deprecated sampling features
- best_of
- logits_processors
- beam_search

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



## Unsupported Models

## Unsupported Models

vLLM V1 excludes models tagged with `SupportsV0Only` to focus on high-throughput,
decoder-only inference. The unsupported categories are:

- **Embedding Models**  
  vLLM V1 does not yet include a `PoolingModelRunner`.  
  *Example*: `BAAI/bge-m3`

- **Mamba Models**  
  Models using selective state space mechanisms (instead of  standard transformer
  attention) are not supported.  
  *Examples*:  
    - Pure Mamba models (e.g. `BAAI/mamba-large`)  
    - Hybrid Mamba-Transformer models (e.g. `ibm-ai-platform/Bamba-9B`)

- **Encoder-Decoder Models**  
  vLLM V1 is optimized for decoder-only transformers. Models that require
  cross-attention between separate encoder and decoder components are not supported.  
  *Example*: `facebook/bart-large-cnn`

For a complete list of supported models and additional details, please refer to our
[documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).
Support for encoder-decoder architectures is not planned in the near future.

## FAQ
