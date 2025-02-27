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
  piecewise CUDA graph introduced in this [PR](https://github.com/vllm-project/vllm/pull/10058) ; consequently, FP8 and other quantizations are not supported. 

- **FP8 KV Cache**: FP8 KV Cache is not yet supported in V1.

## Unsupported models

All model with `SupportsV0Only` tag in the model definition is not supported by V1. 

- **Pooling Models**: Pooling models are not supported in V1 yet.
- **Encoder-Decoder**: vLLM V1 is currently limited to decoder-only Transformers. 
  Please check out our 
  [documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for a 
  more detailed list of the supported models. Encoder-decoder models support is not 
  happending soon. 


## FAQ
