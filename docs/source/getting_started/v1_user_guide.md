# vLLM V1 User Guide

## Why vLLM v1?
Previous blog post [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

## Semantic changes and deprecated features

### Logprobs
- vLLM v1 now supports both sample logprobs and prompt logprobs, as introduced in this [PR](https://github.com/vllm-project/vllm/pull/9880).
- **Current Limitations**: 
  - v1 prompt logprobs do not support prefix caching.
  - v1 logprobs are computed before logits post-processing, so penalty 
  adjustments and temperature scaling are not applied.
- The team is actively working on implementing logprobs that include post-sampling adjustments.

### Encoder-Decoder
- vLLM v1 is currently limited to decoder-only Transformers. Please check out our 
  [documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for a 
  more detailed list of the supported models. Encoder-decoder models support is not 
  happending soon. 

## Unsupported features


### LoRA
- LoRA works for V1 on the main branch, but its performance is inferior to that of V0.
  The team is actively working on improving the performance [PR](https://github.com/vllm-project/vllm/pull/13096).

### Spec decode other than ngram
- Currently, only ngram spec decode is supported in V1 after this [PR](https://github.com/vllm-project/vllm/pull/12193).

### KV Cache Swapping & Offloading & FP8 KV Cache
- vLLM v1 does not support KV Cache swapping, offloading, and FP8 KV Cache yet. The 
  team is working actively on it.


## Unsupported models


## FAQ
