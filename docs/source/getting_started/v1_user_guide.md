# vLLM V1 User Guide

## Why vLLM v1?
Previous blog post [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

## Semantic changes and deprecated features

### Logprobs
- vLLM v1 now supports both sample logprobs and prompt logprobs, as introduced in this [PR](https://github.com/vllm-project/vllm/pull/9880).
- In vLLM v1, logprobs are computed before logits post-processing, so penalty adjustments and temperature scaling are not applied.
- The team is actively working on implementing logprobs that include post-sampling adjustments, incorporating both penalties and temperature scaling.

### Encoder-Decoder
- vLLM v1 is currently limited to decoder-only Transformers. Please check out our [documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for a more detailed list of the supported models. 

## Unsupported features

vLLM v1 does not support the following features yet.

### FP8 KV Cache
- This feature is available in vLLM v0 ant not in v1. With v0, you can enable 
  FP8 KV 
  Cache by specifying:
  ```--kv-cache-dtype fp8```

### CPU Offload
- vLLM v1 does not supports CPU offload. vLLM v0 has the CPU offload 
  implementation in `vllm/worker/model_runner.py` that you can specify with 
  `--cpu-offload-gb 1` (1gb)


## Unsupported models


## FAQ
