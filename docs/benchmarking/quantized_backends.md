# Compare Quantized Linear Backends

## Summary

This example compares Marlin and Humming on one A100 with a GPTQ INT4 model.

- Humming had 2.8%–15.2% higher output throughput.
- Both scored 1,138/1,319 on GSM8K, but 11 numerical answers differed.
- First startup took 79s with Humming and 60s with Marlin. Warm starts took 32–34s.

## Results

Output tokens/s, median of three runs per backend. Each case used 96 random
prompts and 128 output tokens. All 3,456 measured requests succeeded.

| Input tokens | Concurrency | Marlin | Humming | Change |
| --- | --- | --- | --- | --- |
| 128 | 1 | 209.4 | 215.4 | +2.8% |
| 128 | 8 | 1468.1 | 1511.0 | +2.9% |
| 128 | 32 | 3753.0 | 3878.9 | +3.4% |
| 2048 | 1 | 168.2 | 176.4 | +4.9% |
| 2048 | 8 | 558.5 | 621.5 | +11.3% |
| 2048 | 32 | 723.1 | 833.3 | +15.2% |

## Test conditions

- A100 SXM 80GB, 500W; clocks were not locked.
- [Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/tree/e9c932ac1893a49ae0fc497ad6e1e86e2e39af20), symmetric INT4 g128, no act-order, FP16, TP=1.
- vLLM [5225a1f17](https://github.com/vllm-project/vllm/commit/5225a1f17f9cf721efd5ad793e6b63798ca30306), Humming 0.1.18, PyTorch 2.14.0+cu130.
- Same checkpoint, FlashAttention 2, CUDA Graph mode, 8 GiB KV cache, and batch limits. Prefix cache was off.
- GSM8K: five-shot, temperature 0, seed 42, 512-token limit; one run per backend.

## Limits

These results cover one checkpoint and GPU configuration. They do not justify
a general default change. First-start times exclude model downloads: compiler
caches were fresh, but OS and shared FlashInfer caches were warm.

See the [reproduction steps](quantized_backends_reproduce.md) for commands and
checks, or the [original scripts and data](https://gist.github.com/jiangLLM/b77407a91899553309bfe941ed621cf6)
for the exact run.
