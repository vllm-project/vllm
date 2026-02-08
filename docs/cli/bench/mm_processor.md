# vllm bench mm-processor

## Overview

`vllm bench mm-processor` profiles the multimodal input processor pipeline of
vision-language models. It measures per-stage latency from the HuggingFace
processor through to the encoder forward pass, helping you identify
preprocessing bottlenecks and understand how different image resolutions or
item counts affect end-to-end request time.

The benchmark supports two data sources: synthetic random multimodal inputs
(`random-mm`) and HuggingFace datasets (`hf`). Warmup requests are run before
measurement to ensure stable results.

## Quick Start

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 50 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

## Measured Stages

| Stage | Description |
|-------|-------------|
| `hf_processor_time` | Time spent in the HuggingFace processor |
| `hashing_time` | Time spent hashing multimodal inputs |
| `cache_lookup_time` | Time spent looking up the processor cache |
| `prompt_update_time` | Time spent updating prompt tokens |
| `preprocessor_total_time` | Total preprocessing time |
| `encoder_forward_time` | Time spent in the encoder model forward pass |

The benchmark also reports `num_encoder_calls` per request and end-to-end
latency (TTFT + decode time). Use `--metric-percentiles` to select which
percentiles to report (default: p99) and `--output-json` to save results.

For more examples (HF datasets, warmup, JSON output), see
[Benchmarking CLI — Multimodal Processor Benchmark](../../benchmarking/cli.md#multimodal-processor-benchmark).

## JSON CLI Arguments

--8<-- "docs/cli/json_tip.inc.md"

## Arguments

--8<-- "docs/generated/argparse/bench_mm_processor.inc.md"
