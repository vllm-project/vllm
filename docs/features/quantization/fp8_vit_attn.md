# FP8 ViT Encoder Attention

For visual understanding workloads with large images (e.g. QHD, 4K) and relatively
short text prompts/generation, the ViT encoder attention can become a significant
bottleneck, especially when the text model is quantized (e.g. NVFP4). vLLM
supports optional FP8 quantization for the ViT encoder attention via the
FlashInfer cuDNN backend. Q/K/V are quantized on-the-fly to FP8 before the
cuDNN attention call.

!!! note
    - Currently supports Qwen3-VL family models only (`qwen3_vl`, `qwen3_vl_moe`,
      `qwen3_5`, `qwen3_5_moe`, and other models using Qwen3 ViT).
    - Dynamic scaling is not compatible with ViT full CUDA graphs.
    - Performance gains are mostly visible at QHD/4K resolutions or multi-image
      requests. Smaller images may see no speedup due to quantization overhead
      (3 quantization kernel launches + un-padding).
    - FP8 tensor-core speedup is more pronounced on GB300 than GB200.

## Requirements

- FlashInfer cuDNN backend with cuDNN >= 9.17.1.

## Usage

Enable FP8 ViT attention by passing `--mm-encoder-attn-dtype fp8` together
with `--mm-encoder-attn-backend FLASHINFER`:

```bash
vllm serve $MODEL \
    --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8
```

By default (no scale file), **dynamic scaling** is used: a 16-entry circular
buffer of observed Q/K/V amax values drives per-forward scale updates. This
matches BF16 accuracy without any calibration but adds a small per-forward
overhead.

## Calibrate-Once, Reuse Workflow (Recommended)

For production, calibrate static scales on a representative dataset once and
reuse them to avoid the dynamic overhead:

```bash
# Step 1: calibrate and save scales (runs dynamic scaling for 16 passes,
# then dumps the learned scales to JSON).
vllm bench mm-processor \
    --model $MODEL --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8 \
    --mm-encoder-fp8-scale-save-path /path/to/scales.json \
    --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
    --num-prompts 100

# Step 2: serve with static scales (no dynamic overhead).
vllm serve $MODEL \
    --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8 \
    --mm-encoder-fp8-scale-path /path/to/scales.json
```

Saved scales are multiplied by `--mm-encoder-fp8-scale-save-margin` (default
`1.5`) to leave headroom against activation outliers not present in the
calibration set. The default has been validated to generalize across datasets
(e.g. VisionArena-Chat calibration maintains BF16 accuracy on ChartQA).

## Scale File Format

```json
{
    "visual.blocks.0.attn.attn": {"q": 224.0, "k": 198.0, "v": 210.0},
    "visual.blocks.1.attn.attn": {"q": 218.0, "k": 195.0, "v": 207.0}
}
```

Keys `q_scale` / `k_scale` / `v_scale` are accepted as aliases.

## Performance

**Core cuDNN attention kernel** (PyTorch profiler, `cudnn_generated_fort_native_sdpa_sm100_flash_fprop`, head_dim=128, seq_len=8192):

| Hardware | BF16 | FP8 | Speedup |
| -------- | ---- | ---- | ------- |
| GB200 | 350 us | 312 us | **1.12x** |
| GB300 | 300 us | 211 us | **1.42x** |

**End-to-end encoder forward time** (Qwen3-VL-30B-A3B-Instruct on GB200, 3 images/request):

| Resolution | BF16 median | FP8 median | Speedup |
| ---------- | ----------- | ---------- | ------- |
| HD (720x1280) | 31.77 ms | 36.39 ms | 0.87x |
| FullHD (1080x1920) | 57.99 ms | 58.73 ms | ~same |
| QHD (1440x2560) | 131.83 ms | 122.30 ms | **1.08x** |
| 4K (2160x3840) | 543.44 ms | 460.31 ms | **1.18x** |

Crossover is around FullHD with 3 images/request. At QHD and above, FP8 wins.

## Accuracy

ChartQA, Qwen3-VL-8B-Instruct, 500 samples. FP8 static uses scales calibrated
on VisionArena-Chat (with default 1.5x margin):

| Metric | BF16 | FP8 dynamic | FP8 static |
| ------ | ---- | ----------- | ---------- |
| relaxed_accuracy | 0.780 | 0.776 | 0.780 |
| anywhere_accuracy | 0.806 | 0.816 | 0.814 |
| exact_match | 0.584 | 0.582 | 0.578 |

All three configurations match within statistical noise, confirming that
static scales calibrated on one dataset generalize to another.
