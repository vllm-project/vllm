# [Experimental] Specialized Models

This directory contains experimental, hand-tuned implementations for a small number of selected models. Each subdirectory targets a specific combination of model architecture (including all tensor shapes), quantization scheme, attention backend, and hardware.

For example, `deepseek_v3_2_nvfp4/` targets `nvidia/DeepSeek-V3.2-NVFP4` with FP8 FlashInfer sparse MLA on Blackwell GPUs.

**To opt in, set `VLLM_USE_SPECIALIZED_MODELS=1`.** When enabled, vLLM will prefer a specialized implementation over the generic one if a match is available.

## Development Philosophy

These implementations prioritize iteration speed and checkpoint-specific performance over broad reuse. They may target a very narrow use case and are not expected to cover the full vLLM feature surface. Known limitations include:

- Parallelism strategy support may be incomplete (e.g. TP only, no EP, or vice versa).
- `torch.compile` compatibility may be limited or untested.
- Behavior with checkpoint formats outside the intended target is unsupported.

Also, code duplication across implementations is intentional — each model should be free to evolve and be optimized independently without risk of regressing another.

Code here is experimental and may be short-lived. Generic features and anything intended for long-term support should live in `../models/`.
