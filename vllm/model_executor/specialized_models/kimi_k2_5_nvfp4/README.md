# nvidia/Kimi-K2.5-NVFP4

An optimized implementation for `nvidia/Kimi-K2.5-NVFP4`.

This specialization keeps the Kimi-K2.5 multimodal wrapper, but replaces the
text backbone with a model-specific NVFP4 implementation tuned for the
Blackwell + FlashInfer MLA/MoE path used by this checkpoint.

It intentionally targets the `nvidia/Kimi-K2.5-NVFP4` runtime profile and
falls back to the generic Kimi implementation for other Kimi-K2.5 variants.

## Usage

```bash
VLLM_USE_SPECIALIZED_MODELS=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA \
VLLM_USE_FLASHINFER_MOE_FP4=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
vllm serve nvidia/Kimi-K2.5-NVFP4 \
    --language-model-only \
    --kv-cache-dtype fp8
```
