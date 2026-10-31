# Packed FP8 KV Cache with Triton MLA

The `TRITON_MLA` attention backend supports dense MLA decode using
`kv_cache_dtype="fp8_ds_mla"` on CUDA GPUs with compute capability 8.0 or
newer, including the NVIDIA A100.

## Supported configuration

Select `TRITON_MLA` in the attention configuration, set the KV cache dtype to
`fp8_ds_mla`, and use a `bfloat16` model dtype. This path supports MLA models
with latent rank 512 and RoPE dimension 64.

This configuration is opt-in. The default KV cache dtype is `auto`.

```python
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite-Chat",
    dtype="bfloat16",
    kv_cache_dtype="fp8_ds_mla",
    attention_config={"backend": "TRITON_MLA"},
)
```

The packed format quantizes the latent cache to E4M3 FP8 with one FP32 scale
per group of 128 values. RoPE values remain in BF16. Queries and attention
computation use BF16; an A100 does not need native FP8 tensor cores for this
configuration.

`fp8_ds_mla` is a distinct cache format from `fp8` and `fp8_e4m3`. Those
formats continue to require compute capability 8.9 or newer with the Triton
MLA backend.

## Memory and performance

Each packed token uses 656 bytes of KV storage, compared with 1152 bytes for
the corresponding BF16 MLA cache. This reduces raw KV storage by about 43%.
Model weights and temporary buffers also use GPU memory, so the increase in
available context length depends on the model and serving configuration.

FP8 cache conversion adds work during attention. This configuration can be
slower than BF16 decode; benchmark your workload when choosing between memory
capacity and latency. See [Quantized KV Cache](quantized_kvcache.md) for other
cache quantization options.
