# MiniCPM5-15B-A2.5B XPU monokernel

The MiniCPM5 MoE monokernel is available on supported Intel Battlemage devices
with at least 32 Xe cores at TP1 or TP2. Xe2-enabled `vllm-xpu-kernels` builds
include the extension by default. Install a compatible wheel that contains the
MiniCPM5 extension and pass `--monokernel` to enable it at runtime. Set
`MINICPM5_MONO_KERNELS_ENABLED=OFF` to omit the extension when building from
source.

The FP16 standard-decode tuning targets 1–8 live requests (`M=1–8`) per step,
with one query token per request. This count excludes graph padding. The
serving example selects these decode settings.

## Launch

Set `MODEL_DIR` to a local MiniCPM5 checkpoint directory. Use online per-tensor
FP8 quantization with FP16 or BF16 activations and a matching KV-cache dtype.

```bash
vllm serve "$MODEL_DIR" --monokernel --tensor-parallel-size 2 \
  --dtype float16 --quantization fp8_per_tensor \
  --max-num-seqs 8 --no-enable-prefix-caching \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Use `--tensor-parallel-size 1` for one device or `--enforce-eager` for eager
execution. Weight reload, weight transfer, and sleep mode are unsupported while
the monokernel is active.
