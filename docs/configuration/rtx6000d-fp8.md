# RTX 6000D B12X block-FP8 tail projection

This opt-in path targets NVIDIA RTX 6000D, compute capability 12.0, 156 SMs.

`KernelConfig.enable_rtx6000d_fp8_tail` allows B12X to implement the BF16-input,
128-by-128 block-scaled FP8 weight shape `(3392, 4096)` on this RTX 6000D
SKU. It pads the output dimension to 3456, restores the logical width before
adding bias, and retains the logical parameter and loader shape. Prepared
weight and scale storage is reused during reload for CUDA graph replay.
Other shapes and devices retain their existing eligibility checks.

```bash
vllm serve MODEL --linear-backend b12x \
    --kernel-config '{"enable_rtx6000d_fp8_tail": true}'
```

The option changes eligibility, not backend priority. B12X must be installed
and selected. Unsupported layers retain the existing selector fallback.
This is a backend integration of padding, not a new FP8 arithmetic kernel.
