# Compressed-tensors LUT-B

vLLM can load compressed-tensors LUT-B linear weights and run them with MXFP8
activation quantize-dequantize emulation. This is useful for checking a LUT-B
checkpoint on hardware without the native LUT-B linear kernel, such as an H100.

The loader expects 3-bit symmetric codebook weights with `block_structure: [8,
64]`, E8M0 (`torch.uint8`) scales, and no activation quantization in the
checkpoint metadata. Input dimensions must be divisible by 64, and each output
partition must be divisible by 8.

This support requires a compressed-tensors version that includes LUT-B's
`type: codebook` schema. vLLM must pin that version when the support lands.

```bash
vllm serve <compressed-tensors-lut-b-checkpoint> \
  --linear-backend emulation
```

During model loading, vLLM expands LUT-B weights to BF16. At inference, it
rounds activations through MXFP8 before the linear operation. The emulation
path preserves LUT-B numerics for validation, but it does not preserve the
checkpoint's compressed memory footprint or provide native-kernel throughput.
