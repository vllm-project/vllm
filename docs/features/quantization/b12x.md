# b12x Linear and MoE Backends

[b12x](https://pypi.org/project/b12x/) provides optional CUDA kernels for
NVIDIA SM120 and SM121 GPUs. Install the dependency with:

```bash
uv pip install "vllm[b12x]"
```

b12x linear kernels participate in automatic selection after established
optimized backends and before emulation. Select the linear and MoE backends
explicitly with:

```bash
vllm serve <model> \
    --linear-backend b12x \
    --moe-backend b12x
```

Only pass `--moe-backend b12x` for a compatible NVFP4 or MXFP4 MoE model. The
linear and MoE backends can be selected independently.

b12x uses MXFP8 activations by default for MXFP4 MoE and the checkpoint's
activation format for NVFP4 MoE. MXFP4 falls back to BF16 when its A8 path does
not support the model configuration. Set `VLLM_B12X_MOE_FP4_FORCE_A16=1` to
force BF16 activations for either FP4 weight format.

## Supported Configurations

| Backend | Supported configurations |
| ------- | ------------------------ |
| Linear | Per-tensor FP8, 128x128 block FP8, MXFP8, NVFP4, and MXFP4 |
| MoE | Tensor-parallel MXFP4 weights with BF16 or MXFP8 activations; NVFP4 weights with BF16, NVFP4, or MXFP8 activations |

Dense W4A16 layers are not handled by b12x and continue to use another
compatible backend such as Marlin. The b12x MoE backend does not support expert
parallelism, expert maps, EXL3, or NF3.
