# Kernel Backends

A quantized layer runs on a kernel chosen at load time from its weight and
activation scheme, the platform, and the device. This page covers quantized
linear layers: the backends that provide kernels and how to select one. MoE
expert layers are selected separately with `--moe-backend`; see
[Fused MoE Kernel Features](../../design/moe_kernel_features.md).

The tables are generated at docs build time by
`docs/mkdocs/gen_files/generate_quantization_tables.py` from the kernel
registries and `_LINEAR_BACKEND_KERNEL_MAP` in
`vllm/model_executor/kernels/linear/__init__.py`. A kernel added to a registry
appears here on the next build.

<style>
td:first-child code {
  white-space: nowrap;
}
</style>

## Linear backends

These are the values of `--linear-backend`, with the platforms and quantization
types each has a kernel for:

--8<-- "gen:linear-backends"

- A quantization type is named for the weight format. **FP8 block** is FP8 with
  block-wise scales, **FP8 W8A16** is FP8 weights with 16-bit activations, and
  **WNA16** is integer weights such as GPTQ, AWQ, and compressed-tensors W4A16
  and W4A8.
- A backend listed for a platform and type can still be unavailable on a
  device: a kernel needs its library installed and can require a GPU
  generation, a layer shape, or a scale layout.

## How a backend is chosen

By default (`--linear-backend auto`) vLLM tries the kernels registered for the
platform and quantization type in priority order and uses the first one that
supports the device and can implement the layer. The selected kernel class is
logged at startup for most quantization types, and that log is the
authoritative answer for a deployment.

A layer that requests quantized activations can be given a kernel that runs
16-bit activations when no other kernel is available.
`VLLM_STRICT_QUANT_SCHEME=1` forbids this; see
[Activation Quantization Fallbacks](README.md#activation-quantization-fallbacks).

## Selecting a backend

`--linear-backend` sets the backend for every quantization type:

```bash
vllm serve <model> --linear-backend marlin
```

`linear_backend_per_quant` in `--kernel-config` sets the backend for one
quantization type and takes precedence over `--linear-backend`. Other types
keep their setting:

```bash
vllm serve <model> \
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"humming"}}'
```

Its keys are:

--8<-- "gen:linear-families"

`w8a8`, `w8a16`, `w4a4`, and `w4a16` in a key give the weight and activation
bits the layer requests. NVFP4 weight-only layers (`nvfp4_w4a16`) use only the
kernels that run 16-bit activations, so a backend listed for NVFP4 can have no
kernel for that key.

With a backend set, a quantization type is restricted to that backend's
kernels:

- If the backend has no kernel for the type on the platform, the type keeps
  automatic selection and a warning is logged.
- If the backend has a kernel but it cannot run on the device or implement the
  layer, startup fails.
