# Kernel Selection

A quantized layer runs on a kernel chosen at load time from its weight and
activation scheme, the platform, and the device. This page covers quantized
linear layers: which kernels exist, the order they are tried in, and how to
override the choice. MoE expert layers are selected separately with
`--moe-backend`; see [Fused MoE Kernel Features](../../design/moe_kernel_features.md).

The tables are generated at docs build time by
`docs/mkdocs/gen_files/generate_quantization_tables.py` from the kernel
registries and `_LINEAR_BACKEND_KERNEL_MAP` in
`vllm/model_executor/kernels/linear/__init__.py`. A kernel added to a registry
appears here on the next build.

## Layer types

Linear kernels are grouped by the scheme of the layer they run. Each group has
its own priority list and its own key for
[per-layer-type overrides](#overriding-the-choice):

--8<-- "gen:linear-families"

`w8a8`, `w8a16`, `w4a4`, and `w4a16` in a key give the weight and activation
bits the layer requests. `mixed_precision` covers integer-weight layers such
as GPTQ, AWQ, and compressed-tensors W4A16 and W4A8.

## How a linear kernel is chosen

For each layer type vLLM tries the kernels registered for the platform in
priority order and uses the first one that supports the device and can
implement the layer. The selected kernel class is logged at startup for most
layer types, and that log is the authoritative answer for a deployment.

The [priority lists](#linear-kernels-by-layer-type) show the registered order.
The selected kernel is often not the first entry:

- A kernel is skipped when its library is not installed, when it does not
  support the GPU, or when it cannot handle the layer's shape or scales.
- Selection code reorders or bypasses the list for some devices and layer
  types. For example, Humming moves ahead of Marlin on compute capability 9.0,
  and NVFP4 weight-only layers (`nvfp4_w4a16`) consider only the kernels that
  run 16-bit activations.
- `VLLM_BATCH_INVARIANT=1` changes selection for some layer types and can
  override `--linear-backend`.
- A list can contain kernels that run 16-bit activations for a layer that
  requests quantized ones. `VLLM_STRICT_QUANT_SCHEME=1` skips kernels that
  would drop or change the requested activation quantization; see
  [Activation Quantization Fallbacks](README.md#activation-quantization-fallbacks).

## Overriding the choice

`linear_backend_per_quant` in `--kernel-config` sets the backend for one layer
type, using the keys from the [layer types](#layer-types) table, which are the
keys read by the selectors on this page. Other layer types keep automatic
selection:

```bash
vllm serve <model> \
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"humming"}}'
```

`--linear-backend` sets the backend for every layer type.
`linear_backend_per_quant` takes precedence over it.

```bash
vllm serve <model> --linear-backend marlin
```

With a backend set, a layer type is restricted to that backend's kernels:

- If the backend has no kernel for the layer type on the platform, the layer
  type keeps automatic selection and a warning is logged.
- If the backend has a kernel but it cannot run on the device or implement the
  layer, startup fails.

The table shows the layer types for which each backend has a kernel in a
priority list. A key can have a narrower set of candidates than its layer type:
`nvfp4_w4a16` considers only the weight-only kernels, so a backend listed for
NVFP4 can still have no kernel for it.

--8<-- "gen:linear-backends"

## Linear kernels by layer type

Kernel names are the classes printed in the startup log. **Backend** is the
`--linear-backend` value that restricts selection to the kernel; a kernel
without one cannot be pinned.

--8<-- "gen:linear-kernel-priority"
