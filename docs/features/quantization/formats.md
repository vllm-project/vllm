# Methods and Schemes

This page defines the terms the quantization pages use and lists the names
vLLM accepts for quantization methods and schemes.

The tables are generated at docs build time by
`docs/mkdocs/gen_files/generate_quantization_tables.py` from the registries in
`vllm/config/quantization.py` and `vllm/model_executor/layers/quantization/`
and from the platform allow-lists. A name added to a registry appears here on
the next build. Names registered by
[plugins](README.md#out-of-tree-quantization-plugins) are not listed.

<style>
td:first-child code {
  white-space: nowrap;
}
</style>

## Terms

| Term | Meaning | Where it appears |
| ---- | ------- | ---------------- |
| **Method** | The implementation that loads a quantized model. Most methods read one checkpoint format and are named after the toolkit that writes it, e.g. `compressed-tensors`, `modelopt`, `quark`. | `quant_method` in the model's `config.json`, or `--quantization` |
| **Online quantization** | Quantizing full-precision weights while the model loads. | `--quantization <shorthand>`, `--quantization-config` |
| **Scheme** | The numeric format of one tensor: its data format, its scale format, and what shares a scale. | The `weight` and `activation` fields of `--quantization-config` |
| **Kernel** | The code that runs a layer for a given weight and activation scheme. | `--linear-backend`, `--moe-backend`; see [Kernel Backends](kernels.md) |

A method names how a model is loaded; a scheme names the format of one tensor.
One checkpoint can hold layers with different schemes, and kernels are selected
per layer from the scheme.

`W<n>A<m>` gives the bits of weights and activations: W4A16 is 4-bit weights
with 16-bit activations, and W4A4 quantizes both. Weight-only schemes save
memory; the matrix multiply still runs in 16-bit unless activations are
quantized too.

## Methods

These are the values of `--quantization` other than the
[online shorthands](#online-shorthands). For a pre-quantized checkpoint vLLM
reads `quant_method` from `config.json` and selects the method, so the flag is
normally omitted. A method that does not correspond to a checkpoint format is
used only when it is passed explicitly.

`--quantization compressed-tensors` requires a pre-quantized checkpoint with
`quantization_config` or `compression_config` in its `config.json`.
For an unquantized checkpoint, omit the flag to load the original weights.

--8<-- "gen:quantization-methods"

- Names on one row are aliases of one implementation.
- **Platforms** lists where vLLM accepts the name at startup. ROCm and XPU
  accept only the names on their allow-list; CUDA and CPU accept every name. An
  accepted method can still have no kernel for a given device; see
  [Kernel Backends](kernels.md).
- A deprecated method also needs `--allow-deprecated-quantization`.

## Online shorthands

Passed to `--quantization`, a shorthand quantizes the weights of a checkpoint
that is not quantized. The columns show the weight scheme it applies to linear
layers and to MoE expert layers; `–` leaves that kind of layer unquantized.

--8<-- "gen:quantization-shorthands"

- A shorthand is not a scheme, even when the two share a name: read the
  columns. A cell that is not a name is a scheme that has no name of its own
  and is reachable only through the shorthand.
- Shorthands set weights only. The activation scheme is the default of the
  method that implements the weight scheme, and it can depend on the kernel
  that is selected.
- The same names are accepted inside `--quantization-config`, including on a
  checkpoint that is already quantized. See [Online Quantization](online.md).

## Scheme names

These names are accepted by the `weight` and `activation` fields of
`--quantization-config`:

```bash
vllm serve <model> --quantization online \
    --quantization-config '{"moe":{"weight":"mxfp4","activation":"mxfp8"}}'
```

--8<-- "gen:quantization-schemes"

- **Data** is the element format. E*x*M*y* is a float with *x* exponent and
  *y* mantissa bits: E4M3 is 8-bit and E2M1 is 4-bit, each with a sign bit.
  E8M0 is an unsigned 8-bit exponent, i.e. a power of two.
- **Scale** is the format of the scale factor and what shares one value. A
  second entry is an outer scale applied on top of the first.
- **Scale computed** separates static scales, which are stored in the
  checkpoint or computed once at load, from dynamic ones, which are computed
  from the tensor on every forward pass. Weights use static scales. Dynamic
  scales are how activations are quantized without calibration data.
- **Online weight for** lists the kinds of layer that online quantization can
  quantize to this scheme. `lm_head` marks a scheme that applies only to the
  lm_head, selected with `targets`. Every name parses on both fields, but a
  name with `–` here is rejected or ignored as an online weight, and only some
  methods accept an `activation` override. See [Online Quantization](online.md).

The table shows each scheme as vLLM's `QuantKey` describes it. Formats that are
not a data format plus scales are approximated: `lut_b` stores 3-bit indices
into a codebook, which appears here as the scale, and an E8M0 scale per 32
elements that the key does not describe.

MXFP8 and MXFP4 are the
[OCP Microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
formats. With `b` bits per element and an `s`-bit scale shared by `g` elements,
a weight costs about `b + s / g` bits per parameter: 4.25 for MXFP4 and 4.5 for
NVFP4.

## See also

- [Kernel Backends](kernels.md): which kernel runs a layer, and how to pin one.
- [Online Quantization](online.md): the `--quantization-config` schema,
  per-layer targeting, and re-quantizing a quantized checkpoint.
- [Quantized KV Cache](quantized_kvcache.md): a separate setting from weight
  quantization.
