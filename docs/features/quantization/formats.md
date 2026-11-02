# Formats and Vocabulary

Quantization pages use four easily confused terms:

| Term | Meaning | Where it appears |
| ---- | ------- | ---------------- |
| **Checkpoint method** | The format (usually the toolkit) that produced the file, e.g. `compressed-tensors`, `modelopt`, `quark`. | `quant_method` in `config.json`; auto-detected |
| **Scheme** | The precision recipe, e.g. `fp8_per_tensor`, `mxfp4`, or a W4A16 layout. | `--quantization`, `--quantization-config` |
| **Granularity** | What shares one scale: tensor, channel, token, block, or group. | Inside a scheme |
| **Quant key** | vLLM's internal descriptor of a scheme: data type, scale type, group shape, symmetry. | Kernel dispatch and the plugin API |

A checkpoint method is a **container**; a scheme is a **recipe**. One
`compressed-tensors` checkpoint can hold two schemes, e.g. FP8 linear layers and NVFP4
expert layers. Kernels see schemes, not containers.

## W8A8, W4A16, W4A8, W4A4

The `W<n>A<m>` notation gives the stored bits for **w**eights and **a**ctivations:

| Notation | Weights | Activations | What it buys | Typical formats |
| -------- | ------- | ----------- | ------------ | --------------- |
| W8A8 | 8-bit | 8-bit | 2× smaller weights, 8-bit tensor-core GEMM | FP8, INT8, MXFP8 |
| W4A16 | 4-bit | 16-bit | 4× smaller weights, dequantize inside the kernel | GPTQ/AWQ, MXFP4, NVFP4 W4A16 |
| W4A8 | 4-bit | 8-bit | 4× smaller weights plus a faster GEMM where supported | INT4 W4A8, MXFP4 weights with MXFP8 activations |
| W4A4 | 4-bit | 4-bit | 4× smaller weights, FP4 tensor cores | NVFP4, MXFP4 |

Activation quantization turns a memory optimization into a compute optimization.
Weight-only schemes help memory-bound decode but usually do not make the GEMM itself
faster.

`W4A16` describes storage, not a single checkpoint format: GPTQ and AWQ (INT4 groups
with zero-points) and MXFP4/NVFP4 weight-only variants all qualify, and their kernels
are not interchangeable.

## Scaling granularity

| Granularity | Scales | Used by | Consequence |
| ----------- | ------ | ------- | ----------- |
| Per-tensor | 1 per tensor | FP8 per-tensor, INT8 | Cheapest to apply; one outlier compresses the whole tensor |
| Per-channel | 1 per output channel | FP8 per-channel, INT8 per-channel | Good default for weights |
| Per-token | 1 per activation row | FP8 per-token, MXFP8 | Required for dynamic activation scaling |
| Per-block 128×128 (weights), 1×128 (activations) | ~1 per 16k weights | FP8 block, DeepSeek-style | Robust to outliers; more scale traffic |
| Per-group 32 | 1 per 32 weights | MXFP4, MXFP8 | OCP microscaling standard |
| Per-group 16 plus global | 1 per 16 weights, 1 fp32 per tensor | NVFP4 | Finer than MX; two-level scaling |

"Block" and "group" both mean a rectangular tile that shares a scale. OCP MX fixes the
group at 32 elements; NVFP4 fixes it at 16.

## Numeric formats

- **FP8 E4M3** — 8-bit float with 4 exponent and 3 mantissa bits. The default FP8
  storage type. AMD gfx94x uses the `E4M3FNUZ` variant; vLLM converts on load, so the
  checkpoint format is unchanged.
- **FP8 E5M2** — wider range, fewer mantissa bits. Used for KV caches, not weights.
- **MXFP8 / MXFP4** — OCP microscaling: E4M3 or E2M1 data with an **E8M0 scale per 32
  elements**.
- **NVFP4** — E2M1 data with an **E4M3 scale per 16 elements** and a second-level
  **fp32 per-tensor scale**. Group size, scale type, and kernels all differ from MXFP4.
- **INT8 / INT4** — uniform integers, with zero-points in GPTQ/AWQ-style formats and
  symmetric scales in most FP8-style INT8 recipes.

## Weight and activation schemes

`quantization_config` accepts these names on the `weight` and `activation` fields of
`linear` and `moe`. Each is a quant key; the table is generated from `QUANT_KEY_NAMES`
in [`vllm/config/quantization.py`](../../../vllm/config/quantization.py).

--8<-- "gen:quantization-schemes"

Seven shorthands bundle one weight and one activation choice. Three of them (`mxfp8`,
`mxfp4`, `nvfp4_per_token`) are also quant key names; the rest are shorthands only.

| Shorthand | Weight recipe | Activation recipe | Applies to |
| --------- | ------------- | ----------------- | ---------- |
| `fp8_per_tensor` | FP8, one scale per weight tensor | FP8 per-token where supported, otherwise per-tensor | linear, MoE |
| `fp8_per_block` | FP8, 128×128 weight blocks | FP8, 1×128 activation blocks | linear, MoE |
| `fp8_per_channel` | FP8, one scale per output channel | FP8 per-token | linear, MoE |
| `mxfp8` | MXFP8 | MXFP8 | linear, MoE |
| `mxfp4` | MXFP4 | MXFP4 on MoE and most linear backends, 16-bit on some | linear, MoE |
| `int8_per_channel_weight_only` | INT8, one scale per output channel | dynamic per-token INT8 | MoE only |
| `nvfp4_per_token` | NVFP4 | NVFP4 per-token | MoE only |

`mxfp4` and `mxfp8` are also checkpoint method names, so `--quantization mxfp8`
resolves to the checkpoint method when the model has quantized weights and to the
online shorthand when it does not. See
[online quantization](online.md) for the configuration schema, per-layer targeting,
and shorthand defaults.

## Bits per parameter

With `b` bits per weight and an `s`-bit scale shared by `g` weights, a scheme costs
approximately `b + s / g` bits per parameter:

| Scheme | Bits per parameter | Relative to BF16 |
| ------ | ------------------ | ---------------- |
| BF16 / FP16 | 16 | 1× |
| FP8 per-tensor | ~8 | 2× smaller |
| MXFP8 (E8M0 per 32) | 8.25 | ~1.9× smaller |
| INT8 W8A8 | ~8 | 2× smaller |
| INT4 group-128 (fp16 scale) | ~4.1 | ~3.9× smaller |
| MXFP4 (E8M0 per 32) | 4.25 | ~3.8× smaller |
| NVFP4 (E4M3 per 16, fp32 per tensor) | ~4.5 | ~3.6× smaller |

These numbers cover weights only. Activations, CUDA graphs, and workspace are extra,
and the KV cache is a separate budget; see [quantized KV cache](quantized_kvcache.md).

## See also

- [Online quantization](online.md) — quantize at load time, per-layer targeting.
- [LLM Compressor](llm_compressor/README.md) — produce calibrated checkpoints.
- [NVIDIA ModelOpt](modelopt.md) and [AMD Quark](quark.md) — vendor checkpoint formats.
- [Fused MoE modular kernels](../../design/fused_moe_modular_kernel.md) — how a
  scheme reaches MoE kernels.
