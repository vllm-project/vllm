# Online Quantization

Online quantization lets you take a BF16/FP16 model and quantize its Linear
and MoE weights to lower precision (such as FP8) at load time, without needing
a pre-quantized checkpoint or calibration data. Weights are converted during
model loading and activations are dynamically scaled during each forward pass.

## Quick Start

Pass a scheme name to the `quantization` parameter:

```python
from vllm import LLM

# Per-tensor FP8 quantization (one scale per weight tensor)
llm = LLM("meta-llama/Llama-3.1-8B", quantization="fp8_per_tensor")

# Per-block FP8 quantization (128x128 block scaling for weights and 1x128 block scaling for activations)
llm = LLM("meta-llama/Llama-3.1-8B", quantization="fp8_per_block")

# MXFP8 quantization for weights and activations
llm = LLM("meta-llama/Llama-3.1-8B", quantization="mxfp8")

# MXFP4 weight; activation quantization depends on the `linear_backend` picked
llm = LLM("meta-llama/Llama-3.1-8B", quantization="mxfp4")

# MXFP4 MOE-only weight and activation quantization
llm = LLM(
    "Qwen/Qwen3.5-35B-A3B",
    quantization="mxfp4",
    quantization_config={"linear": {"activation": None, "weight": None}}
)
```

Or with the CLI:

```bash
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_tensor
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_block
vllm serve meta-llama/Llama-3.1-8B --quantization mxfp8
vllm serve meta-llama/Llama-3.1-8B --quantization mxfp4

vllm serve Qwen/Qwen3.5-35B-A3B --quantization mxfp4 \
    --quantization-config '{"linear":{"activation":null,"weight":null}}'
```

## Supported Schemes

| Scheme | Weight recipe | Activation recipe | Notes |
| ------ | ------------- | ------------------ | ----- |
| `fp8_per_tensor` | fp8_e4m3 data, fp32 per-tensor scale | fp8_e4m3 data, fp32 per-tensor scale | On some GPUs (Ada, Hopper) linear activations use per-token scaling for better performance |
| `fp8_per_block` | fp8_e4m3 data, fp32 per-128x128-block scale | fp8_e4m3 data, fp32 per-1x128-block scale | |
| `mxfp8` | fp8_e4m3 data, e8m0 per-1x32-block scale | fp8_e4m3 data, e8m0 per-1x32-block scale | Requires SM 100+ (Blackwell or newer) for w8a8, other GPUs use a w8a16 fallback |
| `mxfp4` | fp4_e2m1 data, e8m0 per-1x32-block scale ([OCP MX specs](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)) | - linear: fp4_e2m1 data, e8m0 per-1x32-block scale in some backends, or BF16. <br> - MOE: fp4_e2m1 data, e8m0 per-1x32-block scale. | Linear MXFP4 backend is auto-selected per platform, not enforcing activation dtype. Some use BF16 activation. Use `--linear-backend` to pin one (e.g. `--linear-backend flashinfer`). |
| `mxfp4_hadamard32` | `mxfp4`, with the weight rotated along its input dimension by a 32-wide block Hadamard matrix before quantization | `mxfp4`, with activations rotated by the same matrix before quantization | Linear only. The rotation is orthogonal, so the GEMM is unchanged up to quantization error, which it shrinks by spreading outliers across each 32-element block. The activation rotation runs in PyTorch before the selected MXFP4 linear backend. |

## Advanced Configuration

For fine-grained control, use a `quantization_config` dictionary.

### Schema

```yaml
quantization_config:
  linear:
    weight: <name>      # see QUANT_KEY_NAMES in vllm/config/quantization.py
    activation: <name>
    hadamard_size: 32   # optional power of two, so far mxfp4 only: see `mxfp4_hadamard32`
    phase: {decode: <format>, prefill: <format>, max_decode_tokens: <rows>, scale: <scale>}
  moe:
    weight: <name>
    activation: <name>
  ignore: [<layer-name-or-regex-or-fnmatch-pattern>, ...]
```

`linear` and `moe` accept a full `{weight, activation}` dict, or a bare
string. A string resolves first against the `--quantization` shorthands
(taking the matching layer-kind slot), then against `QUANT_KEY_NAMES` as a
weight name. Unset fields fall back to the `--quantization` shorthand's
defaults, or for already-quantized checkpoints to whatever the checkpoint
declares. An explicitly configured `activation: null` is not the same as an
omitted activation: it requests unquantized (BF16) activations instead of the
method's default.

On XPU, non-block FP8 scaled-mm linear layers default to W8A16; setting `--linear-backend xpu` forces W8A8. Use `--linear-backend xpu_woq` to explicitly select weight-only quantization (W8A16). Setting `--linear-backend torch` also forces W8A8 but runs the GEMM through `torch._scaled_mm` instead of the custom XPU kernel.

The CLI accepts the same shape as JSON or as dotted keys:

```bash
vllm serve <model> --quantization-config '{"moe":{"activation":"mxfp8"}}'
vllm serve <model> --quantization-config.moe.activation mxfp8
```

### Activation overrides

`quantization_config` lets you select an activation format independently of
the weight format. For online quantization, it can be used to override the
default activation quantization of MXFP4 layers; other online weight formats
reject an activation override.

For example, to enable MXFP4 MoE weights along MXFP8 activations:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B --quantization online \
    --quantization-config '{"moe":{"weight":"mxfp4","activation":"mxfp8"}}'
```

For pre-quantized models, the supported activation overrides are specific to each
weight quantization type; today this is for example available for MXFP4
MoE checkpoints where you can opt into e.g. FP8 activations:

```bash
vllm serve openai/gpt-oss-20b --quantization-config.moe.activation mxfp8
```

Combine with `--moe-backend` to pin a specific kernel family.

For MXFP4 linear layers on RDNA3/RDNA3.5 (gfx11), `int8_per_group_32` selects
the W4A8 int8-dot GEMV for decode (batches of up to 8 tokens); larger batches
run weight-only. On W4A4 checkpoints this replaces the MXFP4 activation
quantization:

```bash
vllm serve <mxfp4-model> --quantization-config.linear.activation int8_per_group_32
```

### Separate decode and prefill activations

`linear.phase` sets the activations of small linear calls (decode) and large
ones (prefill) separately, served from one weight copy. A call with at most
`max_decode_tokens` input rows uses `decode`. Rows count every token in the
call, so decode tokens scheduled with a long prefill use the `prefill` format,
and under CUDA graphs the padded capture size picks the phase: graphs captured
above `max_decode_tokens` run the `prefill` format. Each phase is `unquantized`
(the model dtype) or the quantized format of the weights:

| Weights | Linear backend | `decode` | `prefill` | `scale` |
| ------- | -------------- | -------- | --------- | ------- |
| NVFP4 W4A16 | b12x | `unquantized` or `nvfp4` | `unquantized` or `nvfp4` | `dynamic_tensor` (default) with `nvfp4`, else unset |
| MXFP8, checkpoint or online | b12x | `unquantized` or `mxfp8` | `unquantized` or `mxfp8` | unset |

`prefill` is required and `decode` defaults to `unquantized`. Both rows need
BF16 activations (`--dtype bfloat16`).
`max_decode_tokens` is required when `decode` and `prefill` differ, and unused
when they match. A phase applies to NVFP4 linears when it uses `nvfp4` or only `unquantized`, and to
MXFP8 linears when it uses `mxfp8` or only `unquantized`; mixing `nvfp4` and
`mxfp8` is rejected, since it needs two weight copies. For example, to keep
BF16 activations for decode and quantize prefills to NVFP4:

```bash
vllm serve <model> --linear-backend b12x \
    --quantization-config '{"linear": {"phase": {"decode": "unquantized",
        "prefill": "nvfp4", "max_decode_tokens": 1024}}}'
```

Loading raises an error when a linear the phase applies to cannot serve it,
naming the weight format, the phases and the platform, and logs a warning when
no linear does, for example in an unquantized model. Other linears keep their
own activations: unquantized linears and `draft_lm_head_quantization` lm_head
copies use the model dtype. A checkpoint lm_head in a served format follows the
phase like other linears, so prefill-sized lm_head calls (e.g. prompt logprobs)
use the `prefill` format. NVFP4 linears whose per-GPU input size is not a
multiple of 128 keep BF16 activations. `phase` cannot be combined with `activation`,
`moe.phase` is not supported, and online-quantized linears reject it unless
their weights are MXFP8 (`--quantization mxfp8`). For an MXFP8 checkpoint, pass
the phase with `--quantization-config` alone: adding `--quantization mxfp8`
applies online MXFP8 to every linear and MoE layer, re-quantizing the
checkpoint's MXFP8 layers and quantizing the rest. The phase applies to every eligible
checkpoint linear; `ignore` only affects online quantization. Batched (BMM)
MXFP8 layers keep their own activations.

`dynamic_tensor` shares one NVFP4 scale, computed from the absolute maximum,
across every request in a call, so outputs depend on batch composition. NVFP4
decode calls run in CUDA graphs and take the scale from the rows the model
runner marks as real, which requires `VLLM_MOE_SKIP_PADDING=1` (the default);
CUDA graph capture fails for forward passes that do not mark them, such as
draft models. With unquantized decode and NVFP4 prefill, `max_decode_tokens`
must be at least the largest CUDA graph capture size, which keeps NVFP4 out of
CUDA graphs and so needs no padding mask. NVFP4 phases do not
support data parallelism, encoder CUDA graphs, client-supplied prompt or
multimodal embeddings, or `VLLM_BATCH_INVARIANT`.
Phases need `VLLM_USE_LAYERNAME=1`, the default on torch 2.11 and later.
MXFP8 activations have per-block scales, so padding rows cannot affect other
rows; a split still depends on the rows in a step and is not supported with
`VLLM_BATCH_INVARIANT`.

### Re-quantizing layers to a different precision

An online target can re-quantize a checkpoint-quantized (prequantized) layer at load time.

vLLM uses the checkpoint method to load and dequantize the checkpoint weight, and then applies the online quantization method. This requires the checkpoint
method to implement weight dequantization, unsupported source and target
combinations raise an error.

Currently supported combinations are:

| Checkpoint layer | Online target | Requirements |
| ---------------- | ------------- | ------------ |
| Serialized ModelOpt MXFP8 linear | any online linear method | Target-specific hardware requirements |
| Serialized compressed-tensors NVFP4 MoE | any online MoE method | Target-specific hardware requirements |
| Serialized ModelOpt NVFP4 MoE | any online MoE method | Target-specific hardware requirements |
| Serialized Quark NVFP4 MoE | any online MoE method | Target-specific hardware requirements |

For example, serialized ModelOpt MXFP8 linears can be converted to
per-channel-weight/per-token-activation FP8 at load time:

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --linear-backend auto \
  --quantization-config.linear fp8_per_channel
```

Similarly, for example, NVFP4 MoE checkpoints can be re-quantized to any online MoE method,
for example compressed-tensors NVFP4 experts to online MXFP4:

```bash
vllm serve RedHatAI/Qwen3-30B-A3B-NVFP4 \
  --quantization-config '{"targets":{"*mlp.experts*":"mxfp4"}}'
```

!!! info

    Re-quantization is available for both linear and MoE layers.

    Checkpoint-exclusion policy is not enforced by online quantization/re-quantization. Only exclusion through `--quantization-config.targets`/`--quantization-config.ignore` is enforced. 

### Online quantization on unquantized layers from partially-quantized checkpoints

Online quantization can be used on already quantized checkpoints independently of their original `quant_method` (`modelopt`, `compressed-tensors`, `quark`, etc.), for layers that are left unquantized in the original checkpoint.

The checkpoint `quant_method` remains responsible for its quantized layers, while the selected unquantized layers use the requested online method.

For example:

```bash
vllm serve amd/Qwen3.5-35B-A3B-MXFP4 \
  --quantization-config.linear mxfp8
```

adds MXFP8 quantization to the dense linear layers of a Quark checkpoint where only MOE experts are quantized.

!!! info
    `quantization_config.ignore` is an online-only exclusion: the original `quant_method` relies solely on its own ignore implementation and on the ignored layers specified in `config.json`.

### Separate Schemes for Dense and MoE Layers

You can apply different quantization schemes to dense linear layers and MoE expert layers via the `linear` and `moe` fields. Each accepts either a full spec dict, or a bare string naming an online shorthand (e.g. `"fp8_per_block"`) or weight format (e.g. `"fp8_per_block_static"`); fields not set fall back to the shorthand defaults.

```python
from vllm import LLM

# Linear: per-block FP8; MoE: per-tensor FP8 (inherited from the shorthand)
llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "linear": "fp8_per_block",
    },
)
```

Or,

```python
from vllm import LLM

# Linear: per-tensor FP8 (inherited); MoE: per-block FP8
llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "moe": "fp8_per_block",
    },
)
```

### Excluding Layers from Quantization

Use the `ignore` parameter to skip specific layers. It accepts exact layer names, regex patterns (prefixed with `re:`), and patterns understood by [`fnmatch.fnmatch`](https://docs.python.org/3/library/fnmatch.html#fnmatch.fnmatch):

```python
from vllm import LLM

llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "ignore": [
            # exact layer name
            "model.layers.1.self_attn.o_proj",
            # regex: skip all QKV projections
            "re:.*[qkv]_proj",
            # fnmatch: skip all MoE experts
            "*mlp.experts*",
        ],
    },
)
```

!!! note
    For fused layers (e.g., `qkv_proj` which fuses `q_proj`, `k_proj`, `v_proj`), patterns may match the fused name directly or all of its unfused shard names.

### Fine-Grained Per-Layer Quantization Schemes

Use the `targets` parameter to apply different online shorthands to different layers, instead of one scheme applied everywhere via `linear`/`moe`. Keys are exact layer names, regex patterns (prefixed with `re:`), or patterns understood by [`fnmatch.fnmatch`](https://docs.python.org/3/library/fnmatch.html#fnmatch.fnmatch); values are shorthand names (`fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8`, `mxfp4`, `mxfp4_hadamard32`, `int8_per_channel_weight_only`, `nvfp4_per_token`) or a `{weight, activation}` spec, which allows overriding the activation quantization of a single layer.

Example:

```python
from vllm import LLM

llm = LLM(
    "Qwen/Qwen3.5-35B-A3B",
    quantization="online",
    quantization_config={
        "targets": {
            # exact layer name
            "model.layers.0.self_attn.o_proj": "fp8_per_tensor",
            # regex: quantize all QKV projections
            r"re:.*self_attn\.qkv_proj.*": "mxfp8",
            # fnmatch: quantize all MoE experts
            "*mlp.experts*": "mxfp4",
            # spec form: MXFP4 experts with BF16 (unquantized) activations
            "*shared_expert*": {"weight": "mxfp4", "activation": None},
        },
    },
)
```

Or with the CLI:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --quantization online \
  --quantization-config '{"targets":{"model.layers.0.self_attn.o_proj":"fp8_per_tensor","re:.*self_attn\\.qkv_proj.*":"mxfp8","*mlp.experts*":"mxfp4"}}'
```

!!! info

    - `targets` is mutually exclusive with online `linear` and `moe`: set one or the other, not both.
    - A layer that matches no `targets` pattern is left unchanged from its checkpoint dtype.
    - A layer name may not match both `targets` and `ignore`, raising an error.
    - A layer may not match more than one `targets` pattern, raising an error.
    - fnmatch-style patterns are supported only by online quantization; they are
      not applied to Quark or compressed-tensors configurations.
