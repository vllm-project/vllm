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

# MXFP4 weight quantization (1x32 per-block FP4_E2M1 with E8M0 scale)
# Activation format depends on the selected linear/MoE backend
llm = LLM("meta-llama/Llama-3.1-8B", quantization="mxfp4")

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

## Supported Schemes

| Scheme | Weight recipe | Activation recipe | Notes |
| ------ | ------------- | ------------------ | ----- |
| `fp8_per_tensor` | fp8_e4m3 data, fp32 per-tensor scale | fp8_e4m3 data, fp32 per-tensor scale | On some GPUs (Ada, Hopper) linear activations use per-token scaling for better performance |
| `fp8_per_block` | fp8_e4m3 data, fp32 per-128x128-block scale | fp8_e4m3 data, fp32 per-1x128-block scale | |
| `mxfp8` | fp8_e4m3 data, e8m0 per-1x32-block scale | fp8_e4m3 data, e8m0 per-1x32-block scale | Requires SM 100+ (Blackwell or newer) for w8a8, other GPUs use a w8a16 fallback |
| `mxfp4` | fp4_e2m1 data, e8m0 per-1x32-block scale ([OCP MX specs](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)) | - linear: fp4_e2m1 data, e8m0 per-1x32-block scale in some backends, or BF16. <br> - MOE: fp4_e2m1 data, e8m0 per-1x32-block scale. | Linear MXFP4 backend is auto-selected per platform, not enforcing activation dtype. Some use BF16 activation. Use `--linear-backend` to pin one (e.g. `--linear-backend flashinfer`). |

## Advanced Configuration

For fine-grained control, use a `quantization_config` dictionary.

### Schema

```yaml
quantization_config:
  linear:
    weight: <name>      # see QUANT_KEY_NAMES in vllm/config/quantization.py
    activation: <name>
  moe:
    weight: <name>
    activation: <name>
  ignore: [<layer-name-or-regex>, ...]
```

`linear` and `moe` accept a full `{weight, activation}` dict, or a bare
string. A string resolves first against the `--quantization` shorthands
(taking the matching layer-kind slot), then against `QUANT_KEY_NAMES` as a
weight name. Unset fields fall back to the `--quantization` shorthand's
defaults, or for already-quantized checkpoints to whatever the checkpoint
declares.

On XPU, non-block FP8 scaled-mm linear layers default to W8A16; setting `--linear-backend xpu` forces W8A8. Use `--linear-backend xpu_woq` to explicitly select weight-only quantization (W8A16).

The CLI accepts the same shape as JSON or as dotted keys:

```bash
vllm serve <model> --quantization-config '{"moe":{"activation":"mxfp8"}}'
vllm serve <model> --quantization-config.moe.activation mxfp8
```

### Activation overrides on already-quantized checkpoints

For checkpoint-quantized models, `quantization_config` lets you pick an
activation format independently of the baked-in weights. The supported
overrides are checkpoint-specific; today this is wired up for MXFP4 MoE
checkpoints (gpt-oss) where you can opt into FP8 activations:

```bash
vllm serve openai/gpt-oss-20b --quantization-config.moe.activation mxfp8
```

Combine with `--moe-backend` to pin a specific kernel family.

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

Use the `ignore` parameter to skip specific layers. It accepts exact layer names and regex patterns (prefixed with `re:`):

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
        ],
    },
)
```

!!! note
    For fused layers (e.g., `qkv_proj` which fuses `q_proj`, `k_proj`, `v_proj`), the ignore pattern must match the **unfused** shard names (`q_proj`, `k_proj`, `v_proj`), not the fused name.
