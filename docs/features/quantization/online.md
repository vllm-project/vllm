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
```

Or with the CLI:

```bash
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_tensor
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_block
vllm serve meta-llama/Llama-3.1-8B --quantization mxfp8
```

## Supported Schemes

| Scheme | Weight recipe | Activation recipe | Notes |
| ------ | ------------- | ------------------ | ----- |
| `fp8_per_tensor` | fp8_e4m3 data, fp32 per-tensor scale | fp8_e4m3 data, fp32 per-tensor scale | On some GPUs (Ada, Hopper) linear activations use per-token scaling for better performance |
| `fp8_per_block` | fp8_e4m3 data, fp32 per-128x128-block scale | fp8_e4m3 data, fp32 per-1x128-block scale | |
| `mxfp8` | fp8_e4m3 data, e8m0 per-1x32-block scale | fp8_e4m3 data, e8m0 per-1x32-block scale | Requires SM 100+ (Blackwell or newer) for w8a8, other GPUs use a w8a16 fallback |

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

### Re-quantizing Unquantized Layers of a Pre-quantized Checkpoint

Some pre-quantized checkpoints are *mixed precision*: they ship a subset of
layers unquantized (fp16/bf16). For example a Quark MXFP4 MoE checkpoint stores
its experts as MXFP4 but keeps the attention and dense-MLP projections in bf16.

When a checkpoint is loaded with such a native quantization method (e.g.
`quark`), you can pass a `quantization_config` overlay to online-requantize
*only* the layers the checkpoint left unquantized, using the online schemes
above. The overlay's `linear` spec selects the online method, and its `ignore`
patterns scope which unquantized layers are re-quantized. Layers already
quantized by the checkpoint, and any layer matched by `ignore`, are untouched.

```python
from vllm import LLM

# Quark MXFP4 MoE checkpoint: experts stay MXFP4, bf16 attention / dense-MLP
# projections are online-requantized to FP8 (per-token-activation,
# per-channel-weight). The MoE gate and lm_head are left in bf16 via `ignore`
# (exact names or `re:` regex, same as elsewhere on this page). MoE experts are
# handled by the checkpoint and never routed here.
llm = LLM(
    "amd/Qwen3.5-35B-A3B-MXFP4",
    quantization_config={
        "linear": "fp8_per_channel",
        "ignore": [
            "lm_head",
            "re:.*mlp\\.gate$",
        ],
    },
)
```

The same overlay can be passed on the CLI:

```bash
vllm serve amd/Qwen3.5-35B-A3B-MXFP4 \
  --quantization-config '{"linear":"fp8_per_channel","ignore":["lm_head","re:.*mlp\\.gate$"]}'
```

For MLA checkpoints (e.g. DeepSeek), the attention up-projection `kv_b_proj`
is numerically sensitive and should stay in bf16, so add it to `ignore`
alongside the router `gate` and `lm_head`:

```python
from vllm import LLM

# Quark MXFP4 DeepSeek (MLA) checkpoint: experts stay MXFP4, the bf16 attention
# / dense-MLP projections are online-requantized to FP8, while the router gate,
# lm_head and the MLA kv_b_proj are kept in bf16 via `ignore`.
llm = LLM(
    "amd/DeepSeek-R1-0528-MXFP4",
    quantization_config={
        "linear": "fp8_per_channel",
        "ignore": [
            "lm_head",
            "re:.*mlp\\.gate$",
            "re:.*kv_b_proj.*",
        ],
    },
)
```

!!! note
    This path is opt-in: with no `quantization_config` overlay, the unquantized layers stay in their original checkpoint dtype.
