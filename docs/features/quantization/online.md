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
    weight: <quant-key-name>      # e.g. fp8_per_block_static
    activation: <quant-key-name>  # e.g. fp8_per_block_dynamic
  moe:
    weight: <quant-key-name>
    activation: <quant-key-name>
  ignore: [<layer-name-or-regex>, ...]
```

`linear` and `moe` are per-layer-kind specs. Each takes a `weight` and an
`activation` key — both naming entries from the public `QUANT_KEY_NAMES`
table (`mxfp8`, `mxfp4`, `fp8_per_tensor_static`, `fp8_per_block_static`,
`fp8_per_block_dynamic`, `fp8_per_token`, `int8_per_channel_static`). Fields
left out fall back to either the `--quantization` shorthand's defaults or, for
already-quantized checkpoints, the value baked into the checkpoint.

`linear` and `moe` also accept a bare string for compactness: an online
shorthand name (e.g. `"fp8_per_block"`) pulls that shorthand's matching
slot, otherwise the string is treated as a weight format name (shorthand for
`{"weight": <name>}`).

The CLI accepts the same shape as JSON via `--quantization-config`, or as
dotted keys for individual fields. The two are equivalent:

```bash
vllm serve <model> --quantization-config '{"moe":{"activation":"mxfp8"}}'
vllm serve <model> --quantization-config.moe.activation mxfp8
```

The dotted form is the easier shape for shell quoting; nested keys merge into
the same dict (e.g. `--quantization-config.linear.weight fp8_per_block_static`
plus `--quantization-config.moe.activation mxfp8`).

### Activation overrides on already-quantized checkpoints

`quantization_config` is also consumed by some checkpoint-quant paths to
let you pick an activation format independently of the weights baked into
the checkpoint. The headline case is gpt-oss MXFP4 weights, where you can
opt into MXFP8 activations:

```bash
# Auto-detected MXFP4 weights from the checkpoint, MXFP8 activations on top.
vllm serve openai/gpt-oss-20b \
    --quantization-config.moe.activation mxfp8

# Same, pinned to the FlashInfer CUTLASS backend.
vllm serve openai/gpt-oss-20b \
    --moe-backend flashinfer_cutlass \
    --quantization-config.moe.activation mxfp8
```

Without an override, gpt-oss runs with BF16 activations (today's default).

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
