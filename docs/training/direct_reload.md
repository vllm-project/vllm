# Direct Weight Reload

Direct reload updates a running model's weights in place. It is the default
when runtime weight updates are enabled (`--weight-transfer-config`, any
backend but `sharded_rdt`). Without them, a model loads and reloads as with
[Layerwise (Re)loading](layerwise.md).

Set `VLLM_RELOAD_MODE=layerwise` to update weights layerwise instead, for
example for the configurations direct reload refuses (see below). The mode is
fixed when the model loads; every later update of that model uses it.

## How It Works

- **Cold start** records each tensor's checkpoint layout after loading.
- **Begin** views every tensor back to its checkpoint layout, in its existing
  storage.
- **Load**: the model's weight loaders write the new checkpoint tensors, as at
  cold start, through `reload_weights` or a checkpoint-format
  [weight transfer](weight_transfer/README.md) backend.
- **Finish** reruns the cold-start post-load processing.

No layer is buffered and no tensor changes address, so captured CUDA graphs
stay valid. Each update must send every weight.

## Online Quantization

[Online quantization](../features/quantization/online.md) works in direct mode.
Each high-precision tensor is quantized as it is loaded, into the layout of the
matching pre-quantized checkpoint; the trainer keeps sending BF16 weights under
their checkpoint names.

```bash
vllm serve Qwen/Qwen3-30B-A3B --quantization fp8_per_block \
    --weight-transfer-config '{"backend": "nccl"}'
```

| `--quantization` | Linear layers | MoE layers |
| - | - | - |
| `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8` | Supported | Supported |
| `mxfp4` (`--quantization-config.moe mxfp4`) | Refused | Supported |
| `nvfp4_per_token` | Stay BF16 | Supported |

With NVFP4, the w1 and w3 (gate and up) weights of an expert share one global
scale, so both must arrive in the same update (between one
`start_weight_update` and `finish_weight_update`), in either order. The first
of the two is the only tensor direct reload holds: it waits until the second
arrives, and an update that delivers only one of them fails at finish.

## Refused Configurations

These configurations serve as usual, but direct reload refuses to update them;
use `VLLM_RELOAD_MODE=layerwise` to update them. The error is raised at the
start of the first weight update, leaving the model unchanged.

- Online quantization not in the table above, or of layers loaded outside
  their weight loaders (e.g. gpt-oss experts).
- `fp8_per_block` layers whose tensor-parallel shards do not fit its blocks.
- The NCCL weight transfer load plan (`use_load_plan`) with online
  quantization.
- `torchao` quantization, Humming kernels, weight offloading, and
  `draft_lm_head_quantization`.
- Weights loaded pre-processed from the weight cache.
- A draft model sharing tensors with a refused target model.

LoRA adapters are supported.
