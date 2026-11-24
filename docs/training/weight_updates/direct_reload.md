# Direct Weight Reload

Direct reload updates a running model's weights in place, and is the default
(`VLLM_RELOAD_MODE=auto`) with runtime weight updates (`--weight-transfer-config`):
their weight-transfer updates and disk `reload_weights` calls reload directly,
and each must load every weight. Anything listed under
[Refused Configurations](#refused-configurations) reloads layerwise instead
(see [Layerwise (Re)loading](layerwise.md)) with a log saying why.
`VLLM_RELOAD_MODE=direct` enables direct reload also without
`--weight-transfer-config` (disk `reload_weights`) and raises an error for
those refusals; `VLLM_RELOAD_MODE=layerwise` turns it off.

## How It Works

- **Cold start** records each tensor's checkpoint layout after loading.
- **Begin** views every tensor back to its checkpoint layout, in its existing
  storage.
- **Load**: the model's weight loaders write the new checkpoint tensors, as at
  cold start, through `reload_weights` or a checkpoint-format
  [weight transfer](README.md) backend.
- **Finish** reruns the cold-start post-load processing.

No layer is buffered and no tensor changes address, so captured CUDA graphs
stay valid. Each update must send every weight.

## Online Quantization

With `VLLM_RELOAD_MODE=direct`, the
[online quantization](../../features/quantization/online.md) schemes below quantize
each high-precision tensor as it is loaded, into the layout of the matching
pre-quantized checkpoint; the trainer keeps sending BF16 weights under their
checkpoint names.

```bash
VLLM_RELOAD_MODE=direct vllm serve Qwen/Qwen3-30B-A3B --quantization fp8_per_block
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

- Weight offloading, `torchao` quantization, `draft_lm_head_quantization`, and
  the `sharded_rdt` weight transfer backend.
- LoRA, in `auto` mode only.
- Online quantization. With `VLLM_RELOAD_MODE=direct` the schemes in the table
  above reload directly, except `mxfp4` and `mxfp4_hadamard32` linear layers,
  `int8_per_channel_weight_only`, `lut_b_moe`, a quantized lm_head, activation
  overrides, `hadamard_size`, re-quantization of checkpoint-quantized layers,
  gpt-oss models, and `fp8_per_block` linear layers whose tensor-parallel
  shards are not multiples of 128.
- Layers using Humming kernels.
- Pre-processed weights (weight cache), with a log even under
  `VLLM_RELOAD_MODE=direct`.

The NCCL weight transfer load plan (`use_load_plan`) requires direct reload and
refuses online quantization.
