# Direct Weight Reload

Direct reload updates a running model's weights in place. It is the default
(`VLLM_RELOAD_MODE=auto`) for engines with runtime weight updates
(`--weight-transfer-config`), except for the configurations below and online
quantization, which reload layerwise (see [Layerwise (Re)loading](layerwise.md)).
`VLLM_RELOAD_MODE=direct` also enables it for disk `reload_weights` and for the
online quantization schemes listed below; `layerwise` disables it.

!!! tip
    Prefer direct reload for RL weight updates. Use layerwise reload only for
    the configurations direct reload refuses (see below).

```bash
VLLM_RELOAD_MODE=direct vllm serve <model>
```

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

With `VLLM_RELOAD_MODE=direct`, the supported
[online quantization](../features/quantization/online.md) schemes below quantize
each high-precision tensor as it is loaded, into the layout of the matching
pre-quantized checkpoint; the trainer keeps sending BF16 weights under their
checkpoint names. Other schemes, and some configurations and models (see
[Refused Configurations](#refused-configurations)), are not supported.

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

In `auto` mode the following reload layerwise. With `VLLM_RELOAD_MODE=direct`,
the configurations raise an error at startup and the modules reload layerwise:

- Online quantization not in the table above: `mxfp4` and `mxfp4_hadamard32`
  linear layers, `int8_per_channel_weight_only`, `lut_b_moe`, a quantized
  lm_head, activation overrides, `hadamard_size`, and re-quantization of
  checkpoint-quantized layers.
- gpt-oss models with online quantization.
- `fp8_per_block` linear layers whose tensor-parallel shards are not
  multiples of 128.
- The NCCL weight transfer load plan (`use_load_plan`) with online
  quantization.
- `torchao` quantization, Humming kernels, weight offloading,
  `draft_lm_head_quantization`, and the `sharded_rdt` weight transfer backend.
