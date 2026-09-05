# Confident Decoding

Confident Decoding is a training-free decoding strategy that selects logits from near-final intermediate layers based on prediction entropy, instead of always using the final layer. It is described in [Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding](https://arxiv.org/abs/2606.21906).

## Usage

Enable via `--additional-config`:

```bash
vllm serve /path/to/model \
  --additional-config '{
    "enable_multi_layer_entropy_selection": true,
    "select_method": "trough",
    "p": 1.0,
    "trough_max_backtrack_layers": 10
  }'
```

## Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_multi_layer_entropy_selection` | bool | `false` | Global switch for Confident Decoding |
| `select_method` | str | `"trough"` | Layer selection strategy (`trough`, `trough-m1`, `last-m8`, etc.) |
| `p` | float | `1.0` | Probability of using selected-layer logits; `0.0` equals standard final-layer decoding |
| `trough_max_backtrack_layers` | int | `0` | Max layers to backtrack; `>0` uses this value; `<0` unlimited |
| `trough_backtrack_ratio` | float | `0.0` | Used when `trough_max_backtrack_layers == 0` |
| `trough_log_interval` | int | `0` | Periodic selection statistics logging; `0` disables |

## Supported models (initial)

- Llama family (`LlamaForCausalLM`)

Additional model families are planned in follow-up PRs. See [RFC #48080](https://github.com/vllm-project/vllm/issues/48080).

## Limitations

- Pipeline parallelism (`pp > 1`) is not supported and the feature is disabled automatically.
- Confident Decoding adds bounded `lm_head` and entropy overhead for candidate layers.
- `p=0.0` should match standard final-layer decoding and is the recommended regression setting.
