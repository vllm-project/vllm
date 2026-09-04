Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

To enable this feature, use the `--load-format fastsafetensors` command-line argument.

Loading pre-sharded checkpoints
-------------------------------

Use `fastsafetensors_sharded` when the checkpoint already contains one shard per
tensor-parallel rank. Each worker loads only files matching its rank and copies
all of that rank's parts to the GPU in one FastSafetensors transfer. This avoids
the expensive cross-node weight distribution step required when every rank
cannot read its final shard directly:

```bash
vllm serve /path/to/sharded/model \
    --load-format fastsafetensors_sharded
```

The default filename pattern is
`model-rank-{rank}-part-{part}.safetensors`, matching checkpoints created by
[`save_sharded_state_offline.py`](../../../examples/features/sharded_state/save_sharded_state_offline.py).
Set a custom pattern or tune FastSafetensors through
`--model-loader-extra-config`:

```bash
vllm serve /path/to/sharded/model \
    --load-format fastsafetensors_sharded \
    --model-loader-extra-config \
    '{"pattern":"rank-{rank}-{part}.safetensors","max_threads":16}'
```

Supported keys are `pattern`, `nogds`, `bbuf_size_kb`, `max_threads`,
`max_copy_block_size`, and `debug_log`. When `nogds` is omitted, the loader
tries GDS first and retries with `nogds=true` if GDS initialization fails before
any tensor is copied. Set `nogds=false` to require GDS or `nogds=true` to skip
GDS.
