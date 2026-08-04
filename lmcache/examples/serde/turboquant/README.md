# TurboQuant Serde End-to-End Example

This example demonstrates the per-adapter serde feature: the L2 disk adapter
quantizes KV cache to **TurboQuant** before writing to disk, and dequantizes back to
the original dtype on prefetch.

## What it does

1. Starts an `lmcache server` with:
   - **L1**: 20 GB CPU memory cache, LRU eviction
   - **L2**: filesystem (disk) adapter at `/tmp/lmcache_turboquant_serde_disk`
   - **Serde**: `TurboQuant` (`TurboQuant preset`) attached to the L2 adapter
2. Starts vLLM connected via `LMCacheMPConnector`
3. Sends an inference request — KV is computed, written to L1, then asynchronously
   serialized (TurboQuant) and stored to L2 disk
4. Calls the lmcache HTTP API to **force-clear L1** (CPU cache)
5. Re-sends the same request — L1 misses, L2 prefetch fires, the serialized
   bytes are loaded from disk and **deserialized** back into KV-shaped buffers,
   then vLLM resumes from cache

## Files

- `run_serde_turboquant_example.sh` — full end-to-end: `lmcache server` + `vllm serve` + real inference, then clear L1 and re-infer to hit the L2 path.

## Quick sanity check (no vLLM required)

The pytest suite includes TurboQuant serde tests that exercise direct CUDA, MockL2, and filesystem-backed L2 round-trips without needing vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/v1/distributed/serde/test_turboquant.py -q -s
```

## Requirements

- vLLM installed (`vllm serve` works)
- `lmcache` CLI installed (`lmcache server --help` works)
- 1 GPU (default `CUDA_VISIBLE_DEVICES=0`)
- PyTorch + Triton CUDA environment

## Run

```bash
./run_serde_turboquant_example.sh
```

You can override defaults via environment variables:

```bash
MODEL="meta-llama/Llama-3.1-8B-Instruct" \
GPU_DEVICE=0 \
L1_SIZE_GB=20 \
LMCACHE_PORT=6555 \
VLLM_PORT=8000 \
TQ_PRESET=turboquant_k8v4 \
TQ_BLOCK_SIZE=16 \
./run_serde_turboquant_example.sh
```

Server output is streamed to stdout. Logs are also saved under
`/tmp/lmcache_turboquant_serde_example/{lmcache,vllm}.log` (override with `TMP_DIR`).

## L2 adapter config syntax

The serde is attached per-adapter via a `serde` sub-dict in the `--l2-adapter`
JSON. Each adapter independently decides whether to use serde.

```json
{
  "type": "fs",
  "base_path": "/tmp/lmcache_turboquant_serde_disk",
  "serde": {
    "type": "turboquant",
    "preset": "turboquant_k8v4",
    "block_size": 16
  }
}
```

To disable serde for an adapter, omit the `serde` field.

## Adding a custom serde

1. Implement `Serializer` and `Deserializer` from
   `lmcache.v1.distributed.serde`
2. Register a factory:

   ```python
   from lmcache.v1.distributed.serde import (
       AsyncSerdeProcessor,
       register_serde_factory,
   )

   def _create_my_serde(config: dict):
       return AsyncSerdeProcessor(MySerializer(), MyDeserializer())

   register_serde_factory("mine", _create_my_serde)
   ```

3. Reference it in the adapter config: `"serde": {"type": "mine", ...}`
