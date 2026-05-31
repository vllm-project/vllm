# CPU KV Cache Offloading

vLLM can offload KV cache blocks that don't fit in GPU memory to CPU RAM and reload them when needed.
This is handled by the `SimpleCPUOffloadConnector` using the CUDA DMA engine, bridging between GPU VRAM and regular
CPU RAM.

The `SimpleCPUOffloadConnector` builds on [prefix caching](automatic_prefix_caching.md) instead of throwing away evicted
blocks. These blocks are copied to a pinned buffer of CPU memory.
Transfers run on low-priority CUDA streams and don't interfere with the forward pass.

**Good for**: long or repeated contexts (shared system prompts, documents) and models under GPU memory pressure.  
**Not for**: cross-node KV cache transfer. This connector offloads to local CPU RAM only. If you need to ship KV cache
between a dedicated prefill node and a decode node over the network, see [Disaggregated Prefilling](disagg_prefill.md).

## Enabling it

```bash
# Recommended: via environment variable
VLLM_USE_SIMPLE_KV_OFFLOAD=1 vllm serve <model> --kv-cache-offload-gb 64

# Or directly via --kv-transfer-config
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 68719476736}
  }'
```

Requires prefix caching (on by default in V1).

## Configuration options

All options go inside `kv_connector_extra_config`:

| Key                         | Type | Default | Description                                                                                       |
|-----------------------------|------|---------|---------------------------------------------------------------------------------------------------|
| `cpu_bytes_to_use`          | int  | 8 GB    | Total CPU buffer size in bytes, split evenly across TP ranks.                                     |
| `cpu_bytes_to_use_per_rank` | int  | —       | Per-rank override. Takes precedence over `cpu_bytes_to_use`.                                      |
| `lazy_offload`              | bool | `false` | `true`: offload blocks just before GPU eviction. `false`: offload as soon as KV data is computed. |
| `async_register_cache`      | bool | `false` | See below.                                                                                        |

## Async startup pinning

Before the GPU can use regular system RAM, every page in the buffer must be pinned. The OS needs to lock each page at
a fixed physical address so the GPU's DMA engine always knows where to write.

Two operations are needed before the GPU can use regular system RAM:

1. **Zero-fill** — `torch.zeros` allocates the buffer and writes zeros to every page, forcing the OS to back each
   page with physical RAM. At 50 GB/s memory bandwidth, 1 TB takes roughly 20 seconds.
2. **Pinning** — `cudaHostRegister` walks every page again and locks it at a fixed physical address so the GPU's DMA
   engine can reach it directly. For 1 TB that's 256 million pages and can take several minutes.

By default, both steps happen synchronously on startup, blocking the HTTP server until they finish.

Set `async_register_cache: true` to move **both** steps to a background thread. The engine starts immediately and
serves requests using GPU-only caching. CPU offloading activates automatically once all workers finish — no restart
needed.

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 1099511627776,
      "async_register_cache": true
    }
  }'
```

Use this whenever the CPU buffer is large (tens of GB+) and fast startup matters, e.g. autoscaling.

### What happens during the pin window

While pinning is in progress, the scheduler gate is closed and only GPU VRAM is used.
In the unlikely event that the GPU VRAM is full before the pinning is done, the engine falls back to recomputing the
prefix as a cache miss.
