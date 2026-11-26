# LMCache Examples

This folder demonstrates how to use LMCache with vLLM v1 for KV cache
offloading, disaggregated prefilling, and KV cache sharing.

## Integration modes

LMCache integrates with vLLM v1 in two ways:

- **In-process mode** (`LMCacheConnectorV1`): LMCache runs inside the vLLM
  process and is configured through environment variables or a YAML config
  file (`LMCACHE_CONFIG_FILE`). This is the simplest way to add single-node
  CPU/disk offloading.
- **Multi-process (MP) mode** (`LMCacheMPConnector`): LMCache runs as a
  standalone server (`lmcache server`) that owns the KV cache storage; one or
  more vLLM instances connect to it. This is the recommended mode for
  distributed KV storage and for sharing KV cache across instances. See the
  [LMCache docs](https://docs.lmcache.ai) for the full MP setup.

### MP mode with the extensible KV cache

vLLM's extensible KV cache (`--enable-extensible-kv-cache`) allocates the KV
cache through the CUDA VMM API, which legacy CUDA IPC cannot share.
`LMCacheMPConnector` declares legacy CUDA IPC by default, which keeps the
extensible KV cache off. With `"lmcache.mp.use_vmm_api": true` in its
`kv_connector_extra_config`, also as a `MultiConnector` child, it declares
POSIX file descriptor export instead: vLLM allocates the KV cache exportable,
and the LMCache server imports each layer through a file descriptor. vLLM
refuses to start if the extensible KV cache is then off, e.g. with
`kv_cache_memory_bytes`.

Next to `NixlConnector` in a `MultiConnector`, on a host with an IMEX channel
(e.g., GB200/GB300) the KV cache gets a FABRIC handle only, and the LMCache server imports it
through FABRIC instead; this is not yet verified, and the server needs an IMEX
channel too. Without an IMEX channel, NIXL's transfers within a host fall back
to TCP.

It requires:

- an LMCache release that includes the file descriptor channel
  ([LMCache/LMCache#6266](https://github.com/LMCache/LMCache/pull/6266)) and
  declares `kv_cache_access`
  ([LMCache/LMCache#6487](https://github.com/LMCache/LMCache/pull/6487)),
- a CUDA GPU, with vLLM and the LMCache server on the same host, running as the
  same user and sharing `/dev/shm` (`--ipc host` for containers),
- a TCP server URL (`"lmcache.mp.host": "tcp://..."` and `"lmcache.mp.port"`),
  from whose port the file descriptor socket path is derived.

## 1. CPU offload (in-process)

- `python cpu_offload_lmcache.py` - CPU offloading with `LMCacheConnectorV1`
  for vLLM v1.

## 2. CPU offload (multi-process)

- `bash cpu_offload_lmcache_mp.sh` - CPU offloading with `LMCacheMPConnector`,
  using a standalone `lmcache server`. vLLM provides a built-in shortcut for
  this setup via `--kv-offloading-backend lmcache` and
  `--kv-offloading-size <GiB>`.

## 3. Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using
NIXL on a single node.

### Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 2 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Run
`cd disagg_prefill_lmcache_v1`
to get into `disagg_prefill_lmcache_v1` folder, and then run

```bash
bash disagg_example_nixl.sh
```

to run disaggregated prefill and benchmark the performance.

### Components

#### Server Scripts

- `disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `disagg_prefill_lmcache_v1/disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_prefill_lmcache_v1/disagg_example_nixl.sh` - Main script to run the example

#### Configuration

- `disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server
- `disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` - Configuration for decoder server

#### Log Files

The main script generates several log files:

- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server

## 4. KV Cache Sharing

The `kv_cache_sharing_lmcache_v1.py` example demonstrates how to share KV
caches between vLLM v1 instances through a centralized LMCache server.
