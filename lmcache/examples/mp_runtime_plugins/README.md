# LMCache MP Runtime Plugin Examples

## Overview

This directory contains example runtime plugins designed for the
**multiprocess (MP / ZMQ) server** mode of LMCache.

> **Key difference from non-MP plugins:**
> In MP mode the `LMCACHE_RUNTIME_PLUGIN_CONFIG` environment variable
> carries an **aggregated JSON dict** with sections like `mp_config`,
> `storage_manager_config`, and `obs_config` — rather than a single
> `LMCacheEngineConfig` JSON.

## Design

For architecture details, component diagrams, and data flow, see
[docs/design/v1/multiprocess/mp_runtime_plugin.md](../../docs/design/v1/multiprocess/mp_runtime_plugin.md).

## Files

| File | Language | Description |
|------|----------|-------------|
| `mp_plugin.py` | Python | Parses the aggregated MP config and runs a periodic status reporter |
| `mp_heartbeat.sh` | Bash | Extracts config fields via `jq` and runs a heartbeat loop |

## Environment Variables

Plugins receive the following environment variables:

| Variable | Description |
|----------|-------------|
| `LMCACHE_RUNTIME_PLUGIN_CONFIG` | Aggregated JSON config (see below) |

> **Note:** Unlike the non-MP (vLLM integration) mode, the MP server
> does **not** have multiple roles (e.g. `SCHEDULER`, `WORKER`).
> All plugins run in the single MP server process. The filename
> prefix-based role filtering has no effect in MP mode.

### Config JSON Structure (MP mode)

```json
{
  "mp_config": {
    "host": "localhost",
    "port": 5555,
    "chunk_size": 256,
    "max_workers": 1,
    "hash_algorithm": "blake3",
    "engine_type": "default",
    "runtime_plugin_locations": ["examples/mp_runtime_plugins/"]
  },
  "storage_manager_config": {
    "l1_manager_config": {
      "memory_config": {
        "size_in_bytes": 10737418240,
        "use_lazy": true
      }
    },
    "eviction_config": {
      "eviction_policy": "LRU",
      "trigger_watermark": 0.8,
      "eviction_ratio": 0.2
    }
  },
  "obs_config": {
    "enabled": true,
    "metrics_enabled": true,
    "logging_enabled": true,
    "tracing_enabled": false
  }
}
```

## Quick Start

Launch the LMCache MP server with the plugin directory:

```bash
python -m lmcache.v1.multiprocess.server \
    --host localhost --port 5555 \
    --l1-size-gb 10 \
    --eviction-policy LRU \
    --runtime-plugin-locations examples/mp_runtime_plugins/
```

You should see plugin output in the server logs:

```
[mp_plugin] Started
[mp_plugin] MP server: host=localhost  port=5555  chunk_size=256
[mp_plugin] Storage: L1=10.0GB  eviction=LRU  watermark=0.8
[mp_plugin] heartbeat #0
```
