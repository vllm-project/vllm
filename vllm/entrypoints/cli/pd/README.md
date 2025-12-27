# vLLM Prefill-Decode Disaggregated Deployment (pdjob)

This module provides a way to deploy vLLM with **Prefill-Decode (PD) Disaggregation** using Ray for distributed orchestration.

## Overview

PD Disaggregation separates the prefill phase (processing input tokens) and the decode phase (generating output tokens) onto different GPU workers. This allows for:

- **Better resource utilization**: Prefill is compute-intensive, decode is memory-bandwidth-intensive
- **Independent scaling**: Scale prefill and decode workers independently based on workload
- **Improved throughput**: Parallel execution of prefill and decode for different requests

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                              Ray Cluster                                 │
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   Prefill    │     │   Prefill    │     │    Decode    │  ...       │
│  │   Worker 1   │     │   Worker N   │     │   Worker 1   │            │
│  │  (vllm serve)│     │  (vllm serve)│     │  (vllm serve)│            │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘            │
│         │                    │                    │                     │
│         └────────────────────┼────────────────────┘                     │
│                              │                                          │
│                    ┌─────────▼─────────┐                               │
│                    │   Proxy Server    │                               │
│                    │  (Load Balancer)  │                               │
│                    └─────────┬─────────┘                               │
│                              │                                          │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   External Clients  │
                    └─────────────────────┘
```

## Module Structure

```text
vllm/entrypoints/cli/pd/
├── __init__.py
├── README.md              # This file
├── base.py                # Base classes: BasePDJob, _BaseVllmService, Service, ProxyServer
├── config.py              # Configuration parsing from YAML
└── multiple_prefills.py   # MultiplePrefillsPDJob implementation
```

## Quick Start

### 1. Start a Ray Cluster

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address='<head-node-ip>:6379'
```

### 2. Prepare Configuration File

Create a `config.yaml` file (see `examples/pdjob/config.yaml` for a complete example):

```yaml
timeout: 3600
num_gpus: 16
gpus_per_worker: 1
working_dir: /tmp/vllm
envs: {}

scheduler:
  port: 8021
  # Custom command to start the proxy server
  # Placeholders:
  #   {PORT} - scheduler port
  #   $PREFILL_HOST_1, $PREFILL_PORT_1, $PREFILL_URL_1, ... - prefill service info
  #   $DECODE_HOST_1, $DECODE_PORT_1, $DECODE_URL_1, ...   - decode service info
  command: "python toy_proxy_server.py --port {PORT} --prefiller-hosts $PREFILL_HOST_1 --prefiller-ports $PREFILL_PORT_1 --decoder-hosts $DECODE_HOST_1 --decoder-ports $DECODE_PORT_1"

general:
  params:
    model: /path/to/your/model
    host: 0.0.0.0
    port: '{PORT}'
    served_model_name: my_model
    _extra_params: --trust-remote-code

prefill:
  envs:
    VLLM_ATTENTION_BACKEND: FLASHMLA
  num_gpus: 8
  replicas: 1
  params:
    tensor_parallel_size: 8
    kv_transfer_config: '{"kv_connector":"NixlConnector","kv_role":"kv_producer",...}'
    _extra_params: ''

decode:
  envs:
    VLLM_ATTENTION_BACKEND: FLASHMLA
  num_gpus: 8
  replicas: 1
  params:
    tensor_parallel_size: 1
    data_parallel_size: 8
    kv_transfer_config: '{"kv_connector":"NixlConnector","kv_role":"kv_consumer",...}'
    _extra_params: ''
```

### 3. Run pdjob

```bash
vllm pdjob --config=/path/to/config.yaml
```

## Configuration Reference

### Top-level Configuration

| Field | Type | Description |
|-------|------|-------------|
| `timeout` | int | Timeout in seconds for service startup |
| `num_gpus` | int | Total number of GPUs in the cluster |
| `gpus_per_worker` | int | GPUs per vLLM worker |
| `working_dir` | str | Working directory for Ray |
| `envs` | dict | Global environment variables |

### Scheduler Configuration

| Field | Type | Description |
|-------|------|-------------|
| `port` | int | Proxy server port |
| `command` | str | Command to start the proxy server (required) |

#### Command Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{PORT}` | Scheduler port |
| `$PREFILL_HOST_N` | Prefill service N's host |
| `$PREFILL_PORT_N` | Prefill service N's port |
| `$PREFILL_URL_N` | Prefill service N's full URL |
| `$DECODE_HOST_N` | Decode service N's host |
| `$DECODE_PORT_N` | Decode service N's port |
| `$DECODE_URL_N` | Decode service N's full URL |

### Role Configuration (prefill/decode)

| Field | Type | Description |
|-------|------|-------------|
| `envs` | dict | Environment variables for this role |
| `num_gpus` | int | GPUs required per replica |
| `replicas` | int | Number of replicas |
| `params` | dict | vLLM serve parameters |
| `params._extra_params` | str | Additional CLI arguments |

### Supported KV Connectors

The following KV connectors are registered in `vllm/distributed/kv_transfer/kv_connector/factory.py`:

| Connector | Description |
|-----------|-------------|
| `NixlConnector` | NIXL-based KV cache transfer for disaggregated prefill-decode |
| `P2pNcclConnector` | Point-to-point NCCL transfer for high-speed GPU communication |
| `MultiConnector` | Multi-backend connector supporting multiple KV transfer methods |
| `SharedStorageConnector` | Shared storage based connector using external storage |
| `LMCacheConnectorV1` | LMCache integration for KV cache management |
| `LMCacheMPConnector` | LMCache multi-process connector |
| `OffloadingConnector` | KV cache offloading to CPU/storage |
| `DecodeBenchConnector` | Decode benchmarking connector for performance testing |

## Execution Flow

1. **Configuration Parsing** (`config.py`)
   - Parse YAML configuration
   - Merge general params with role-specific params
   - Extract KV connector type

2. **Job Initialization** (`pdjob.py`)
   - Initialize Ray connection
   - Select job type based on KV connector
   - Create job instance

3. **Service Startup** (`multiple_prefills.py`)
   - Create placement groups for GPU allocation
   - Start prefill service actors
   - Get KV IP from first prefill service
   - Start decode service actors with KV IP

4. **Health Check** (`base.py`)
   - Wait for all services to be healthy
   - Check `/health` endpoint for each service

5. **Proxy Server Startup** (`base.py`)
   - Create ProxyServer actor on head node
   - Replace placeholders in command with actual service URLs
   - Execute proxy server command

## Example Proxy Server Commands

### Using toy_proxy_server.py

```yaml
command: "python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port {PORT} --prefiller-hosts $PREFILL_HOST_1 $PREFILL_HOST_2 --prefiller-ports $PREFILL_PORT_1 $PREFILL_PORT_2 --decoder-hosts $DECODE_HOST_1 $DECODE_HOST_2 --decoder-ports $DECODE_PORT_1 $DECODE_PORT_2"
```

### Using Cargo-based proxy

```yaml
command: "cargo run --release -- --policy consistent_hash --vllm-pd-disaggregation --prefill $PREFILL_URL_1 --prefill $PREFILL_URL_2 --decode $DECODE_URL_1 --decode $DECODE_URL_2 --host 0.0.0.0 --port {PORT}"
```

## Troubleshooting

### Common Issues

1. **"Unrecognized kv_connector"**
   - Ensure your `kv_transfer_config` uses a supported connector
   - Check that the connector name is spelled correctly

2. **Service health check timeout**
   - Increase `timeout` in configuration
   - Check Ray logs for startup errors
   - Verify GPU resources are available

3. **Proxy server fails to start**
   - Verify the `command` path is correct
   - Check that all placeholders are properly replaced
   - Review proxy server logs

### Debugging

```bash
# View Ray logs
ray logs

# Check Ray dashboard
# Navigate to http://<head-node-ip>:8265

# Enable verbose logging
vllm pdjob --config=config.yaml 2>&1 | tee pdjob.log
```

## Related Documentation

- [NIXL Connector Usage](../../../docs/features/nixl_connector_usage.md)
- [Disaggregated Encoder](../../../docs/features/disagg_encoder.md)
- [Example Configuration](../../../examples/pdjob/config.yaml)
