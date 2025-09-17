# NixlConnector Usage Guide

NixlConnector is a high-performance KV cache transfer connector for vLLM's disaggregated prefilling feature. It provides fully asynchronous send/receive operations using the NIXL library for efficient cross-process KV cache transfer.

## Prerequisites

### Installation

Install the NIXL library:

- Refer to [NIXL official repository](https://github.com/ai-dynamo/nixl) for installation instructions
- The required NIXL version can be found in `requirements/kv_connectors.txt` and other relevant config files

### Transport Configuration

NixlConnector uses NIXL library for underlying communication, which supports multiple transport backends. UCX (Unified Communication X) is the primary default transport library used by NIXL. Configure transport environment variables:

```bash
# Example UCX configuration, adjust according to your enviroment
export UCX_TLS=all  # or specify specific transports like "rc,ud,sm"
export UCX_NET_DEVICES=all  # or specify network devices like "mlx5_0:1"
```

!!! tode
    When using UCX as the transport backend, NCCL environment variables (like `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) are not applicable to NixlConnector, so configure UCX-specific environment variables instead of NCCL variables.

## Basic Usage (on the same host)

### Producer (Prefiller) Configuration

Start a prefiller instance that produces KV caches:

```bash
# 1st GPU as prefiller
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B \
  --port 8100 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### Consumer (Decoder) Configuration

Start a decoder instance that consumes KV caches:

```bash
# 2nd GPU as decoder
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-0.6B \
  --port 8200 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### Proxy Server

Use a proxy server to route requests between prefiller and decoder:

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 \
  --prefiller-hosts localhost \
  --prefiller-ports 8100 \
  --decoder-hosts localhost \
  --decoder-ports 8200
```

## Environment Variables

- `VLLM_NIXL_SIDE_CHANNEL_PORT`: Port for NIXL handshake communication
    - Default: 5600
    - **Required for both prefiller and decoder instances**
    - Each vLLM worker needs a unique port on its host; using the same port number across different hosts is fine
    - For TP/DP setups: base_port + dp_rank * tp_size
    - Used for initial connection establishment between prefiller and decoder

- `VLLM_NIXL_SIDE_CHANNEL_HOST`: Host for side channel communication
    - Default: "localhost"
    - Set when prefiller and decoder are on different machines
    - Connection info is passed via KVTransferParams from prefiller to decoder for handshake

- `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`: Timeout for request abortion (seconds) (optional)
    - Default: 120

## Multi-Instance Setup

### Multiple Prefiller Instances on Different Machines

```bash
# Prefiller 1 on Machine A (example IP: ${IP1})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP1} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Prefiller 2 on Machine B (example IP: ${IP2})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP2} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
```

### Multiple Decoder Instances on Different Machines

```bash
# Decoder 1 on Machine C (example IP: ${IP3})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP3} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# Decoder 2 on Machine D (example IP: ${IP4})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP4} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

### Proxy for Multiple Instances

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 \
  --prefiller-hosts ${IP1} ${IP2} \
  --prefiller-ports 8000 8000 \
  --decoder-hosts ${IP3} ${IP4} \
  --decoder-ports 8000 8000
```

### KV Role Options

- **kv_producer**: For prefiller instances that generate KV caches
- **kv_consumer**: For decoder instances that consume KV caches from prefiller
- **kv_both**: Enables symmetric functionality where the connector can act as both producer and consumer. This provides flexibility for experimental setups and scenarios where the role distinction is not predetermined.

## Example Scripts/Code

Refer to these example scripts in the vLLM repository:

- `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh`
- `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`
- `tests/v1/kv_connector/nixl_integration/test_accuracy.py`
