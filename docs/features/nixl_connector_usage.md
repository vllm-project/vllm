# NixlConnector Usage Guide

NixlConnector is a high-performance KV cache transfer connector for vLLM's disaggregated prefilling feature. It provides fully asynchronous send/receive operations using the NIXL library for efficient cross-process KV cache transfer.

## Prerequisites

### Installation

Install the NIXL library: `uv pip install nixl`, as a quick start on Nvidia platform.

- Refer to [NIXL official repository](https://github.com/ai-dynamo/nixl) for more installation instructions
- The specified required NIXL version can be found in [requirements/kv_connectors.txt](../../requirements/kv_connectors.txt) and other relevant config files

For ROCm platform, the [base ROCm docker file](../../docker/Dockerfile.rocm_base) includes RIXL and ucx already.

- Refer to [RIXL official repository](https://github.com/rocm/rixl) for more information
- The supportive libraries for RIXL can be found in [requirements/kv_connectors_rocm.txt](../../requirements/kv_connectors_rocm.txt)
- In the future we may remove RIXL from docker image file and users will be able to install from pre-compiled binary packages

For non-cuda platform, please install nixl with ucx build from source, instructed as below.

```bash
python tools/install_nixl_from_source_ubuntu.py
```

### Transport Configuration

NixlConnector uses NIXL library for underlying communication, which supports multiple transport backends. UCX (Unified Communication X) is the primary default transport library used by NIXL. Configure transport environment variables:

```bash
# Example UCX configuration, adjust according to your environment
export UCX_TLS=all  # or specify specific transports like "rc,ud,sm,^cuda_ipc" ..etc
export UCX_NET_DEVICES=all  # or specify network devices like "mlx5_0:1,mlx5_1:1"
```

!!! tip
    When using UCX as the transport backend, NCCL environment variables (like `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) are not applicable to NixlConnector, so configure UCX-specific environment variables instead of NCCL variables.

#### Selecting a NIXL transport backend (plugin)

NixlConnector can use different NIXL transport backends (plugins). By default, NixlConnector uses UCX as the transport backend.

To select a different backend, set `kv_connector_extra_config.backends` in `--kv-transfer-config`.

### Example: using LIBFABRIC backend

```bash
vllm serve <MODEL> \
  --kv-transfer-config '{
    "kv_connector":"NixlConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{"backends":["LIBFABRIC"]}
  }'
```

You can also pass JSON keys individually using dotted arguments, and you can append list elements using `+`:

```bash
vllm serve <MODEL> \
  --kv-transfer-config.kv_connector NixlConnector \
  --kv-transfer-config.kv_role kv_both \
  --kv-transfer-config.kv_connector_extra_config.backends+ LIBFABRIC
```

!!! note
    Backend availability depends on how NIXL was built and what plugins are present in your environment. Refer to the [NIXL repository](https://github.com/ai-dynamo/nixl) for available backends and build instructions.

## Basic Usage (on the same host)

### Producer (Prefiller) Configuration

Start a prefiller instance that produces KV caches

```bash
# 1st GPU as prefiller
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B \
  --port 8100 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'
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
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'
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
    - For TP/DP deployments, each worker's port on a node is computed as: base_port + dp_rank (e.g., with `--data-parallel-size=2` and base_port=5600, dp_rank 0..1 use port 5600, 5601 on that node).
    - Used for the initial NIXL handshake between the prefiller and the decoder

- `VLLM_NIXL_SIDE_CHANNEL_HOST`: Host for side channel communication
    - Default: "localhost"
    - Set when prefiller and decoder are on different machines
    - Connection info is passed via KVTransferParams from prefiller to decoder for handshake

- `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`: Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request. (Optional)
    - Default: 480
    - If a request is aborted and the decoder has not yet read the KV-cache blocks through the nixl channel, the prefill instance will release its KV-cache blocks after this timeout to avoid holding them indefinitely.

## Multi-Instance Setup

### Multiple Prefiller Instances on Different Machines

```bash
# Prefiller 1 on Machine A (example IP: ${IP1})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP1} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# Prefiller 2 on Machine B (example IP: ${IP2})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP2} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

### Multiple Decoder Instances on Different Machines

```bash
# Decoder 1 on Machine C (example IP: ${IP3})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP3} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# Decoder 2 on Machine D (example IP: ${IP4})
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP4} \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
vllm serve Qwen/Qwen3-0.6B --port 8000 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
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

For multi-host DP deployment, only need to provide the host/port of the head instances.

### KV Role Options

- **kv_producer**: For prefiller instances that generate KV caches
- **kv_consumer**: For decoder instances that consume KV caches from prefiller
- **kv_both**: Enables symmetric functionality where the connector can act as both producer and consumer. This provides flexibility for experimental setups and scenarios where the role distinction is not predetermined.

!!! tip
    NixlConnector currently does not distinguish `kv_role`; the actual prefiller/decoder roles are determined by the upper-level proxy (e.g., `toy_proxy_server.py` using `--prefiller-hosts` and `--decoder-hosts`).
    Therefore, `kv_role` in `--kv-transfer-config` is effectively a placeholder and does not affect NixlConnector's behavior.

### KV Load Failure Policy

The `kv_load_failure_policy` setting controls how the system handles failures when the decoder instance loads KV cache blocks from the prefiller instance:

- **fail** (recommended): Immediately fail the request with an error when KV load fails. This prevents performance degradation by avoiding recomputation of prefill work on the decode instance.
- **recompute** (default): Recompute failed blocks locally on the decode instance. This may cause performance _jitter_ on decode instances as the scheduled prefill will delay and interfere with other decodes. Furthermore, decode instances are typically configured with low-latency optimizations.

!!! warning
    Using `kv_load_failure_policy="recompute"` can lead to performance degradation in production deployments. When KV loads fail, the decode instance will execute prefill work with decode-optimized configurations, which is inefficient and defeats the purpose of disaggregated prefilling. This also increases tail latency for other ongoing decode requests.

## Experimental Feature

### Heterogeneous KV Layout support

Support use case: Prefill with 'HND' and decode with 'NHD' with experimental configuration

```bash
--kv-transfer-config '{..., "enable_permute_local_kv":"True"}'
```

## Example Scripts/Code

Refer to these example scripts in the vLLM repository:

- [run_accuracy_test.sh](../../tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh)
- [toy_proxy_server.py](../../tests/v1/kv_connector/nixl_integration/toy_proxy_server.py)
- [test_accuracy.py](../../tests/v1/kv_connector/nixl_integration/test_accuracy.py)
