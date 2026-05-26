# MoRIIOConnector Usage Guide

`MoRIIOConnector` is a high-performance KV connector used for KV cache transfer in PD disaggregated deployments, designed specifically for the ROCm platform. It is built using ROCm's [MoRI-IO](https://github.com/rocm/mori) communication library for point-to-point communication with ultra-low overhead, part of the broader MoRI (Modular RDMA Interface) framework.

> [!NOTE]
> MoRI is a bottom-up, modular, and composable framework for building high-performance communication applications with a strong focus on RDMA + GPU integration. Inspired by the role of MLIR in compiler infrastructure, MORI provides reusable and extensible building blocks that make it easier for developers to adopt advanced techniques such as IBGDA (Infiniband GPUDirect Async) and GDS (GPUDirect Storage).

## Prerequisites

### Installation

**Docker:** MoRI is shipped with the official ROCm vLLM image: `vllm/vllm-openai-rocm:v0.21.0` (and later).

**Manual installation:** MoRI wheel can be installed with

```bash
pip install amd_mori
```

Refer to the [Dockerfile.rocm_base](../../docker/Dockerfile.rocm_base) for more information, or [official MoRI repository](https://github.com/rocm/mori) for instructions on how to build MoRI from source.

For instructions on installing appropriate NIC userspace libraries, see [Installing NIC userspace libraries](#appendix-installing-nic-userspace-libraries).

## Basic usage (single host)

Start the proxy first; the producer and consumer instances will retry registration until the proxy is reachable.

### Producer (prefiller) configuration

Start a prefiller instance that produces KV caches

```bash
# Prefill instance (GPU 0-3) 
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
 
vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --port 20005 \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "proxy_ping_port": "36367",
      "http_port": "20005",
      "handshake_port": "6301",
      "notify_port": "6105"
    }
  }'
```

### Consumer (decoder) configuration

Start a decoder instance that consumes KV caches:

```bash
# Decode instance (GPU 4-7)
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HIP_VISIBLE_DEVICES=4,5,6,7

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --port 40005 \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "http_port": "40005",
      "proxy_ping_port": "36367",
      "handshake_port": "7301",
      "notify_port": "7501"
    }
  }'
```

### Proxy server

The proxy fronts the producer and consumer instances and routes incoming requests to them. `vllm-router` is the recommended proxy; it can be installed manually or run as a Docker container. Note that the port `36367` below is the `proxy_ping_port` configured on each vLLM instance.

**Manual install:**

```bash
pip install vllm-router
vllm-router \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:36367" \
```

**Docker:**

```bash
docker run \
  --network host \
  vllm/vllm-router:nightly \
  vllm-router \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:36367" \
```

Alternatively, you can use the reference implementation proxy shipped with vLLM:

```bash
cd <path_to>/vllm
pip install quart aiohttp msgpack
python examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py
```

## Configuration

The connector is configured at two levels: the application level and the transport level.

### Application-level configuration

**Modes:** MoRI has two modes of operation: WRITE and READ mode.

- In WRITE mode, the producer actively pushes computed KV blocks after every layer into the consumer's memory.
- In READ mode, the consumer pulls the KV blocks from the producer all at once, as soon as it has been notified those blocks are ready.

WRITE mode is used by default. READ mode can be configured by setting `--kv-transfer-config.kv_connector_extra_config.read_mode true`.

**Control-plane configuration:** MoRI moves the actual KV bytes over an RDMA (or xGMI) fast-path. The producer and consumer instances still need a set of out-of-band TCP channels to exchange block ids, RDMA queue-pair metadata, report liveness, and signal transfer completions.

The following keys are used to configure this metadata exchange, all part of the `kv_connector_extra_config`:

- `proxy_ip`: IP address of the disaggregation proxy/router that fronts the prefiller and decoder. Each vLLM instance uses it to register itself and to send heartbeats so the proxy knows where to route incoming requests.
- `proxy_ping_port`: TCP port on `proxy_ip` where the proxy listens for instance heartbeats and registration messages. Used to detect dead vLLM instances and keep routing tables fresh.
- `http_port`: HTTP port that this vLLM instance exposes its OpenAI-compatible API on. The proxy registers this port, and forwards user requests to this port once it has picked an instance.
- `handshake_port`: TCP port used for the one-time MoRI engine handshake between a prefiller and a decoder. The two sides exchange RDMA engine descriptors here before any KV transfer can happen.
- `notify_port`: TCP port used for control and synchronization messages between prefiller and decoder. Used differently in the two modes:
    - WRITE mode:
        - **Block allocation:** the decoder notifies the prefiller about its block ids, so the prefiller can push its computed KV blocks into the correct place on the decoder instance.
        - **Completion:** once all blocks have been transferred, the prefiller notifies the decoder that it's safe to use its blocks.
    - READ mode:
        - **Completion:** once the decoder has read all blocks from the prefiller, it notifies the prefiller so it can free its KV cache blocks.

> [!NOTE]
> `notify_port` is used as a *base* port: each (DP rank, TP rank) pair within an instance uses `notify_port + offset` where the offset is based on the rank. Make sure the range starting at `notify_port` is free on the host.

### Transport configuration

MoRI has two transport backends: RDMA and xGMI. You can select backend using `--kv-transfer-config.kv_connector_extra_config.backend $BACKEND`, with `$BACKEND` being `rdma` or `xgmi`. RDMA is the default backend and should be used in multi-node deployments.

The configuration options for each backend are as follows.

#### RDMA backend

- `qp_per_transfer`: number of RDMA Queue Pairs (QPs) used per transfer. More QPs let a single transfer be striped over multiple QPs to increase NIC
  concurrency, at the cost of more RDMA resources.
- `post_batch_size`: how many RDMA Work Requests (WR) are batched into one `ibv_post_send` doorbell. Defaults to -1, meaning the backend default. Larger batches
  reduce the posting overhead per WR, whereas smaller batches can reduce latency for small transfers.
- `num_workers`: number of worker threads MoRI uses to post and poll transfer completions.

> [!TIP]
> Example: to set the number of QPs per transfer to 4, use the flag `--kv-transfer-config.kv_connector_extra_config.qp_per_transfer 4`. All other flags are
set analogously.

Advanced users can also configure MoRI itself using environment variables such as `MORI_IO_QP_MAX_SEND_WR`, `MORI_IO_QP_MAX_CQE`, etc. These are MoRI library variables and are separate from vLLM's own `VLLM_MORIIO_*` settings. Refer to the [MoRI repository](https://github.com/rocm/mori) for more information.

#### xGMI backend

Use xGMI when the prefiller and decoder run on the same physical host so transfers go over the AMD GPU fabric and skip the NIC entirely. Currently only configured using MoRI-specific environment variables; see the [MoRI repository](https://github.com/rocm/mori).

## Multi-node deployment

The example below shows how to run a 1P1D deployment on two nodes. We run the proxy on the same node as the prefill instance.

### On both nodes

```bash
# Set on both nodes before running any command
export PREFILL_IP=<node1-ip>
export DECODE_IP=<node2-ip>

# Adjust to the image tag you want to run
export VLLM_IMAGE=vllm/vllm-openai-rocm:v0.21.0
```

### On node 1

Start the proxy first, then the prefill instance.

Proxy:

```bash
docker run \
  --network host \
  --rm \
  vllm/vllm-router:nightly \
  vllm-router \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:36367" \
```

Prefill instance:

```bash
docker run \
  --name moriio-prefill \
  --init --network host --ipc host --privileged \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size 256G \
  --group-add video --group-add render \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v /sys:/sys \
  -e VLLM_ROCM_USE_AITER=1 \
  $VLLM_IMAGE \
  deepseek-ai/DeepSeek-R1-0528 \
    --port 8100 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config '{
      "kv_connector": "MoRIIOConnector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {
        "proxy_ip": "'"${PREFILL_IP}"'",
        "proxy_ping_port": "36367",
        "http_port": "8100",
        "handshake_port": "6301",
        "notify_port": "61005"
      }
    }'
```

### On node 2

Decode instance:

```bash
docker run \
  --name moriio-decode \
  --init --network host --ipc host --privileged \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size 256G \
  --group-add video --group-add render \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v /sys:/sys \
  -e VLLM_ROCM_USE_AITER=1 \
  $VLLM_IMAGE \
  deepseek-ai/DeepSeek-R1-0528 \
    --port 8200 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enable-expert-parallel \
    --kv-transfer-config '{
      "kv_connector": "MoRIIOConnector",
      "kv_role": "kv_consumer",
      "kv_connector_extra_config": {
        "proxy_ip": "'"${PREFILL_IP}"'",
        "proxy_ping_port": "36367",
        "http_port": "8200",
        "handshake_port": "6301",
        "notify_port": "61005"
      }
    }'
```

## Troubleshooting

### `availDevices.size() > 0` assertion failure — incompatible NIC userspace libraries

**Problem:** vLLM fails to launch with the following log:

```bash
libibverbs: Warning: Driver bnxt_re does not support the kernel ABI of 6 (supports 1 to 1) for device /sys/class/infiniband/rdma4
...
ker: /app/mori/src/io/rdma/backend_impl.cpp: mori::io::RdmaManager::RdmaManager(const RdmaBackendConfig, application::RdmaContext *): Assertion `availDevices.size() > 0' failed.
```

**Fix:** The installed RDMA userspace libraries do not match the driver and firmware version installed on the host. You must install NIC userspace libraries corresponding to your RDMA kernel module and firmware version. See [Installing NIC userspace
libraries](#appendix-installing-nic-userspace-libraries) for more information.

## Appendix: installing NIC userspace libraries

To run MoRI with RDMA, your environment must have the necessary RDMA userspace libraries installed that match the associated kernel module and firmware version.

The official image `vllm/vllm-openai-rocm:v0.21.0` (and later) comes pre-installed with userspace libraries for the following NICs and kernel module versions:

- AINIC (AMD Pensando Pollara): version `1.117.3-hydra`, tested with `ioinic-dkms=25.11.1.001`
- Thor2 (Broadcom): version `235.2.86.0`, tested with `bnxt-en-dkms=1.10.3.235.2.86.0`, `bnxt-re-dkms=235.2.86.0`

Refer to [Dockerfile.rocm](../../docker/Dockerfile.rocm) for more details. For users with NICs, kernel modules, and/or FW other than those stated above we refer to
the vendors' own installation instructions.

## Further reading

- [Next-Level Inference: Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation](https://vllm.ai/blog/2026-04-07-moriio-kv-connector).
