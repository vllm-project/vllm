# MoRIIOConnector Usage Guide

`MoRIIOConnector` is a high-performance KV connector used for KV cache transfer in PD disaggregated deployments, built on ROCm's [MoRI-IO](https://github.com/rocm/mori) communication library for point-to-point communication with ultra-low overhead.

## Prerequisites

### Installation

**Docker:** MoRI is shipped with the official ROCm vLLM image: `vllm/vllm-openai-rocm:nightly`.

**Manual installation:** MoRI wheel can be installed with

```bash
pip install amd_mori
```

Refer to the [Dockerfile.rocm_base](../../docker/Dockerfile.rocm_base) for more information, or [official MoRI repository](https://github.com/rocm/mori) for instructions on how to build MoRI from source.

For instructions on installing appropriate NIC userspace libraries, see [Installing NIC userspace libraries](#appendix-installing-nic-userspace-libraries).

## Basic usage (single host)

Start the proxy first; the producer and consumer instances will retry registration until the proxy is reachable.

The `127.0.0.1` example below is for local-only runs where the proxy, prefiller, and decoder share the same network namespace, such as three bare-metal processes on one node or all three processes in one container. For separate containers, use a reachable node/container IP for `proxy_ip` and usually leave `host_ip` unset.

Because the prefiller and decoder share one network namespace in this example, they use distinct `handshake_port` and `notify_port` values. MoRIIO opens one port per DP/TP rank, increasing from the configured base port with `base + dp_rank * tp_size + tp_rank`. For `-tp 4`, `handshake_port: 6301` uses ports `6301-6304` and `notify_port: 6105` uses `6105-6108`; the decoder uses separate ranges, `7301-7304` and `7501-7504`.

### Producer (prefiller) configuration

Start a prefiller instance that produces KV caches

```bash
# Prefill instance (GPU 0-3)
export VLLM_ROCM_USE_AITER=1
export HIP_VISIBLE_DEVICES=0,1,2,3

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --port 20005 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "host_ip": "127.0.0.1",
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
export HIP_VISIBLE_DEVICES=4,5,6,7

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --port 40005 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "host_ip": "127.0.0.1",
      "http_port": "40005",
      "proxy_ping_port": "36367",
      "handshake_port": "7301",
      "notify_port": "7501"
    }
  }'
```

### Proxy server

The proxy fronts the producer and consumer instances and routes incoming requests to them. `vllm-router` is the recommended proxy; it can be installed manually or run as a Docker container. Note that the port `36367` below is the `proxy_ping_port` configured on each vLLM instance.

**Docker:**

```bash
docker run \
  --network host \
  vllm/vllm-router:nightly \
  vllm-router \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:36367"
```

**Manual install:**

```bash
pip install vllm-router
vllm-router \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:36367"
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

**Control-plane configuration:** MoRI moves KV bytes over RDMA/xGMI, but producers and consumers also need out-of-band TCP channels for handshake, block id exchange, liveness, and completion signaling. These keys live under `kv_connector_extra_config`:

- `proxy_ip`: IP address of the disaggregation proxy/router. For normal multi-container or multi-node deployments, use a real node/container IP reachable from the vLLM instances. `127.0.0.1` is only for local tests where the proxy and vLLM instances share the same network namespace.
- `host_ip`: Optional local IP address advertised by this vLLM instance. In most deployments, leave this unset. vLLM will infer it from the route to `proxy_ip`. Set it explicitly only when you need to force a specific address, such as local-only `127.0.0.1` tests.
- `proxy_ping_port`: TCP port on `proxy_ip` where the proxy listens for instance heartbeats and registration messages. It must match the router's `--vllm-discovery-address` port.
- `http_port`: HTTP port that this vLLM instance exposes its OpenAI-compatible API on. It must match this instance's `vllm serve --port`; vLLM does not infer it from the CLI argument.
- `handshake_port`: TCP port used for the one-time MoRI engine handshake between a prefiller and a decoder. The two sides exchange RDMA engine descriptors here before any KV transfer can happen.
- `notify_port`: TCP port used for control and synchronization messages between prefiller and decoder. Used differently in the two modes:
    - WRITE mode: **Block allocation:** the decoder notifies the prefiller about its block ids, so the prefiller can push its computed KV blocks into the correct place on the decoder instance. **Completion:** once all blocks have been transferred, the prefiller notifies the decoder that it's safe to use its blocks.
    - READ mode: **Completion:** once the decoder has read all blocks from the prefiller, it notifies the prefiller so it can free its KV cache blocks.

!!! note
    `handshake_port` and `notify_port` are *base* ports. Each (DP rank, TP rank) pair within an instance uses `base + dp_rank * tp_size + tp_rank`, so a TP4 instance needs four monotonically increasing ports for each base. Make sure both ranges are free on the host.

If `host_ip` is unset and the route to `proxy_ip` uses loopback, vLLM warns and falls back to `get_ip()`. If auto configuration still resolves to loopback or `0.0.0.0`, startup fails and `host_ip` must be set explicitly.

If `MORI_SOCKET_IFNAME` is unset, vLLM sets it to the interface that owns the selected `host_ip`. If it is already set, vLLM does not override it.

For separate Docker containers without `--network host`, put the containers on the same Docker network and set `proxy_ip` to the proxy container's reachable address:

- Create a user-defined network:

```bash
docker network create moriio-net
```

- Start the proxy container on that network with a stable name, for example `moriio-proxy`.

- Use the proxy container name or IP as `proxy_ip` and omit `host_ip`:

```json
"kv_connector_extra_config": {
  "proxy_ip": "moriio-proxy",
  "proxy_ping_port": "36367",
  "http_port": "8100",
  "handshake_port": "6301",
  "notify_port": "61005"
}
```

- To use the container IP directly, find it with:

```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' moriio-proxy
```

### Transport configuration

MoRI has two transport backends: RDMA and xGMI. You can select backend using `--kv-transfer-config.kv_connector_extra_config.backend $BACKEND`, with `$BACKEND` being `rdma` or `xgmi`. RDMA is the default backend and should be used in multi-node deployments.

The configuration options for each backend are as follows.

#### RDMA backend

- `qp_per_transfer`: number of RDMA Queue Pairs (QPs) used per transfer. More QPs let a single transfer be striped over multiple QPs to increase NIC concurrency, at the cost of more RDMA resources.
- `post_batch_size`: how many RDMA Work Requests (WR) are batched into one `ibv_post_send` doorbell. Defaults to -1, meaning the backend default. Larger batches reduce the posting overhead per WR.
- `num_workers`: number of worker threads MoRI uses to post and poll transfer completions.

Advanced users can also configure MoRI itself using environment variables such as `MORI_IO_QP_MAX_SEND_WR`, `MORI_IO_QP_MAX_CQE`, etc. These are MoRI library variables and are separate from vLLM's own `VLLM_MORIIO_*` settings. Refer to the [MoRI repository](https://github.com/rocm/mori) for more information.

`MORI_RDMA_DEVICES` and `MORI_IB_GID_INDEX` are optional MoRI environment variables. If unset, MoRI uses its own defaults. vLLM does not infer or modify them. On multi-NIC hosts, set `MORI_RDMA_DEVICES` explicitly if the default active-device set includes a NIC that should not carry KV RDMA traffic; otherwise MoRI may choose that NIC and fail with RDMA registration or QP connection errors.

#### xGMI backend

Use xGMI when the prefiller and decoder run on the same physical host so transfers go over AMD Infinity Fabric/xGMI and skip the NIC entirely. Currently only configured using MoRI-specific environment variables; see the [MoRI repository](https://github.com/rocm/mori).

## Multi-node deployment

### Docker container flags for RDMA

When running RDMA inside the ROCm vLLM container, launch the prefill and decode containers with host networking plus GPU/RDMA device access. The important container settings are:

- `--network host`: lets vLLM infer `host_ip` and `MORI_SOCKET_IFNAME` from host routes.
- `--device /dev/kfd --device /dev/dri`: exposes AMD GPU devices.
- `--device /dev/infiniband`: exposes RDMA verbs devices.
- `--ulimit memlock=-1`: allows RDMA memory registration.

```bash
docker run \
  --init --network host --ipc host --privileged \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 256G \
  --group-add video --group-add render \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -e VLLM_ROCM_USE_AITER=1 \
  vllm/vllm-openai-rocm:nightly \
  <model-and-vllm-args>
```

Use the same MoRIIO configuration shown in the host-network example below. If you do not use `--network host`, set `proxy_ip` to a reachable container name or container IP.

### Host-network MiniMax M3 RDMA examples

The following examples use MiniMax M3 with MoRIIO READ mode over RDMA. The router runs on the prefill node. `proxy_ip` is a real control-plane IP, `host_ip` is omitted, and vLLM infers each instance's advertised IP and `MORI_SOCKET_IFNAME` from the route to the router.

Set shared environment on every node:

```bash
export ROUTER_IP=<prefill-node-control-ip>
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HSA_ENABLE_SDMA=1
```

Set `MORI_RDMA_DEVICES` only when MoRI's default active-device set is not correct for the host:

```bash
# export MORI_RDMA_DEVICES=<comma-separated-rdma-devices>
```

Start the router on the prefill node:

```bash
podman run --rm --network host docker.io/vllm/vllm-router:nightly \
  vllm-router \
    --host 0.0.0.0 \
    --port 30000 \
    --vllm-pd-disaggregation \
    --kv-connector moriio \
    --vllm-discovery-address 0.0.0.0:36367 \
    --policy consistent_hash \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash \
    --log-level info
```

#### 1P1D: TP8 prefill + TP8 decode

Start the TP8 prefill instance on node 1:

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --host 0.0.0.0 \
  --port 8100 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "'"${ROUTER_IP}"'",
      "proxy_ping_port": 36367,
      "http_port": 8100,
      "handshake_port": 6301,
      "notify_port": 61005,
      "read_mode": true,
      "backend": "rdma",
      "qp_per_transfer": 4,
      "num_workers": 4
    }
  }'
```

Start the TP8 decode instance on node 2:

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --host 0.0.0.0 \
  --port 8200 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "'"${ROUTER_IP}"'",
      "proxy_ping_port": 36367,
      "http_port": 8200,
      "handshake_port": 7301,
      "notify_port": 7501,
      "read_mode": true,
      "backend": "rdma",
      "qp_per_transfer": 4,
      "num_workers": 4
    }
  }'
```

#### 2P1D: two TP4 prefills + one TP8 decode

Run each vLLM command in a separate terminal or process. The two prefill instances share node 1, so they use different GPU sets and non-overlapping `handshake_port` / `notify_port` ranges.

This topology uses heterogeneous TP sizes.

Start the first TP4 prefill instance on node 1:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --host 0.0.0.0 \
  --port 8100 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "'"${ROUTER_IP}"'",
      "proxy_ping_port": 36367,
      "http_port": 8100,
      "handshake_port": 6301,
      "notify_port": 61005,
      "read_mode": true,
      "backend": "rdma",
      "qp_per_transfer": 4,
      "num_workers": 4
    }
  }'
```

Start the second TP4 prefill instance on node 1:

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7

vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --host 0.0.0.0 \
  --port 8101 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "'"${ROUTER_IP}"'",
      "proxy_ping_port": 36367,
      "http_port": 8101,
      "handshake_port": 6305,
      "notify_port": 61009,
      "read_mode": true,
      "backend": "rdma",
      "qp_per_transfer": 4,
      "num_workers": 4
    }
  }'
```

Start the TP8 decode instance on node 2:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --host 0.0.0.0 \
  --port 8200 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "'"${ROUTER_IP}"'",
      "proxy_ping_port": 36367,
      "http_port": 8200,
      "handshake_port": 7301,
      "notify_port": 7501,
      "read_mode": true,
      "backend": "rdma",
      "qp_per_transfer": 4,
      "num_workers": 4
    }
  }'
```

For these examples:

- vLLM infers `host_ip` and `MORI_SOCKET_IFNAME`.
- MoRI chooses `MORI_IB_GID_INDEX` when it is unset.
- `proxy_ping_port` must match the port in the router's `--vllm-discovery-address`.
- `http_port` must match this instance's `vllm serve --port`.
- In 2P1D, each prefill on the same host needs a unique `--port` / `http_port` pair and non-overlapping `handshake_port` / `notify_port` ranges.
- You choose `ROUTER_IP`, `MORI_RDMA_DEVICES` if needed, ports, roles, model, and any model/runtime options.
- `read_mode` is set inside `kv_connector_extra_config`; the old `VLLM_MORIIO_CONNECTOR_READ_MODE` environment variable is deprecated.

## Troubleshooting

### `availDevices.size() > 0` assertion failure

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

The official image `vllm/vllm-openai-rocm:nightly` comes pre-installed with userspace libraries for the following NICs and kernel module versions:

- AINIC (AMD Pensando Pollara): version `1.117.3-hydra`, tested with `ioinic-dkms=25.11.1.001`
- Thor2 (Broadcom): version `235.2.86.0`, tested with `bnxt-en-dkms=1.10.3.235.2.86.0`, `bnxt-re-dkms=235.2.86.0`

Refer to [Dockerfile.rocm](../../docker/Dockerfile.rocm) for more details. For users with NICs, kernel modules, and/or FW other than those stated above we refer to
the vendors' own installation instructions.

## Further reading

- [Next-Level Inference: Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation](https://vllm.ai/blog/2026-04-07-moriio-kv-connector).
