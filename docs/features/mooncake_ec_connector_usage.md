# ECMooncakeConnector Usage Guide

`ECMooncakeConnector` moves multimodal **encoder embeddings** (encoder cache)
between a disaggregated encoder instance and a language-model instance over
[Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake). Use it
when the vision encoder should scale independently of prefill and decode.

For KV-cache transfer between prefiller and decoder, see
[MooncakeConnector Usage Guide](mooncake_connector_usage.md). That path uses
`MooncakeConnector`, not `ECMooncakeConnector`.

Added in [PR #41567](https://github.com/vllm-project/vllm/pull/41567).
Overview of disaggregated encoding:
[Disaggregated Encoder](disagg_encoder.md).

!!! important
    Static proxy flags (`--ec-consumer-zmq-addrs`) support Mooncake EC only in
    **1E + 1PD** mode (`--prefill-servers-urls disable`). For 1E + 1P + 1D,
    register `ec_zmq_addrs` on the **prefill** consumer through the proxy's
    dynamic registration API instead of the static CLI flag. See
    `examples/disaggregated/disaggregated_encoder/README.md`.

## Prerequisites

Install Mooncake Transfer Engine:

```bash
# CUDA 13 (vLLM default)
uv pip install mooncake-transfer-engine-cuda13

# CUDA 12
uv pip install mooncake-transfer-engine
```

The two wheels are the same release built against different CUDA majors.
Importing the wrong one fails with
`libcudart.so.<major>: cannot open shared object file`.

## 1E + 1PD

One encoder producer and one combined prefill/decode consumer. The consumer
listens on `ec_ip` + `ec_port` (ZMQ control). The proxy must advertise that
address with `--ec-consumer-zmq-addrs`.

### Encoder (producer)

The producer must run with `tensor_parallel_size=1`,
`pipeline_parallel_size=1`, and `data_parallel_size=1`.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --port 19534 \
  --gpu-memory-utilization 0.35 \
  --enable-request-id-headers \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --allowed-local-media-path /path/to/media \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
      "mooncake_protocol": "rdma"
    }
  }'
```

### Prefill+Decode (consumer)

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --port 19537 \
  --gpu-memory-utilization 0.75 \
  --enable-mm-embeds \
  --enable-request-id-headers \
  --no-enable-prefix-caching \
  --allowed-local-media-path /path/to/media \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_consumer",
    "ec_ip": "127.0.0.1",
    "ec_port": 19019,
    "ec_connector_extra_config": {
      "mooncake_protocol": "rdma"
    }
  }'
```

`--enable-mm-embeds` lets the PD instance inject transferred embeddings
instead of re-running the vision encoder.

### Proxy

```bash
python examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py \
  --host 0.0.0.0 \
  --port 10002 \
  --encode-servers-urls http://127.0.0.1:19534 \
  --prefill-servers-urls disable \
  --decode-servers-urls http://127.0.0.1:19537 \
  --ec-consumer-zmq-addrs tcp://127.0.0.1:19019
```

`--ec-consumer-zmq-addrs` must align with `--decode-servers-urls` (the
combined PD instances). For data-parallel consumers, list each replica
consecutively and set `--ec-consumer-dp-size`.

### Correctness script

```bash
# TCP (no RDMA device required; CI default)
MOONCAKE_EC_PROTOCOL=tcp \
  bash tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh

# RDMA
MOONCAKE_EC_PROTOCOL=rdma \
  bash tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh
```

The TCP path sets `MC_FORCE_TCP=1` so Transfer Engine does not auto-select
RDMA when an HCA is present.

## Configuration

Producer and consumer must use the same `mooncake_protocol`.

| Setting | Where | Default | Notes |
| --------- | ------- | --------- | ------- |
| `mooncake_protocol` | `ec_connector_extra_config` | `rdma` | Set `tcp` when no verbs device is available. With TCP, also export `MC_FORCE_TCP=1`. |
| `ec_ip` / `ec_port` | consumer `ECTransferConfig` | `127.0.0.1` / `14579` | First ZMQ control port. Tensor-parallel rank `r` listens on `ec_port + dp_index * tp_size + r`. |
| `ec_buffer_size` | `ECTransferConfig` | `1e9` | Registered receive arena on the consumer, in bytes. |
| `control_timeout_s` | `ec_connector_extra_config` | `30` | Reservation RPC timeout. |
| `push_wait_timeout_s` | `ec_connector_extra_config` | `60` | How long the consumer waits for a matching encoder push. |
| `WITH_NVIDIA_PEERMEM` | process environment | `1` | Read by Mooncake. Set `0` on hosts without `nvidia-peermem` (DMA-BUF path). |
| `--ec-consumer-zmq-addrs` | proxy | empty | `tcp://<ec_ip>:<ec_port>` for each PD replica. Required for Mooncake EC. |

## 1E + 1P + 1D

The filesystem example is `examples/disaggregated/disaggregated_encoder/disagg_1e1p1d_example.sh`
(`ECExampleConnector` for encoder cache, `NixlConnector` for KV). P→D KV can
use `MooncakeConnector` instead; see
[MooncakeConnector Usage Guide](mooncake_connector_usage.md).

Do **not** pass `--ec-consumer-zmq-addrs` together with independent prefill
URLs: the static proxy flag is E+PD only. For Mooncake EC with a dedicated
prefill, use dynamic registration and set `ec_zmq_addrs` on the prefill
consumer as described in
`examples/disaggregated/disaggregated_encoder/README.md`.

## Troubleshooting

| Symptom | Likely cause |
| --------- | ---------------- |
| `Install mooncake-transfer-engine` / `libcudart.so.*` | Wrong or missing Mooncake wheel for this CUDA major. |
| `Failed to register memory ... Bad address [14]` | GPUDirect registration failed. Set `WITH_NVIDIA_PEERMEM=0`. |
| Proxy startup error about E+PD only | `--ec-consumer-zmq-addrs` used with a real `--prefill-servers-urls`. |
| Consumer waits then errors; encoder never finishes the push | `--ec-consumer-zmq-addrs` does not match consumer `ec_ip`/`ec_port`, or producer/consumer protocols differ. |
| TCP transfers stall on a host that also has RDMA | Export `MC_FORCE_TCP=1`. |
| Producer rejected at startup | Encoder used tensor, pipeline, or data parallelism. Keep the producer unsharded. |
