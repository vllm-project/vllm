# MooncakeConnector Usage Guide

## About Mooncake

Mooncake aims to enhance the inference efficiency of large language models (LLMs), especially in slow object storage environments, by constructing a multi-level caching pool on high-speed interconnected DRAM/SSD resources. Compared to traditional caching systems, Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine.

For more details about Mooncake, please refer to [Mooncake project](https://github.com/kvcache-ai/Mooncake) and [Mooncake documents](https://kvcache-ai.github.io/Mooncake/).

## Prerequisites

### Installation

Install mooncake through pip: `uv pip install mooncake-transfer-engine-cuda13`.

vLLM defaults to CUDA 13. On a CUDA 12 environment install `mooncake-transfer-engine` instead — the two are the same release built against different CUDA majors, and the wrong one fails to import with `libcudart.so.<major>: cannot open shared object file`.

If you observe PD transfer data mismatches (`dst != src`) with the CUDA 13
package, enable the mitigations in
[Transfer reliability](#transfer-reliability) below (see
[vllm #42395](https://github.com/vllm-project/vllm/issues/42395),
[Mooncake #2086](https://github.com/kvcache-ai/Mooncake/issues/2086)).

Refer to [Mooncake official repository](https://github.com/kvcache-ai/Mooncake) for more installation instructions

## Usage

### Prefiller Node (192.168.0.2)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

### Decoder Node (192.168.0.3)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

### Proxy

```bash
python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
```

Now you can send requests to the proxy server through port 8000.

## Environment Variables

- `VLLM_MOONCAKE_BOOTSTRAP_PORT`: Port for Mooncake bootstrap server
    - Default: 8998
    - Required only for prefiller instances
    - For headless instances, must be the same as the master instance
    - Each instance needs a unique port on its host; using the same port number across different hosts is fine

- `WITH_NVIDIA_PEERMEM`: Selects how mooncake registers GPU memory for RDMA. Read by mooncake, not vLLM.
    - Default: 1, which uses `ibv_reg_mr()` and requires the `nvidia-peermem` kernel module to be loaded
    - Set to 0 to use the DMA-BUF path, which does not need that module. Required on hosts where `nvidia-peermem` is not loaded, such as GB200
    - With the container image, pass it at run time: `docker run -e WITH_NVIDIA_PEERMEM=0 ...`
    - Symptom when left unset on such a host: `Failed to register memory <addr>: Bad address [14]` from `rdma_context.cpp`, and KV transfers fail

- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`: Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request. (Optional)
    - Default: 480
    - If a request is aborted and the decoder has not yet notified the prefiller, the prefill instance will release its KV-cache blocks after this timeout to avoid holding them indefinitely.

### Transfer reliability

Under concurrent PD load, some Mooncake transfer-engine builds (notably
`mooncake-transfer-engine-cuda13==0.3.10.post2`) can produce destination bytes
that do not match the producer source for very large coalesced descriptors.
vLLM mitigations (env vars or `kv_connector_extra_config` keys):

- `VLLM_MOONCAKE_MAX_TRANSFER_BYTES` / `max_transfer_bytes`: Split any single
  transfer descriptor larger than this size into contiguous chunks (recommended
  starting value: `262144` for multimodal PD workloads).
- `VLLM_MOONCAKE_SYNC_AFTER_TRANSFER` / `sync_after_transfer`: Call
  `torch.cuda.synchronize()` after each Mooncake batch transfer on producer and
  consumer (reduces visibility races at some throughput cost).
- `VLLM_MOONCAKE_VERIFY_TRANSFER_INTEGRITY` / `verify_transfer_integrity`:
  Debug-only SHA-256 check that producer source memory is unchanged after
  transfer (does not verify remote destination bytes).

Example prefill/decode extra config:

```json
{
  "kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "max_transfer_bytes": 262144,
    "sync_after_transfer": true
  }
}
```

## KV Transfer Config

### KV Role Options

- **kv_producer**: For prefiller instances that generate KV caches
- **kv_consumer**: For decoder instances that consume KV caches from prefiller
- **kv_both**: Enables symmetric functionality where the connector can act as both producer and consumer. This provides flexibility for experimental setups and scenarios where the role distinction is not predetermined.

### kv_connector_extra_config

- **num_workers**: Size of thread pool for one prefiller worker to transfer KV caches by mooncake. (default 10)
- **mooncake_protocol**: Mooncake connector protocol. (default "rdma")
- **device_name**: Comma-separated whitelist of RDMA devices (e.g. `"mlx5_0,mlx5_1"`) to restrict topology discovery to. Empty discovers every device. Useful on hosts exposing a mix of InfiniBand and RoCE ports, where both peers must settle on the same link layer.
- **max_transfer_bytes**: Split descriptors larger than this many bytes (see [Transfer reliability](#transfer-reliability))
- **sync_after_transfer**: Synchronize CUDA after each Mooncake batch transfer (default false)
- **verify_transfer_integrity**: Debug SHA-256 check of producer source after transfer (default false)

## Example Scripts/Code

Refer to these example scripts in the vLLM repository:

- [run_mooncake_connector.sh](../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh)
- [mooncake_connector_proxy.py](../../examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py)
