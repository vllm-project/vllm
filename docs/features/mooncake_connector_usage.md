# MooncakeConnector Usage Guide

## About Mooncake

Mooncake aims to enhance the inference efficiency of large language models (LLMs), especially in slow object storage environments, by constructing a multi-level caching pool on high-speed interconnected DRAM/SSD resources. Compared to traditional caching systems, Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine.

For more details about Mooncake, please refer to [Mooncake project](https://github.com/kvcache-ai/Mooncake) and [Mooncake documents](https://kvcache-ai.github.io/Mooncake/).

## Prerequisites

### Installation

Install mooncake through pip: `uv pip install mooncake-transfer-engine`.

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
python examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
```

Now you can send requests to the proxy server through port 8000.

## Environment Variables

- `VLLM_MOONCAKE_BOOTSTRAP_PORT`: Port for Mooncake bootstrap server
    - Default: 8998
    - Required only for prefiller instances
    - For headless instances, must be the same as the master instance
    - Each instance needs a unique port on its host; using the same port number across different hosts is fine

- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`: Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request. (Optional)
    - Default: 480
    - If a request is aborted and the decoder has not yet notified the prefiller, the prefill instance will release its KV-cache blocks after this timeout to avoid holding them indefinitely.

## KV Transfer Config

### KV Role Options

- **kv_producer**: For prefiller instances that generate KV caches
- **kv_consumer**: For decoder instances that consume KV caches from prefiller
- **kv_both**: Enables symmetric functionality where the connector can act as both producer and consumer. This provides flexibility for experimental setups and scenarios where the role distinction is not predetermined.

### kv_connector_extra_config

- **num_workers**: Size of thread pool for one prefiller worker to transfer KV caches by mooncake. (default 10)
- **mooncake_protocol**: Mooncake connector protocol. (default "rdma")

## Example Scripts/Code

Refer to these example scripts in the vLLM repository:

- [run_mooncake_connector.sh](../../examples/online_serving/disaggregated_serving/mooncake_connector/run_mooncake_connector.sh)
- [mooncake_connector_proxy.py](../../examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py)
