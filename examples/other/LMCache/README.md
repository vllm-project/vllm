
# LMCache Disaggregated Prefill Example

This folder demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node.

## Prerequisites

- Install [LMCache](https://github.com/ai-dynamo/lmcache)
- Install [NIXL](https://github.com/ai-dynamo/nixl) 
- At least 2 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

## Usage

Simply run

```bash
bash disagg_example_nixl.sh
```

to run disaggregated prefill and benchmark the performance.

## Components

### Server Scripts
- `disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder

### Configuration
- `configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server
- `configs/lmcache-decoder-config.yaml` - Configuration for decoder server

### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server
- `benchmark.log` - Performance benchmark logs and results

