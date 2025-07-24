# LMCache Examples
Please Note: HPU integration for LMCache will be upstreamed. After that, the following test cases can be used.

This folder demonstrates how to use LMCache for disaggregated prefilling, CPU offloading and KV cache sharing.

## 1. Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using lm or redis on a single node.

### Prerequisites
- At least 2 HPU cards
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct
- https://github.com/LMCache/LMCache/pull/1066 needed for lmcache

### Usage

Run
`cd disagg_prefill_lmcache_v1`
to get into `disagg_prefill_lmcache_v1` folder, and then run

```bash
PT_HPU_GPU_MIGRATION=1 VLLM_USE_V1=1 VLLM_SKIP_WARMUP=True PT_HPU_ENABLE_LAZY_COLLECTIVES=true bash disagg_example.sh
```

to run disaggregated prefill and benchmark the performance.

lmserver is default and it's configurable as well as tensor_parallel_size and model name.

Example) redis server, tensor_parallel_size 4 and Llama-3.1-70B-Instruct model

```
PT_HPU_GPU_MIGRATION=1 VLLM_USE_V1=1 VLLM_SKIP_WARMUP=True PT_HPU_ENABLE_LAZY_COLLECTIVES=true bash disagg_example.sh -s redis -t 4 -m meta-llama/Llama-3.1-70B-Instruct
```

### Components

#### Server Scripts
- `disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `../disagg_prefill_lmcache_v1/disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_prefill_lmcache_v1/disagg_example.sh` - Main script to run the example through lm/redis remote server

#### Configuration
- `disagg_prefill_lmcache_v1/configs/lmcache-config-lm.yaml` - Configuration for prefiller/decoder server through lm server
- `disagg_prefill_lmcache_v1/configs/lmcache-config-redis.yaml` - Configuration for prefill/decoder server through redis server

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server

## 2. KV Cache Sharing

The `kv_cache_sharing_lmcache_v1.py` example demonstrates how to share KV caches between vLLM v1 instances.

### Usage

```bash
PT_HPU_GPU_MIGRATION=1 VLLM_USE_V1=1 VLLM_SKIP_WARMUP=True PT_HPU_ENABLE_LAZY_COLLECTIVES=true python kv_cache_sharing_lmcache_v1.py
```

lmserver is default and it's configurable as well as tensor_parallel_size.

Example 1) redis server with port 6380

```bash
PT_HPU_GPU_MIGRATION=1 VLLM_USE_V1=1 VLLM_SKIP_WARMUP=True PT_HPU_ENABLE_LAZY_COLLECTIVES=true python kv_cache_sharing_lmcache_v1.py --remote_server redis --redis_port 6380
```

Example 2) lmserver with port 8108 and tensor_parallel_size 2

```bash
PT_HPU_GPU_MIGRATION=1 VLLM_USE_V1=1 VLLM_SKIP_WARMUP=True PT_HPU_ENABLE_LAZY_COLLECTIVES=true python kv_cache_sharing_lmcache_v1.py --lm_port 8108 --tp_size 2
```
