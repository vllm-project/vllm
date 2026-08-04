# Benchmarking CacheBlend with Muti-Doc QA

## Overview
The benchmark contains two request rounds. The first round (warmup round) sends each document as a single prompt. The second round randomly samples a certain number of preprocessed documents and concatenate them together for each request.

## Run the benchmarking

### Step 1: Start the serving engine

**Baseline1: vLLM**
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000
```

**Baseline2: vLLM + vanilla LMCache**

```bash
LMCACHE_CONFIG_FILE=lmcache.yaml vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

**vLLM + LMCache with blending**

```bash
LMCACHE_CONFIG_FILE=lmcache_blend.yaml vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --no-enable-prefix-caching --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

### Step 2: Send the requests
```bash
python multi_doc_qa.py --num-total-documents 100 --document-length 3000 --output-len 1 --num-requests 100 --num-docs-per-request 5 --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --max-inflight-requests 1 
```