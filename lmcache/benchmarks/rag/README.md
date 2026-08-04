# Benchmarking LLM Performance: RAG Use Case
## Overview

This repository contains benchmarking tools for evaluating the performance of language models in various scenarios. The initial focus of this benchmark is on the RAG (Retrieval-augmented generation) use case. The script `rag.py` simulates RAG workloads, allowing you to analyze the serving engine's throughput and latency.  

### Current Workloads

- **RAG Benchmark**: Simulates a real RAG dataset to evaluate key metrics such as token throughput, average time to first token, and average quality.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Running the RAG Benchmark
To run the RAG benchmark, use launch_lmcache.sh and launch_vllm.sh.  

How to launch:  
After starting the serving engine, run ./launch_lmcache.sh or ./launch_vllm.sh to benchmark LMcache or vllm.  

For launch_lmcache.sh:  
Remember to match KV_STORAGE_SIZE with max_local_cache_size in lmcache config yaml.  
Remember to match KV_CHUNK_SIZE with chunk_size in lmcache config yaml.  

For launch_vllm.sh:  
Remember to change END_INDEX in launch_vllm.sh to the end_index printed by precompute.py in launch_lmcache.sh.  
It should be 150 in the following line(the second number).  
```
"Precompute from 0 to 150 for model mistralai/Mistral-7B-Instruct-v0.2"
```

Use ctrl-C to terminate the benchmark at any time, and the the script will write each request's detailed stats to the output file.  


*Note:* the above command requires there is a serving engine with the `mistralai/Mistral-7B-Instruct-v0.2` model served locally at `http://localhost:8000/v1`. Here's an example command to launch the serving engine:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests
```

Here's an example command to launch the serving engine with LMCache+CacheBlend:  

```bash
LMCACHE_CONFIG_FILE=example_blending.yaml python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.7 --port 8000
```

### What does precompute.py do
If no --end-index provided, it will check kv-storage-size and try to precompute the documents that can be held in this size.  
Used for precomputing some KV cache into storage.  
### Arguments
#### Configure the workload
- `--dataset <str>` The path to the dataset. The format is described in `Dataset format` section.  
- `--start-index <int>` Start from which request in the dataset.
- `--end-index <int>` End before which request in the dataset. If not set, or set to negative value and has precomputation, it will default to the value returned by precompute according to how many requests' KV cache can be held in the given size.  
- `--shuffle` Random shuffle the dataset.  
- `--system-prompt <str>` System prompt before the documents.
- `--query-prompt <str>` Query prompt after the documents and before the question in dataset.
- `--separator <str>` The text used to separate system prompt, documents and query prompt. If enabling blending, should match the blend_separator. If not, should be "".
- `--prompt-build-method <str>` Should be QA or FEW_SHOT, indicating different tasks.
- `--time <int>` The number of seconds as an upper bound for this benchmark. By default no limit.
- `--step-interval <float>` The time interval benchmarking script steps for sending requests.
- `--max-tokens <int>` Maximum number of output tokens for every request.
- `--qps <float>` Query per second. The rate to send requests.
#### Configuring the serving engine connection
- `--model <str>` The model name used by the endpoint.
- `--base-url <str>` The URL endpoint for the language model server.
- `--api-key <str>` API key for the language model server.
#### Configure precompute
To benchmark CacheBlend, we need to precompute the KV cache of documents.  
- `--tokenizer <str>` The tokenizer name. If not provided, by default the same as `--model`.
- `--model-config <str>` The model config name. If not provided, by default the same as `--model`.
- `--kv-storage-size <str>` The size used for KV cache. This will decide how many requests will be sent, because we only precompute KV cache within this limit. The same as max_local_cache_size in LMCache config yaml.
- `--kv-chunk-size <int>` The same as chunk_size in LMCache config yaml.
- `--kv-precision-bit <int>` KV cache precision bit. By default 16 for FP16. Should be a multiple of 8.
#### Configure output
- `--output <str>` The csv file to dump the detailed stats for each query (default = summary.csv)
- `--verbose` Enable verbose logging.

## Benchmark Metrics

- **Throughput**: Request processed per second.  
- **Average TTFT (Time to First Token)**: Average time taken for the model to generate the first token of a response.
- **Average Quality**: Average quality score of generation content.  

## Dataset format
Should be a json file, which is a list of dicts.  
Every item(dict) in the list is one request with the following content.  
```
 {
        "ctxs": [
            {
                "title": "",
                "text": "doc_1"
            },
            {
                "title": "",
                "text": "doc_2"
            },
            {
                "title": "",
                "text": "doc_3"
            }
        ],
        "question": "xxx ?",
        "answers": [
            "yyy"
        ]
    }
```
An example is [CacheBlend musique_s.json](https://github.com/YaoJiayi/CacheBlend/blob/main/inputs/musique_s.json)
