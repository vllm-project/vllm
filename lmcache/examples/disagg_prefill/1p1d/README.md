## Example of Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node.

### Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 2 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Optionally set the visible devices for prefill and decoder instances through environment variable.
By default they are set to 0 and 1 respectively.

```bash
export PREFILLER_DEVICE_ID="1"
export DECODER_DEVICE_ID="0"
```

Run
```bash
bash disagg_example_1p1d.sh
```

to start disaggregated prefill and benchmark the performance.

The script will:

1. Launch 1 decoder instances listening on port 7200.
2. Launch 1 prefill instances listening on ports 7100.
3. Launch a proxy server, listening on port 9100

Press `Ctrl+C` to stop the servers.

#### Example benchmark command

If you have vLLM's serving benchmark tool, you can run the following command to benchmark the serving performance of the disaggregated prefill setup:

```bash
vllm bench serve --port 9100 --seed $(date +%s) \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos
```

Expected output from the benchmark script:

```plaintext
============ Serving Benchmark Result ============
Successful requests:                     30
Benchmark duration (s):                  31.34
Total input tokens:                      224970
Total generated tokens:                  6000
Request throughput (req/s):              0.96
Output token throughput (tok/s):         191.44
Total Token throughput (tok/s):          7369.36
---------------Time to First Token----------------
Mean TTFT (ms):                          313.41
Median TTFT (ms):                        272.83
P99 TTFT (ms):                           837.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.84
Median TPOT (ms):                        8.72
P99 TPOT (ms):                           11.35
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.84
Median ITL (ms):                         8.61
P99 ITL (ms):                            11.43
==================================================
```

### Components

#### Server Scripts
- `disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_example_1p1d.sh` - Main script to run the example

#### Configuration
- `configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server
- `configs/lmcache-decoder-config.yaml` - Configuration for decoder server

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server
