# Moreh-vllm Benchmark

In this directory, We introduce our works and the test result to support high-throughput benchmark.

## Multiprocessing benchmark
- Background:
The benchmark throughput measurements provided by vllm by default operate on a single process and single thread. The problem with single-threaded operation is that, at high throughputs, the thread's processing becomes a bottleneck. Specifically, the results from vllm are not measured in a timely manner, leading to incorrect ITL calculations. This issue ultimately arises due to context switching, even when using asyncio.

- Implementation:
If `--max-connections-per-worker [K]` option is given, M (N / K) processes will be created based on `--num_prompts [N]`. These processes will distribute the total requests as evenly as possible and establish M:1 communication with vllm (or proxy server). Each process is responsible for sending and receiving the requests assigned to it and processing the metrics. After all processes are completed, the final benchmark result is calculated by collecting the metric results of each process.


## Result Trimming
- Background:
To measure the high-throughput of the benchmark, the beginning (where decoding batches gradually increase as they are filled) and the end (where decoding batches gradually decrease as they are completed) must be excluded from the overall benchmark duration.

- Implementation:
Trimming is implemented based on the response metadata for each request collected from the benchmark. The response metadata includes the request transmission time, the first token generation delay, and the inter-token generation delay. Based on this, the generation time of all tokens is reversed, and token information within the user-specified time interval is collected based on that time.


# Test result
The benchmark execution script set the warmup-time to 150 seconds and the cooldown-time to 120 seconds. The trimmed experimental results are listed at the bottom of the benchmark output, allowing us to confirm the following metrics: 1) the benchmark execution time after trimming, 2) the number of tokens generated within the defined time interval, 3) the output token throughput, and 4) the Inter-Token Latency (ITL).
Furthermore, the graph below, which visually compares the difference resulting from the trimming, shows the specific time interval from which token information was aggregated. Additionally, since the width of the bars in the graph represents 1 second, the bar height can be interpreted as tokens per second, which directly represents the output token throughput.

## Test env
- Heimdall (heimdall & gateway)
- PD disaggregation (1P1D)
- External LB enabled on Decode server (8 AsyncLLM processes)

## Test script
```
# Warmup time is set to 150.0s, and Cooldown time is set to 120.0s
vllm bench serve \
    --backend vllm \
    --model "deepseek-ai/DeepSeek-R1" \
    --metric-percentiles "10,25,50,75,90" \
    --percentile-metrics "itl,tps,ttft,e2el" \
    --host "mif-istio.cluster.svc.cluster.local" \
    --port 80 \
    --num-prompts 10800 \
    --max-concurrency 3600 \
    --request-rate 50 \
    --ignore-eos \
    --ready-check-timeout-sec 0 \
    --max-connections-per-worker 432 \
    --dataset-name sharegpt \
    --dataset-path  /app/dataset/ShareGPT_V3_unfiltered_cleaned_split.json\
    --sharegpt-input-len 1000 \
    --sharegpt-output-len 1000 \
    --warmup-time 150.0 \
    --cooldown-time 120.0
```

## Result
```
============ Serving Benchmark Result ============
Number of worker processes:              25
Successful requests:                     10800
Maximum request concurrency:             3600
Request rate configured (RPS):           50.00
Benchmark duration (s):                  591.09
Total input tokens:                      10800000
Total generated tokens:                  10800000
Request throughput (req/s):              18.27
Output token throughput (tok/s):         18271.28
Peak output token throughput (tok/s):    23163.00
Peak concurrent requests:                3864.00
Total Token throughput (tok/s):          36542.55
---------------Time to First Token----------------
Mean TTFT (ms):                          5575.28
Median TTFT (ms):                        5753.45
P10 TTFT (ms):                           1436.68
P25 TTFT (ms):                           2461.82
P50 TTFT (ms):                           5753.45
P75 TTFT (ms):                           6857.50
P90 TTFT (ms):                           10445.36
---------------Inter-token Latency----------------
Mean ITL (ms):                           167.37
Median ITL (ms):                         168.92
P10 ITL (ms):                            128.88
P25 ITL (ms):                            158.62
P50 ITL (ms):                            168.92
P75 ITL (ms):                            177.89
P90 ITL (ms):                            196.06
----------------End-to-end Latency----------------
Mean E2EL (ms):                          172774.31
Median E2EL (ms):                        175687.41
P10 E2EL (ms):                           156867.27
P25 E2EL (ms):                           167408.38
P50 E2EL (ms):                           175687.41
P75 E2EL (ms):                           180192.98
P90 E2EL (ms):                           181481.53
==================================================
tip: install termplotlib and gnuplot to plot the metrics
Serving Benchmark Result after warmup before cooldown
Warm-up Time:                            150.0
Cool-down Time:                          120.0
Total counted tokens at filtering:       6437105
Benchmark duration (s):                  319.96
Total generated tokens:                  6437105
Output token throughput (tok/s):         20118.76
---------------Inter-token Latency----------------
Mean ITL (ms):                           174.31
Median ITL (ms):                         171.26
P10 ITL (ms):                            154.19
P25 ITL (ms):                            164.95
P50 ITL (ms):                            171.26
P75 ITL (ms):                            182.13
P90 ITL (ms):                            200.83
==================================================
```

### Graph
TBD
