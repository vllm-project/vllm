# Workload-Aware KVCache Cache Policy

Nowadays, cloud providers typically use a unified serving engine deployed on GPUs to serve all request types (text, image, file, agent-calls, etc.) for better resource utilization. However, the mean response time of these workloads is different, causing KVCache reuse time differences. For example, humans respond faster when they process image/audio data than to the complex text or file results generated by the LLM. Based on our analysis of real-world LLM traffic from top-level cloud provider Aliyun Bailian ([https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)), we found that the general cache policy (like LRU) for KVCache may not be optimal.

This PR provides a new feature, the Workload-Aware KVCache policy (WA), enhancing the 'FreeKVCacheBlockQueue' data structure to 'WorkloadAwareFreeKVCacheBlockQueue'. This leverages the extra information (i.e., workload-type) corresponding to each KVCache block's request to perform better cache eviction than the default LRU policy used by FreeKVCacheBlockQueue.

This PR introduces a new optional parameter for each request, `metainfo`, which contains the workload type of the request set by the frontend client. For example, a client can set a request's workload type as `text_1`, meaning this request is the first turn of a chat catalog, or `file_2` meaning the request is the second turn of file analysis. Using this workload tag, the cloud provider can classify requests from different business scenarios and guide the vLLM engine to do cache eviction.

Note that the WA policy can be applied beyond the traces from Aliyun Bailian. The WA policy can be useful in any deployment where one vLLM serving engine serves multiple frontend workloads (Chat, Multi-modal, Reasoning, etc.). As soon as the client provides the workload tag in the request, the WA policy can leverage this to perform better cache eviction than LRU.

## Usage

Users or developers can enable this eviction policy via the `--wa-offline-param-path` hyperparameter.
If None is specified, it will fall back to the default FreeKVCacheBlockQueue implementation (LRU).

`type_info` should be specified in the request.

Your can refer our example `benchmark/benchmark_wa.py`, it will use the Aliyun Bailian trace to profile KVCache reuse patterns to improve the cache eviction performance.

## Implementation Details

The KVCache reuse patterns vary across different request categories and can be predicted using historical request information. Therefore, the WA policy estimates a KV cache reuse probability for each workload, and the WA free queue selects the block with the lowest predicted reuse probability in the upcoming time window as the eviction candidate.

Our Workload-Aware KVCache policy (WA) enhances the `FreeKVCacheBlockQueue` (FeQ) data structure to `WorkloadAwareFreeKVCacheBlockQueue` (WAQ). The key difference between WAQ and FeQ is that WAQ maintains multiple doubly linked queues, collectively comprising num\_free\_blocks blocks. Each doubly linked queue corresponds to a specific workload category, and all blocks within a given queue belong to that same category.

Initially, only a single default queue exists, as cache blocks are not yet assigned to any category. During allocation, each KVCacheBlock is tagged with a type\_info field that indicates its workload category. Each free KVCacheBlock will be placed in its corresponding queues and wait to be evicted.

When `append`is called, WAQ inserts the KVCacheBlock into its category's queue.If the queue doesn't exist, it will be created.

When `remove` is called, the block is removed from its category's queue in O(1) time.

When `popLeft`is called, WAQ considers all queues and selects the block with the lowest reuse probability.Since blocks within each queue remain sorted by last accessed time (LRU), only the first block in each queue needs evaluation. The block with the minimal reuse probability is chosen as the victim.

More details analysis about the production trace and the formula about our probability prediction model can be found in our paper [https://www.usenix.org/conference/atc25/presentation/wang-jiahao](https://www.usenix.org/conference/atc25/presentation/wang-jiahao)  (Appeared at USENIX ATC '25).

## Evaluation

We evaluate the effectiveness of WA policy on 7B and 70B model in different GPU cache space.

### Setup

- Model: Qwen/Qwen2.5-7B-Instruct, meta/Llama-3.3-70B-Instruct
- GPU: 1~4 x Nvidia A800 80GB, TP=4 when testing the 70B model.
- Trace: [Aliyun Bailian Trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)
- Qps: First hour 6qps, second hour 6ps.
- Total elements: 43195
- Average input length: 2337.99
- Average output length: 430.34

### Demo

The `benchmark/benchmark_wa.py` script demonstrates a basic implementation of the workload-aware policy's profiling and prediction workflow. This specially designed client simulates multi-turn dialogues by generating requests based on the previous turn's output.

The `benchmark/profiler_utils.py` module provides a cache simulator to profile KVCache reuse patterns across different workloads.

The Bailian Traces dataset contains a two-hour trace at 6 queries per second (QPS). We utilize the first hour's trace to:

1. Profile KVCache reuse patterns for various workloads
2. Generate and export a hyperparameter configuration file

Subsequently, we launch a vLLM engine that loads this hyperparameter file to serve the second hour's trace.

Additionally, `benchmark_wa.py` generates detailed metrics files for analyzing both Query Time to First Token (QTTFT) and Time Per Output Token (TPOT) performance.

### Performance Improvement

Since KVCache hits primarily reduce Time to First Token (TTFT) latency, and Prefill-Decoding (P-D) disaggregation has become prevalent in modern cloud provider deployments, we tested the Prefill-Only component (representing the Prefill-Node in P-D disaggregation) using the 6 QPS trace data. These tests were conducted across varying GPU KVCache block allocations. The reported queued TTFT metric—which includes request queuing time—is particularly critical for user experience evaluation.

#### Qwen 7B model

The max\_num\_batch\_tokens is set as 16384 to improve the GPU utilization. The GPU memory utlization is 0.9. We use the hyperparamter 'num-gpu-blocks-override' to change the cache space.

| num\_gpu\_blocks | WA\_mean\_qttft | LRU\_mean\_qttft | QTTFT\_Improvement (%) | WA\_hit\_rate | LRU\_hit\_rate | Hit\_Rate\_Improvement (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 14016.4 | 22322.4 | 37.21 | 0.1381 | 0.1175 | 17.53 |
| 2048 | 13458.6 | 23545 | 42.84 | 0.1586 | 0.1281 | 23.81 |
| 3072 | 10594.5 | 21969.9 | 51.78 | 0.1753 | 0.1407 | 24.59 |
| 4096 | 8544.2 | 13710.8 | 37.68 | 0.1934 | 0.1566 | 23.5 |
| 5120 | 6003.9 | 10271.6 | 41.55 | 0.2054 | 0.1786 | 15.01 |
| 6144 | 5283.4 | 7877.8 | 32.93 | 0.2245 | 0.2068 | 8.56 |
| 7168 | 2945.9 | 4963 | 40.63 | 0.2392 | 0.2299 | 4.05 |
| 8192 | 2264.1 | 2280.6 | 0.72 | 0.256 | 0.2498 | 2.48 |

We can see that the WA policy can get the cache hit rate improvement from 2.5% to 24.6% than LRU, and reduce the qttft from 0.7% to 52% than LRU. WA policy is better when the cache space is relatively limited.

#### Llama 70B model

Since the system throughtput is 1~2 qps when inferencing the 70B model, we sample the second hour's 6qps trace to 2qps. we prove the ratio of different turns remains the same.

| num\_gpu\_blocks | WA\_mean\_qttft | LRU\_mean\_qttft | QTTFT\_Improvement (%) | WA\_hit\_rate | LRU\_hit\_rate | Hit\_Rate\_Improvement (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 6948.15 | 9064.9 | 23.351 | 0.131199 | 0.109314 | 20.0207 |
| 1024 | 4231.16 | 7808.79 | 45.8154 | 0.166392 | 0.12963 | 28.3594 |
| 2048 | 3299.04 | 4589.6 | 28.1191 | 0.215587 | 0.201457 | 7.01393 |
| 3072 | 2672.74 | 2798.33 | 4.48785 | 0.261666 | 0.259961 | 0.655852 |

We can see that the WA policy can get the cache hit rate improvement from 0.7% to 28% than LRU,and reduce the qttft from 4.5% to 46% than LRU.

## Comments

The Workload-Aware policy (WA) feature is designed for users who observe significant differences between workloads served within a single instance.

Users or developers can enable this eviction policy via the `--wa-offline-param-path` hyperparameter.
If None is specified, it will fall back to the default FreeKVCacheBlockQueue implementation (LRU).
Our example `benchmark_wa.py`offers a demo to generate the hyperparameter file.

When incoming requests lack sufficient workload tagging, this feature should remain disabled.
In such cases,the system will automatically revert to the default FreeKVCacheBlockQueue implementation.
