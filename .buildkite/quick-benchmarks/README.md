
# Quick benchmark

## Introduction

This directory contains a quick performance benchmarking CI for vllm. The goal is to help developers know the impact of their PRs on the performance of vllm. 

This benchmark will be *triggered* upon:
- A PR being merged into vllm.
- Every commit for those PRs with `perf-benchmarks` label.

This benchmark covers latency, throughput and fix-qps serving on A100 (the support for more GPUs is comming later), with different models. The estimated runtime is about 40 minutes.

## Configuring the workload for the quick benchmark

The workload of the quick benchmark contains two parts: latency tests in `latency-tests.json`, and throughput tests in `serving-tests.json`. 

### Latency test

Here is an example of one test inside `latency-tests.json`:

```json
[
    ...
    {
        "test_name": "latency_llama8B_tp1",
        "parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "tensor_parallel_size": 1,
            "load_format": "dummy"
        }
    },
    ...
]
```

In this example:
-  The `test_name` attributes is a unique identifier for the test. In `latency-tests.json`, it must start with `latency_`.
-  The `parameters` attribute control the command line arguments to be used for `benchmark_latency.py`. Note that please use underline `_` instead of the dash `-` when specifying the command line arguments, and `quick-benchmark.sh` will convert the underline to dash when feeding the arguments to `benchmark_latency.py`. For example, the corresponding command line arguments for `benchmark_latency.py` will be `--model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 1 --load-format dummy`

### Serving test


We test the throughput by using `benchmark_serving.py` with request rate = inf to cover the online serving overhead. The corresponding parameters are in `serving-tests.json`, and here is an example:

```
[
    ...
    {
        "test_name": "serving_llama8B_tp1_sharegpt",
        "server_parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "tensor_parallel_size": 1,
            "swap_space": 16,
            "disable_log_stats": "",
            "disable_log_requests": "",
            "load_format": "dummy"
        },
        "client_parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "dataset_name": "sharegpt",
            "dataset_path": "./ShareGPT_V3_unfiltered_cleaned_split.json",
            "num_prompts": 1000
        }
    },
    ...
]
```

Inside this example:
- The `test_name` attribute is also a unique identifier for the test. It must start with `serving_`.
- The `server-parameters` includes the command line arguments for vllm server. 
- The `client-parameters` includes the command line arguments for `benchmark_serving.py`. 


## Visualizing the results

The `results2md.py` helps you put the benchmarking results inside a markdown table. To access it, in your PR page, scroll down