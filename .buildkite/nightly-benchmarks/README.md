# vLLM benchmark suite

## Introduction

This directory contains the performance benchmarking CI for vllm.
The goal is to help developers know the impact of their PRs on the performance of vllm.

This benchmark will be *triggered* upon:
- A PR being merged into vllm.
- Every commit for those PRs with `perf-benchmarks` label.

**Benchmarking Coverage**: latency, throughput and fix-qps serving on A100 (the support for more GPUs is comming later), with different models.

**Benchmarking Duration**: about 1hr.

**For benchmarking developers**: please try your best to constraint the duration of benchmarking to less than 1.5 hr so that it won't take forever to run.


## Configuring the workload

The benchmarking workload contains three parts:
- Latency tests in `latency-tests.json`.
- Throughput tests in `throughput-tests.json`.
- Serving tests in `serving-tests.json`.

See [descriptions.md](tests/descriptions.md) for detailed descriptions. 

### Latency test

Here is an example of one test inside `latency-tests.json`:

```json
[
    {
        "test_name": "latency_llama8B_tp1",
        "parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "tensor_parallel_size": 1,
            "load_format": "dummy",
            "num_iters_warmup": 5,
            "num_iters": 15
        }
    },
]
```

In this example:
-  The `test_name` attributes is a unique identifier for the test. In `latency-tests.json`, it must start with `latency_`.
-  The `parameters` attribute control the command line arguments to be used for `benchmark_latency.py`. Note that please use underline `_` instead of the dash `-` when specifying the command line arguments, and `run-benchmarks-suite.sh` will convert the underline to dash when feeding the arguments to `benchmark_latency.py`. For example, the corresponding command line arguments for `benchmark_latency.py` will be `--model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 1 --load-format dummy --num-iters-warmup 5 --num-iters 15`

Note that the performance numbers are highly sensitive to the value of the parameters. Please make sure the parameters are set correctly.

WARNING: The benchmarking script will save json results by itself, so please do not configure `--output-json` parameter in the json file.


### Throughput test
The tests are specified in `throughput-tests.json`. The syntax is similar to `latency-tests.json`, except for that the parameters will be fed forward to `benchmark_throughput.py`.

The number of this test is also stable -- a slight change on the value of this number might vary the performance numbers by a lot.

### Serving test
We test the throughput by using `benchmark_serving.py` with request rate = inf to cover the online serving overhead. The corresponding parameters are in `serving-tests.json`, and here is an example:

```
[
    {
        "test_name": "serving_llama8B_tp1_sharegpt",
        "qps_list": [1, 4, 16, "inf"],
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
            "backend": "vllm",
            "dataset_name": "sharegpt",
            "dataset_path": "./ShareGPT_V3_unfiltered_cleaned_split.json",
            "num_prompts": 200
        }
    },
]
```

Inside this example:
- The `test_name` attribute is also a unique identifier for the test. It must start with `serving_`.
- The `server-parameters` includes the command line arguments for vLLM server.
- The `client-parameters` includes the command line arguments for `benchmark_serving.py`.
- The `qps_list` controls the list of qps for test. It will be used to configure the `--request-rate` parameter in `benchmark_serving.py`

The number of this test is less stable compared to the delay and latency benchmarks (due to randomized sharegpt dataset sampling inside `benchmark_serving.py`), but a large change on this number (e.g. 5% change) still vary the output greatly.

WARNING: The benchmarking script will save json results by itself, so please do not configure `--save-results` or other results-saving-related parameters in `serving-tests.json`.

## Visualizing the results
The `convert-results-json-to-markdown.py` helps you put the benchmarking results inside a markdown table, by formatting [descriptions.md](tests/descriptions.md) with real benchmarking results.
You can find the result presented as a table inside the `buildkite/performance-benchmark` job page.
If you do not see the table, please wait till the benchmark finish running.
The json version of the table (together with the json version of the benchmark) will be also attached to the markdown file.
The raw benchmarking results (in the format of json files) are in the `Artifacts` tab of the benchmarking.
