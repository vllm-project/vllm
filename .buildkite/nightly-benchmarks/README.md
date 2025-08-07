# vLLM benchmark suite

## Introduction

This directory contains two sets of benchmark for vllm.

- Performance benchmark: benchmark vllm's performance under various workload, for **developers** to gain clarity on whether their PR improves/degrades vllm's performance
- Nightly benchmark: compare vllm's performance against alternatives (tgi, trt-llm and lmdeploy), for **the public** to know when to choose vllm.

See [vLLM performance dashboard](https://perf.vllm.ai) for the latest performance benchmark results and [vLLM GitHub README](https://github.com/vllm-project/vllm/blob/main/README.md) for latest nightly benchmark results.

## Performance benchmark quick overview

**Benchmarking Coverage**: latency, throughput and fix-qps serving on A100 (the support for FP8 benchmark on H100 is coming!) and Intel® Xeon® Processors, with different models.

**Benchmarking Duration**: about 1hr.

**For benchmarking developers**: please try your best to constraint the duration of benchmarking to about 1 hr so that it won't take forever to run.

## Nightly benchmark quick overview

**Benchmarking Coverage**: Fix-qps serving on A100 (the support for FP8 benchmark on H100 is coming!) on Llama-3 8B, 70B and Mixtral 8x7B.

**Benchmarking engines**: vllm, TGI, trt-llm and lmdeploy.

**Benchmarking Duration**: about 3.5hrs.

## Trigger the benchmark

Performance benchmark will be triggered when:

- A PR being merged into vllm.
- Every commit for those PRs with `perf-benchmarks` label AND `ready` label.

Manually Trigger the benchmark

```bash
bash .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh
```

Runtime environment variables:

- `ON_CPU`: set the value to '1' on Intel® Xeon® Processors. Default value is 0.
- `SERVING_JSON`: JSON file to use for the serving tests. Default value is empty string (use default file).
- `LATENCY_JSON`: JSON file to use for the latency tests. Default value is empty string (use default file).
- `THROUGHPUT_JSON`: JSON file to use for the throughout tests. Default value is empty string (use default file).
- `REMOTE_HOST`: IP for the remote vLLM service to benchmark. Default value is empty string.
- `REMOTE_PORT`: Port for the remote vLLM service to benchmark. Default value is empty string.

Nightly benchmark will be triggered when:

- Every commit for those PRs with `perf-benchmarks` label and `nightly-benchmarks` label.

## Performance benchmark details

See [performance-benchmarks-descriptions.md](performance-benchmarks-descriptions.md) for detailed descriptions, and use `tests/latency-tests.json`, `tests/throughput-tests.json`, `tests/serving-tests.json` to configure the test cases.
> NOTE: For Intel® Xeon® Processors, use `tests/latency-tests-cpu.json`, `tests/throughput-tests-cpu.json`, `tests/serving-tests-cpu.json` instead.
>
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

- The `test_name` attributes is a unique identifier for the test. In `latency-tests.json`, it must start with `latency_`.
- The `parameters` attribute control the command line arguments to be used for `vllm bench latency`. Note that please use underline `_` instead of the dash `-` when specifying the command line arguments, and `run-performance-benchmarks.sh` will convert the underline to dash when feeding the arguments to `vllm bench latency`. For example, the corresponding command line arguments for `vllm bench latency` will be `--model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 1 --load-format dummy --num-iters-warmup 5 --num-iters 15`

Note that the performance numbers are highly sensitive to the value of the parameters. Please make sure the parameters are set correctly.

WARNING: The benchmarking script will save json results by itself, so please do not configure `--output-json` parameter in the json file.

### Throughput test

The tests are specified in `throughput-tests.json`. The syntax is similar to `latency-tests.json`, except for that the parameters will be fed forward to `vllm bench throughput`.

The number of this test is also stable -- a slight change on the value of this number might vary the performance numbers by a lot.

### Serving test

We test the throughput by using `vllm bench serve` with request rate = inf to cover the online serving overhead. The corresponding parameters are in `serving-tests.json`, and here is an example:

```json
[
    {
        "test_name": "serving_llama8B_tp1_sharegpt",
        "qps_list": [1, 4, 16, "inf"],
        "server_parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "tensor_parallel_size": 1,
            "swap_space": 16,
            "disable_log_stats": "",
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
- The `client-parameters` includes the command line arguments for `vllm bench serve`.
- The `qps_list` controls the list of qps for test. It will be used to configure the `--request-rate` parameter in `vllm bench serve`

The number of this test is less stable compared to the delay and latency benchmarks (due to randomized sharegpt dataset sampling inside `benchmark_serving.py`), but a large change on this number (e.g. 5% change) still vary the output greatly.

WARNING: The benchmarking script will save json results by itself, so please do not configure `--save-results` or other results-saving-related parameters in `serving-tests.json`.

### Visualizing the results

The `convert-results-json-to-markdown.py` helps you put the benchmarking results inside a markdown table, by formatting [descriptions.md](performance-benchmarks-descriptions.md) with real benchmarking results.
You can find the result presented as a table inside the `buildkite/performance-benchmark` job page.
If you do not see the table, please wait till the benchmark finish running.
The json version of the table (together with the json version of the benchmark) will be also attached to the markdown file.
The raw benchmarking results (in the format of json files) are in the `Artifacts` tab of the benchmarking.

The `compare-json-results.py` helps to compare benchmark results JSON files converted using `convert-results-json-to-markdown.py`.
When run, benchmark script generates results under `benchmark/results` folder, along with the `benchmark_results.md` and `benchmark_results.json`.
`compare-json-results.py` compares two `benchmark_results.json` files and provides performance ratio e.g. for Output Tput, Median TTFT and Median TPOT.

Here is an example using the script to compare result_a and result_b without detail test name.
`python3 compare-json-results.py -f results_a/benchmark_results.json -f results_b/benchmark_results.json --ignore_test_name`

|    | results_a/benchmark_results.json | results_b/benchmark_results.json | perf_ratio        |
|----|----------------------------------------|----------------------------------------|----------|
| 0  | 142.633982                             | 156.526018                             | 1.097396 |
| 1  | 241.620334                             | 294.018783                             | 1.216863 |
| 2  | 218.298905                             | 262.664916                             | 1.203235 |
| 3  | 242.743860                             | 299.816190                             | 1.235113 |

Here is an example using the script to compare result_a and result_b with detail test name.
`python3 compare-json-results.py -f results_a/benchmark_results.json -f results_b/benchmark_results.json`

|   | results_a/benchmark_results.json_name | results_a/benchmark_results.json | results_b/benchmark_results.json_name | results_b/benchmark_results.json | perf_ratio        |
|---|---------------------------------------------|----------------------------------------|---------------------------------------------|----------------------------------------|----------|
| 0 | serving_llama8B_tp1_sharegpt_qps_1          | 142.633982                             | serving_llama8B_tp1_sharegpt_qps_1          | 156.526018                             | 1.097396 |
| 1 | serving_llama8B_tp1_sharegpt_qps_16         | 241.620334                             | serving_llama8B_tp1_sharegpt_qps_16         | 294.018783                             | 1.216863 |
| 2 | serving_llama8B_tp1_sharegpt_qps_4          | 218.298905                             | serving_llama8B_tp1_sharegpt_qps_4          | 262.664916                             | 1.203235 |
| 3 | serving_llama8B_tp1_sharegpt_qps_inf        | 242.743860                             | serving_llama8B_tp1_sharegpt_qps_inf        | 299.816190                             | 1.235113 |
| 4 | serving_llama8B_tp2_random_1024_128_qps_1   | 96.613390                              | serving_llama8B_tp4_random_1024_128_qps_1   | 108.404853                             | 1.122048 |

## Nightly test details

See [nightly-descriptions.md](nightly-descriptions.md) for the detailed description on test workload, models and docker containers of benchmarking other llm engines.

### Workflow

- The [nightly-pipeline.yaml](nightly-pipeline.yaml) specifies the docker containers for different LLM serving engines.
- Inside each container, we run [scripts/run-nightly-benchmarks.sh](scripts/run-nightly-benchmarks.sh), which will probe the serving engine of the current container.
- The `scripts/run-nightly-benchmarks.sh` will parse the workload described in [nightly-tests.json](tests/nightly-tests.json) and launch the right benchmark for the specified serving engine via `scripts/launch-server.sh`.
- At last, we run [scripts/summary-nightly-results.py](scripts/summary-nightly-results.py) to collect and plot the final benchmarking results, and update the results to buildkite.

### Nightly tests

In [nightly-tests.json](tests/nightly-tests.json), we include the command line arguments for benchmarking commands, together with the benchmarking test cases. The format is highly similar to performance benchmark.

### Docker containers

The docker containers for benchmarking are specified in `nightly-pipeline.yaml`.

WARNING: the docker versions are HARD-CODED and SHOULD BE ALIGNED WITH `nightly-descriptions.md`. The docker versions need to be hard-coded as there are several version-specific bug fixes inside `scripts/run-nightly-benchmarks.sh` and `scripts/launch-server.sh`.

WARNING: populating `trt-llm` to latest version is not easy, as it requires updating several protobuf files in [tensorrt-demo](https://github.com/neuralmagic/tensorrt-demo.git).
