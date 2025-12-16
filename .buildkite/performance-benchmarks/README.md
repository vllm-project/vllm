# vLLM benchmark suite

## Introduction

This directory contains a benchmarking suite for **developers** to run locally and gain clarity on whether their PR improves/degrades vllm's performance.
vLLM also maintains a continuous performance benchmark under [perf.vllm.ai](https://perf.vllm.ai/), hosted under PyTorch CI HUD.

## Performance benchmark quick overview

**Benchmarking Coverage**: latency, throughput and fix-qps serving on B200, A100, H100, Intel® Xeon® Processors and Intel® Gaudi® 3 Accelerators with different models.

**Benchmarking Duration**: about 1hr.

**For benchmarking developers**: please try your best to constraint the duration of benchmarking to about 1 hr so that it won't take forever to run.

## Trigger the benchmark

The benchmark needs to be triggered manually:

```bash
bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

Runtime environment variables:

- `ON_CPU`: set the value to '1' on Intel® Xeon® Processors. Default value is 0.
- `SERVING_JSON`: JSON file to use for the serving tests. Default value is empty string (use default file).
- `LATENCY_JSON`: JSON file to use for the latency tests. Default value is empty string (use default file).
- `THROUGHPUT_JSON`: JSON file to use for the throughout tests. Default value is empty string (use default file).
- `REMOTE_HOST`: IP for the remote vLLM service to benchmark. Default value is empty string.
- `REMOTE_PORT`: Port for the remote vLLM service to benchmark. Default value is empty string.

## Performance benchmark details

See [performance-benchmarks-descriptions.md](performance-benchmarks-descriptions.md) for detailed descriptions, and use `tests/latency-tests.json`, `tests/throughput-tests.json`, `tests/serving-tests.json` to configure the test cases.
> NOTE: For Intel® Xeon® Processors, use `tests/latency-tests-cpu.json`, `tests/throughput-tests-cpu.json`, `tests/serving-tests-cpu.json` instead.
For Intel® Gaudi® 3 Accelerators, use `tests/latency-tests-hpu.json`, `tests/throughput-tests-hpu.json`, `tests/serving-tests-hpu.json` instead.
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

#### Default Parameters Field

We can specify default parameters in a JSON field with key `defaults`. Parameters defined in the field are applied globally to all serving tests, and can be overridden in test case fields. Here is an example:

<details>
<summary> An Example of default parameters field </summary>

```json
{
  "defaults": {
    "qps_list": [
      "inf"
    ],
    "server_environment_variables": {
      "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1
    },
    "server_parameters": {
      "tensor_parallel_size": 1,
      "dtype": "bfloat16",
      "block_size": 128,
      "disable_log_stats": "",
      "load_format": "dummy"
    },
    "client_parameters": {
      "backend": "vllm",
      "dataset_name": "random",
      "random-input-len": 128,
      "random-output-len": 128,
      "num_prompts": 200,
      "ignore-eos": ""
    }
  },
  "tests": [
    {
      "test_name": "serving_llama3B_tp2_random_128_128",
      "server_parameters": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "tensor_parallel_size": 2,
      },
      "client_parameters": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
      }
    },
    {
      "test_name": "serving_qwen3_tp4_random_128_128",
      "server_parameters": {
        "model": "Qwen/Qwen3-14B",
        "tensor_parallel_size": 4,
      },
      "client_parameters": {
        "model": "Qwen/Qwen3-14B",
      }
    },
  ]
}
```

</details>

### Visualizing the results

The `convert-results-json-to-markdown.py` helps you put the benchmarking results inside a markdown table, by formatting [descriptions.md](performance-benchmarks-descriptions.md) with real benchmarking results.
You can find the result presented as a table inside the `buildkite/performance-benchmark` job page.
If you do not see the table, please wait till the benchmark finish running.
The json version of the table (together with the json version of the benchmark) will be also attached to the markdown file.
The raw benchmarking results (in the format of json files) are in the `Artifacts` tab of the benchmarking.

The `compare-json-results.py` helps to compare benchmark results JSON files converted using `convert-results-json-to-markdown.py`.
When run, benchmark script generates results under `benchmark/results` folder, along with the `benchmark_results.md` and `benchmark_results.json`.
`compare-json-results.py` compares two `benchmark_results.json` files and provides performance ratio e.g. for Output Tput, Median TTFT and Median TPOT.  
If only one benchmark_results.json is passed, `compare-json-results.py` compares different TP and PP configurations in the benchmark_results.json instead.

Here is an example using the script to compare result_a and result_b with Model, Dataset name, input/output length, max concurrency and qps.
`python3 compare-json-results.py -f results_a/benchmark_results.json -f results_b/benchmark_results.json`

|   | Model | Dataset Name | Input Len | Output Len | # of max concurrency | qps  | results_a/benchmark_results.json | results_b/benchmark_results.json | perf_ratio        |
|----|---------------------------------------|--------|-----|-----|------|-----|-----------|----------|----------|
| 0  | meta-llama/Meta-Llama-3.1-8B-Instruct | random | 128 | 128 | 1000 | 1 | 142.633982                             | 156.526018                             | 1.097396 |
| 1  | meta-llama/Meta-Llama-3.1-8B-Instruct | random | 128 | 128 | 1000 | inf| 241.620334                             | 294.018783                             | 1.216863 |

A comparison diagram will be generated below the table.
Here is an example to compare between 96c/results_gnr_96c_091_tp2pp3 and 128c/results_gnr_128c_091_tp2pp3
<img width="1886" height="828" alt="image" src="https://github.com/user-attachments/assets/c02a43ef-25d0-4fd6-90e5-2169a28682dd" />
