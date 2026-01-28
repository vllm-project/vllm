# Performance Dashboard

The performance dashboard is used to confirm whether new changes improve/degrade performance under various workloads.
It is updated by triggering benchmark runs on every commit with both the `perf-benchmarks` and `ready` labels, and when a PR is merged into vLLM.

The results are automatically published to the public [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm).

## Manually Trigger the benchmark

Use [vllm-ci-test-repo images](https://gallery.ecr.aws/q9t5s3a7/vllm-ci-test-repo) with vLLM benchmark suite.
For x86 CPU environment, please use the image with "-cpu" postfix. For AArch64 CPU environment, please use the image with "-arm64-cpu" postfix.

Here is an example for docker run command for CPU. For GPUs skip setting the `ON_CPU` env var.

```bash
export VLLM_COMMIT=7f42dc20bb2800d09faa72b26f25d54e26f1b694 # use full commit hash from the main branch
export HF_TOKEN=<valid Hugging Face token>
if [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]]; then
  IMG_SUFFIX="arm64-cpu"
else
  IMG_SUFFIX="cpu"
fi
docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}-${IMG_SUFFIX}
```

Then, run below command inside the docker instance.

```bash
bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

When run, benchmark script generates results under **benchmark/results** folder, along with the benchmark_results.md and benchmark_results.json.

### Runtime environment variables

- `ON_CPU`: set the value to '1' on Intel® Xeon® and Arm® Neoverse™ Processors. Default value is 0.
- `SERVING_JSON`: JSON file to use for the serving tests. Default value is empty string (use default file).
- `LATENCY_JSON`: JSON file to use for the latency tests. Default value is empty string (use default file).
- `THROUGHPUT_JSON`: JSON file to use for the throughout tests. Default value is empty string (use default file).
- `REMOTE_HOST`: IP for the remote vLLM service to benchmark. Default value is empty string.
- `REMOTE_PORT`: Port for the remote vLLM service to benchmark. Default value is empty string.

### Visualization

The `convert-results-json-to-markdown.py` helps you put the benchmarking results inside a markdown table with real benchmarking results.
You can find the result presented as a table inside the `buildkite/performance-benchmark` job page.
If you do not see the table, please wait till the benchmark finish running.
The json version of the table (together with the json version of the benchmark) will be also attached to the markdown file.
The raw benchmarking results (in the format of json files) are in the `Artifacts` tab of the benchmarking.

#### Performance Results Comparison

The `compare-json-results.py` helps to compare benchmark results JSON files converted using `convert-results-json-to-markdown.py`.
When run, benchmark script generates results under `benchmark/results` folder, along with the `benchmark_results.md` and `benchmark_results.json`.
`compare-json-results.py` compares two `benchmark_results.json` files and provides performance ratio e.g. for Output Tput, Median TTFT and Median TPOT.  
If only one benchmark_results.json is passed, `compare-json-results.py` compares different TP and PP configurations in the benchmark_results.json instead.

Here is an example using the script to compare result_a and result_b with max concurrency and qps for same Model, Dataset name, input/output length.
`python3 compare-json-results.py -f results_a/benchmark_results.json -f results_b/benchmark_results.json`

***Output Tput (tok/s) — Model : [ meta-llama/Llama-3.1-8B-Instruct ] , Dataset Name : [ random ] , Input Len : [ 2048.0 ] , Output Len : [ 2048.0 ]***

|    | # of max concurrency | qps  | results_a/benchmark_results.json | results_b/benchmark_results.json | perf_ratio        |
|----|------|-----|-----------|----------|----------|
| 0  | 12 | inf | 24.98   | 186.03 |  7.45 |
| 1  | 16 | inf|  25.49  | 246.92 | 9.69 |
| 2  | 24 | inf| 27.74  | 293.34 |  10.57 |
| 3  | 32 | inf| 28.61  |306.69 | 10.72 |

***compare-json-results.py – Command-Line Parameters***  

compare-json-results.py provides configurable parameters to compare one or more benchmark_results.json files and generate summary tables and plots.  
In most cases, users only need to specify --file to parse the desired benchmark results.

| Parameter              | Type               | Default Value           | Description                                                                                           |
| ---------------------- | ------------------ | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `--file`               | `str` (appendable) | *None*                  | Input JSON result file(s). Can be specified multiple times to compare multiple benchmark outputs.     |
| `--debug`              | `bool`             | `False`                 | Enables debug mode. When set, prints all available information to aid troubleshooting and validation. |
| `--plot` / `--no-plot` | `bool`             | `True`                  | Controls whether performance plots are generated. Use `--no-plot` to disable graph generation.        |
| `--xaxis`              | `str`              | `# of max concurrency.` | Column name used as the X-axis in comparison plots (for example, concurrency or batch size).          |
| `--latency`            | `str`              | `p99`                   | Latency aggregation method used for TTFT/TPOT. Supported values: `median` or `p99`.                   |
| `--ttft-max-ms`        | `float`            | `3000.0`                | Reference upper bound (milliseconds) for TTFT plots, typically used to visualize SLA thresholds.      |
| `--tpot-max-ms`        | `float`            | `100.0`                 | Reference upper bound (milliseconds) for TPOT plots, typically used to visualize SLA thresholds.      |

***Valid Max Concurrency Summary***  

Based on the configured TTFT and TPOT SLA thresholds, compare-json-results.py computes the maximum valid concurrency for each benchmark result.  
The “Max # of max concurrency. (Both)” column represents the highest concurrency level that satisfies both TTFT and TPOT constraints simultaneously.  
This value is typically used in capacity planning and sizing guides.  

| # | Configuration  | Max # of max concurrency. (TTFT ≤ 10000 ms) | Max # of max concurrency. (TPOT ≤ 100 ms) | Max # of max concurrency. (Both) | Output Tput @ Both (tok/s) | TTFT @ Both (ms) | TPOT @ Both (ms) |
| - | -------------- | ------------------------------------------- | ----------------------------------------- | -------------------------------- | -------------------------- | ---------------- | ---------------- |
| 0 | results-a      | 128.00                                      | 12.00                                     | 12.00                            | 127.76                     | 3000.82          | 93.24            |
| 1 | results-b      | 128.00                                      | 32.00                                     | 32.00                            | 371.42                     | 2261.53          | 81.74            |

More information on the performance benchmarks and their parameters can be found in [Benchmark README](https://github.com/intel-ai-tce/vllm/blob/more_cpu_models/.buildkite/nightly-benchmarks/README.md) and [performance benchmark description](../../.buildkite/performance-benchmarks/performance-benchmarks-descriptions.md).

## Continuous Benchmarking

The continuous benchmarking provides automated performance monitoring for vLLM across different models and GPU devices. This helps track vLLM's performance characteristics over time and identify any performance regressions or improvements.

### How It Works

The continuous benchmarking is triggered via a [GitHub workflow CI](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) in the PyTorch infrastructure repository, which runs automatically every 4 hours. The workflow executes three types of performance tests:

- **Serving tests**: Measure request handling and API performance
- **Throughput tests**: Evaluate token generation rates
- **Latency tests**: Assess response time characteristics

### Benchmark Configuration

The benchmarking currently runs on a predefined set of models configured in the [vllm-benchmarks directory](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks). To add new models for benchmarking:

1. Navigate to the appropriate GPU directory in the benchmarks configuration
2. Add your model specifications to the corresponding configuration files
3. The new models will be included in the next scheduled benchmark run
