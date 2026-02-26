# Parameter Sweeps

`vllm bench sweep` is a suite of commands designed to run benchmarks across multiple configurations and compare them by visualizing the results.

## Online Benchmark

### Basic

`vllm bench sweep serve` starts `vllm serve` and iteratively runs `vllm bench serve` for each server configuration.

!!! tip
    If you only need to run benchmarks for a single server configuration, consider using [GuideLLM](https://github.com/vllm-project/guidellm), an established performance benchmarking framework with live progress updates and automatic report generation. It is also more flexible than `vllm bench serve` in terms of dataset loading, request formatting, and workload patterns.

Follow these steps to run the script:

1. Construct the base command to `vllm serve`, and pass it to the `--serve-cmd` option.
2. Construct the base command to `vllm bench serve`, and pass it to the `--bench-cmd` option.
3. (Optional) If you would like to vary the settings of `vllm serve`, create a new JSON file and populate it with the parameter combinations you want to test. Pass the file path to `--serve-params`.

    - Example: Tuning `--max-num-seqs` and `--max-num-batched-tokens`:

    ```json
    [
        {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 4096
        },
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096
        }
    ]
    ```

4. (Optional) If you would like to vary the settings of `vllm bench serve`, create a new JSON file and populate it with the parameter combinations you want to test. Pass the file path to `--bench-params`.

    - Example: Using different input/output lengths for random dataset:

    ```json
    [
        {
            "_benchmark_name": "scenario_A",
            "random_input_len": 128,
            "random_output_len": 32
        },
        {
            "_benchmark_name": "scenario_B",
            "random_input_len": 256,
            "random_output_len": 64
        },
        {
            "_benchmark_name": "scenario_C",
            "random_input_len": 512,
            "random_output_len": 128
        }
    ]
    ```

5. Determine where you want to save the results, and pass that to `--output-dir`.

Example command:

```bash
vllm bench sweep serve \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    -o benchmarks/results
```

By default, each parameter combination is benchmarked 3 times to make the results more reliable. You can adjust the number of runs by setting `--num-runs`.

!!! important
    If both `--serve-params` and `--bench-params` are passed, the script will iterate over the Cartesian product between them.
    You can use `--dry-run` to preview the commands to be run.

    We only start the server once for each `--serve-params`, and keep it running for multiple `--bench-params`.
    Between each benchmark run, we call all `/reset_*_cache` endpoints to get a clean slate for the next run.
    In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.

!!! note
    You should set `_benchmark_name` to provide a human-readable name for parameter combinations involving many variables.
    This becomes mandatory if the file name would otherwise exceed the maximum path length allowed by the filesystem.

!!! tip
    You can use the `--resume` option to continue the parameter sweep if an unexpected error occurs, e.g., timeout when connecting to HF Hub.

### SLA Scanner

`vllm bench sweep serve_sla` is a variant of `vllm bench sweep serve` that scans through values of request rate or concurrency (choose using `--sla-variable`) in order to find the tradeoff between latency and throughput. The results can then be [visualized](#visualization) to determine the feasible SLAs.

Example command:

```bash
vllm bench sweep serve_sla \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json
    -o benchmarks/results
```

The algorithm for scanning through different values of `sla_variable` can be summarized as follows:

1. Run the benchmark once with `sla_variable = 1` to simulate serial inference. This results in the lowest possible latency and throughput.
2. Run the benchmark once with `sla_variable = num_prompts` to simulate batch inference over the whole dataset. This results in the highest possible latency and throughput.
3. Estimate the maximum value of `sla_variable` that can be supported by the server without oversaturating it.
4. Run the benchmark over intermediate values of `sla_variable` uniformly using the remaining iterations.

You can override the number of iterations in the algorithm by setting `--sla-iters`.

!!! tip
    This is our equivalent of [GuideLLM's `--profile sweep`](https://github.com/vllm-project/guidellm/blob/v0.5.3/src/guidellm/benchmark/profiles.py#L575).

## Startup Benchmark

`vllm bench sweep startup` runs `vllm bench startup` across parameter combinations to compare cold/warm startup time for different engine settings.

Follow these steps to run the script:

1. (Optional) Construct the base command to `vllm bench startup`, and pass it to `--startup-cmd` (default: `vllm bench startup`).
2. (Optional) Reuse a `--serve-params` JSON from `vllm bench sweep serve` to vary engine settings. Only parameters supported by `vllm bench startup` are applied.
3. (Optional) Create a `--startup-params` JSON to vary startup-specific options like iteration counts.
4. Determine where you want to save the results, and pass that to `--output-dir`.

Example `--serve-params`:

```json
[
    {
        "_benchmark_name": "tp1",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9
    },
    {
        "_benchmark_name": "tp2",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9
    }
]
```

Example `--startup-params`:

```json
[
    {
        "_benchmark_name": "qwen3-0.6",
        "num_iters_cold": 2,
        "num_iters_warmup": 1,
        "num_iters_warm": 2
    }
]
```

Example command:

```bash
vllm bench sweep startup \
    --startup-cmd 'vllm bench startup --model Qwen/Qwen3-0.6B' \
    --serve-params benchmarks/serve_hparams.json \
    --startup-params benchmarks/startup_hparams.json \
    -o benchmarks/results
```

!!! important
    By default, unsupported parameters in `--serve-params` or `--startup-params` are ignored with a warning.
    Use `--strict-params` to fail fast on unknown keys.

## Visualization

### Basic

`vllm bench sweep plot` can be used to plot performance curves from parameter sweep results.

Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values. The plot is organized according to `--fig-by`, `--row-by`, `--col-by`, and `--curve-by`.

Example commands for visualizing [SLA Scanner](#sla-scanner) results:

```bash
# Latency increases as the request rate increases
vllm bench sweep plot benchmarks/results/<timestamp> \
    --var-x request_rate \
    --var-y p99_ttft_ms \
    --row-by random_input_len \
    --col-by random_output_len \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --filter-by 'request_rate<=128'

# Tradeoff between latency and throughput
vllm bench sweep plot benchmarks/results/<timestamp> \
    --var-x request_throughput \
    --var-y median_ttft_ms \
    --row-by random_input_len \
    --col-by random_output_len \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --filter-by 'request_rate<=128'
```

!!! tip
    You can use `--dry-run` to preview the figures to be plotted.

### Pareto chart

`vllm bench sweep plot_pareto` helps pick configurations that balance per-user and per-GPU throughput.

Higher concurrency or batch size can raise GPU efficiency (per-GPU), but can add per user latency; lower concurrency improves per-user rate but underutilizes GPUs; The Pareto frontier shows the best achievable pairs across your runs.

- x-axis: tokens/s/user = `output_throughput` ÷ concurrency (`--user-count-var`, default `max_concurrency`, fallback `max_concurrent_requests`).
- y-axis: tokens/s/GPU = `output_throughput` ÷ GPU count (`--gpu-count-var` if set; else gpu_count is TP×PP*DP).
- Output: a single figure at `OUTPUT_DIR/pareto/PARETO.png`.
- Show the configuration used in each data point `--label-by` (default: `max_concurrency,gpu_count`).

Example:

```bash
vllm bench sweep plot_pareto benchmarks/results/<timestamp> \
  --label-by max_concurrency,tensor_parallel_size,pipeline_parallel_size
```

!!! tip
    You can use `--dry-run` to preview the figures to be plotted.
