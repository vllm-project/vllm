# Parameter Sweeps

## Online Benchmark

### Basic

`vllm bench sweep serve` automatically starts `vllm serve` and runs `vllm bench serve` to evaluate vLLM over multiple configurations.

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
            "random_input_len": 128,
            "random_output_len": 32
        },
        {
            "random_input_len": 256,
            "random_output_len": 64
        },
        {
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

!!! important
    If both `--serve-params` and `--bench-params` are passed, the script will iterate over the Cartesian product between them.
    You can use `--dry-run` to preview the commands to be run.

    We only start the server once for each `--serve-params`, and keep it running for multiple `--bench-params`.
    Between each benchmark run, we call the `/reset_prefix_cache` and `/reset_mm_cache` endpoints to get a clean slate for the next run.
    In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.

!!! note
    By default, each parameter combination is run 3 times to make the results more reliable. You can adjust the number of runs by setting `--num-runs`.

!!! tip
    You can use the `--resume` option to continue the parameter sweep if one of the runs failed.
  
### SLA auto-tuner

`vllm bench sweep serve_sla` is a wrapper over `vllm bench sweep serve` that tunes either the request rate or concurrency (choose using `--sla-variable`) in order to satisfy the SLA constraints given by `--sla-params`.

For example, to ensure E2E latency within different target values for 99% of requests:

```json
[
    {
        "p99_e2el_ms": "<=200"
    },
    {
        "p99_e2el_ms": "<=500"
    },
    {
        "p99_e2el_ms": "<=1000"
    },
    {
        "p99_e2el_ms": "<=2000"
    }
]
```

Example command:

```bash
vllm bench sweep serve_sla \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --sla-params benchmarks/sla_hparams.json \
    --sla-variable max_concurrency \
    -o benchmarks/results
```

The algorithm for adjusting the SLA variable is as follows:

1. Run the benchmark once with maximum possible QPS, and once with minimum possible QPS. For each run, calculate the distance of the SLA metrics from their targets, resulting in data points of QPS vs SLA distance.
2. Perform spline interpolation between the data points to estimate the QPS that results in zero SLA distance.
3. Run the benchmark with the estimated QPS and add the resulting data point to the history.
4. Repeat Steps 2 and 3 until the maximum QPS that passes SLA and the minimum QPS that fails SLA in the history are close enough to each other.

!!! important
    SLA tuning is applied over each combination of `--serve-params`, `--bench-params`, and `--sla-params`.

    For a given combination of `--serve-params` and `--bench-params`, we share the benchmark results across `--sla-params` to avoid rerunning benchmarks with the same SLA variable value.

## Visualization

### Basic

`vllm bench sweep plot` can be used to plot performance curves from parameter sweep results.

Example command:

```bash
vllm bench sweep plot benchmarks/results/<timestamp> \
    --var-x max_concurrency \
    --row-by random_input_len \
    --col-by random_output_len \
    --curve-by api_server_count,max_num_batched_tokens \
    --filter-by 'max_concurrency<=1024'
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
