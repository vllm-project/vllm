# Automated vLLM Server Parameter Tuning

This script automates the process of finding the optimal server parameter combination (`max-num-seqs` and `max-num-batched-tokens`) to maximize throughput for a vLLM server. It also supports additional constraints such as E2E latency and prefix cache hit rate.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Example Use Cases](#example-use-cases)
- [Output](#output)
- [How It Works](#how-it-works)

## Prerequisites

Before running the script, please ensure the following steps are completed:

1. **Clone vLLM & Set Up Branch**: Clone the vLLM repository and check out to your desired branch.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# git checkout <your-branch>
```

1. **Install Environment**: Install or update the correct running environment. For TPU usage, activate your `conda` environment and install the corresponding `torch` and `torch_xla` versions.

2. **Model Configuration**: If you are using a customized model, ensure its configuration files are correctly placed and accessible.

## Configuration

You must set the following variables at the top of the script before execution.

   Note: You can also override the default values below via environment variables when running the script.

```bash
MODEL=meta-llama/Llama-3.3-70B-Instruct SYSTEM=TPU TP=8 DOWNLOAD_DIR='' INPUT_LEN=128 OUTPUT_LEN=2048 MAX_MODEL_LEN=2300 MIN_CACHE_HIT_PCT=0 MAX_LATENCY_ALLOWED_MS=100000000000 NUM_SEQS_LIST="128 256" NUM_BATCHED_TOKENS_LIST="1024 2048 4096" VLLM_LOGGING_LEVEL=DEBUG bash auto_tune.sh
```

| Variable | Description | Example Value |
| --- | --- | --- |
| `BASE` | **Required.** The absolute path to the parent directory of your vLLM repository directory. | `"$HOME"` |
| `MODEL` | **Required.** The Hugging Face model identifier to be served by vllm. | `"meta-llama/Llama-3.1-8B-Instruct"` |
| `SYSTEM`| **Required.** The hardware you are running on. Choices: `TPU` or `GPU`. (For other systems, it might not support saving profiles) | `"TPU"` |
| `TP` | **Required.** The tensor-parallelism size. | `1` |
| `DOWNLOAD_DIR` | **Required.** Directory to download and load model weights from. | `""` (default download path) |
| `INPUT_LEN` | **Required.** Request input length. | `4000` |
| `OUTPUT_LEN` | **Required.** Request output length. | `16` |
| `MAX_MODEL_LEN` | **Required.** Max model length. | `4096` |
| `MIN_CACHE_HIT_PCT` | Prefix cache hit rate in percentage (0-100). Set to `0` to disable. | `60` |
| `MAX_LATENCY_ALLOWED_MS` | The maximum allowed P99 end-to-end latency in milliseconds. Set to a very large number (e.g., `100000000000`) to effectively ignore the latency constraint. | `500` |
| `NUM_SEQS_LIST` | A space-separated string of `max-num-seqs` values to test. | `"128 256"` |
| `NUM_BATCHED_TOKENS_LIST` | A space-separated string of `max-num-batched-tokens` values to test. | `"1024 2048 4096"` |

**Note**: The default `NUM_SEQS_LIST` and `NUM_BATCHED_TOKENS_LIST` are set for medium-sized inputs/outputs. For very short contexts (e.g., 20 input, 20 output tokens), you may need to test larger values for `max-num-seqs`.

## How to Run

1. **Configure**: Edit the script and set the variables in the [Configuration](#configuration) section.
2. **Execute**: Run the script. Since the process can take a long time, it is highly recommended to use a terminal multiplexer like `tmux` or `screen` to prevent the script from stopping if your connection is lost.

```bash
cd <FOLDER_OF_THIS_SCRIPT>
bash auto_tune.sh
```

    Please note that the `bash auto_tune.sh` command cannot contain full or partial path with keyword `vllm`, otherwise `pkill -f vllm` command will also kill this script itself.

## Example Use Cases

Here are a few examples of how to configure the script for different goals:

### 1. Maximize Throughput (No Latency Constraint)

- **Goal**: Find the best `max-num-seqs` and `max-num-batched-tokens` to get the highest possible throughput for 1800 input tokens and 20 output tokens.
- **Configuration**:

```bash
INPUT_LEN=1800
OUTPUT_LEN=20
MAX_MODEL_LEN=2048
MIN_CACHE_HIT_PCT=0
MAX_LATENCY_ALLOWED_MS=100000000000 # A very large number
```

#### 2. Maximize Throughput with a Latency Requirement

- **Goal**: Find the best server parameters when P99 end-to-end latency must be below 500ms.
- **Configuration**:

```bash
INPUT_LEN=1800
OUTPUT_LEN=20
MAX_MODEL_LEN=2048
MIN_CACHE_HIT_PCT=0
MAX_LATENCY_ALLOWED_MS=500
```

#### 3. Maximize Throughput with Prefix Caching and Latency Requirements

- **Goal**: Find the best server parameters assuming a 60% prefix cache hit rate and a latency requirement of 500ms.
- **Configuration**:

```bash
INPUT_LEN=1800
OUTPUT_LEN=20
MAX_MODEL_LEN=2048
MIN_CACHE_HIT_PCT=60
MAX_LATENCY_ALLOWED_MS=500
```

## Output

After the script finishes, you will find the results in a new, timestamped directory created inside `$BASE/auto-benchmark/`.

- **Log Files**: The directory (`$BASE/auto-benchmark/YYYY_MM_DD_HH_MM/`) contains detailed logs for each run:
    - `vllm_log_...txt`: The log output from the vLLM server for each parameter combination.
    - `bm_log_...txt`: The log output from the `vllm bench serve` command for each benchmark run.

- **Final Result Summary**: A file named `result.txt` is created in the log directory. It contains a summary of each tested combination and concludes with the overall best parameters found.

```text
# Example result.txt content
hash:a1b2c3d4...
max_num_seqs: 128, max_num_batched_tokens: 2048, request_rate: 10.0, e2el: 450.5, throughput: 9.8, goodput: 9.8
max_num_seqs: 128, max_num_batched_tokens: 4096 does not meet latency requirement 500
...
best_max_num_seqs: 256, best_num_batched_tokens: 2048, best_throughput: 12.5, profile saved in: /home/user/vllm/auto-benchmark/2024_08_01_10_30/profile
```

  If it cannot find the best parameters, the final row will be `best_max_num_seqs: 0, best_num_batched_tokens: 0, best_throughput: 0`. This can be due to either the server not starting properly, or the latency requirement being too strict.

- **Profiler Trace**: A directory named `profile` is created inside the log directory. It contains the profiler trace file (e.g., `.xplane.pb` for TPU or a `.json` trace for GPU) from the single best-performing run.

## How It Works

The script follows a systematic process to find the optimal parameters:

1. **Find Max GPU Memory Utilization**: The script first determines the highest safe `gpu-memory-utilization` (starting from 0.98 and decreasing) that does not cause an Out-Of-Memory (OOM) error when launching the server. This ensures the benchmark runs use the maximum available memory without crashing.

2. **Iterate and Benchmark**: It then enters a nested loop, iterating through every combination of `max-num-seqs` and `max-num-batched-tokens` provided in the configuration lists.

3. **Latency-Aware Throughput Search**: For each parameter combination:
    - The vLLM server is started.
    - A benchmark is first run with an infinite request rate (`--request-rate inf`).
    - If the resulting P99 E2E latency is within the `MAX_LATENCY_ALLOWED_MS` limit, this throughput is considered the maximum for this configuration.
    - If the latency is too high, the script performs a search by iteratively decreasing the request rate until the latency constraint is met. This finds the highest sustainable throughput for the given parameters and latency requirement.

4. **Track Best Result**: Throughout the process, the script tracks the parameter combination that has yielded the highest valid throughput so far.

5. **Profile Collection**: For the best-performing run, the script saves the vLLM profiler output, which can be used for deep-dive performance analysis with tools like TensorBoard.

## Batched `auto_tune`

The `batch_auto_tune.sh` script allows you to run multiple `auto_tune.sh` experiments sequentially from a single configuration file. It iterates through a list of parameter sets, executes `auto_tune.sh` for each, and records the results back into the input file.

### Prerequisites

- **jq**: This script requires `jq` to parse the JSON configuration file.
- **gcloud**: If you plan to upload results to Google Cloud Storage, the `gcloud` CLI must be installed and authenticated.

### How to Run

1. **Create a JSON configuration file**: Create a file (e.g., `runs_config.json`) containing an array of JSON objects. Each object defines the parameters for a single `auto_tune.sh` run.

2. **Execute the script**:

    ```bash
    bash batch_auto_tune.sh <path_to_json_file> [gcs_upload_path]
    ```

    - `<path_to_json_file>`: **Required.** Path to your JSON configuration file.
    - `[gcs_upload_path]`: **Optional.** A GCS path (e.g., `gs://my-bucket/benchmark-results`) where the detailed results and profiles for each run will be uploaded. If this is empty, the results will be available on the local filesystem (see the log for `RESULT_FILE=/path/to/results/file.txt`).

### Configuration File

The JSON configuration file should contain an array of objects. Each object's keys correspond to the configuration variables for `auto_tune.sh` (see the [Configuration table above](#configuration)). These keys will be converted to uppercase environment variables for each run.

Here is an example `runs_config.json` with two benchmark configurations:

```json
[
  {
    "base": "/home/user",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "system": "TPU", # OR GPU
    "tp": 8,
    "input_len": 128,
    "output_len": 2048,
    "max_model_len": 2300,
    "num_seqs_list": "128 256",
    "num_batched_tokens_list": "8192 16384"
  },
  {
    "base": "/home/user",
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "system": "TPU", # OR GPU
    "tp": 8,
    "input_len": 4000,
    "output_len": 16,
    "max_model_len": 4096,
    "num_seqs_list": "64 128",
    "num_batched_tokens_list": "4096 8192",
    "max_latency_allowed_ms": 500
  }
]
```

### Output

The script modifies the input JSON file in place, adding the results of each run to the corresponding object. The following fields are added:

- `run_id`: A unique identifier for the run, derived from the timestamp.
- `status`: The outcome of the run (`SUCCESS`, `FAILURE`, or `WARNING_NO_RESULT_FILE`).
- `results`: The content of the `result.txt` file from the `auto_tune.sh` run.
- `gcs_results`: The GCS URL where the run's artifacts are stored (if a GCS path was provided).

A summary of successful and failed runs is also printed to the console upon completion.
