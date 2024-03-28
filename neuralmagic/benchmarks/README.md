The benchmarking UI, for all the `nightly` benchmarks is at <a href=https://neuralmagic.github.io/nm-vllm/dev/bench>Benchmarking UI</a>

# Directory Structure:

- `scripts/*.py` : Benchmark scripts that perform the metric computation.

- `scripts/logging/benchmark_result.py` : All scripts related to log benchmark results.

- `configs/*.json` : Config JSON files. These JSONs define what benchmark script to run and what combination of script parameters to use. 

- `run_*.py` : Benchmark drivers. Given a config JSON, executes all the commands defined by the config JSON.

# Benchmarking Scripts:
There are two types of benchmarks we Run.
1. `scripts/benchmark_serving.py` : The serving use case where a vLLM server and a client are spawned as separate processes. The client posts N (num_prompts) requests to the server at X requests per second (qps). This lets us observe the mean, median, 90th percentile, 99th percentile statistics on some important metrics such as,
   - TTFT (Time To First Token)
   - TPOT (Time Per Output Token) and
   - Request Latency
   - Request Throughput
   - Token Throughput
2. `scripts/benchmark_throughput.py` : The "Engine" use case where a vLLM engine is submitted N requests. This lets us observe metrics such as,
   - Request throughput, and
   - Token throughput

Both of these benchmarking scripts are a clone of the <a href=https://github.com/vllm-project/vllm/tree/main/benchmarks>upstream benchmarking scripts</a> with the same name, but with some changes. However, given the same set of requests, both the versions should produce the same numbers.

# How to Run benchmarks

`python3 -m neuralmagic.benchmarks.run_benchmarks -i neuralmagic/benchmarks/configs/benchmark_throughput.json -o ./out`

### Command Breakdown
 - `neuralmagic/benchmarks/configs/benchmark_throughput.json` is a JSON config file describing what benchmarking script to run and what script-parameter combinations to run it with. Examples of the config files can be found <a href=https://github.com/neuralmagic/nm-vllm/tree/main/neuralmagic/benchmarks/configs>here</a>.

 - `neuralmagic.benchmarks.run_benchmarks` is the benchmark driver. The benchmarking scripts `scripts/benchmark_serving.py` and `scripts/benchmark_throughput.py` have their own individual drivers at `run_benchmark_server.py` and `run_benchmark_throughput.py`. The `run_benchmarks.py` simply invokes the correct driver based on the information in the config file.

 - `./out` is the output directory where the benchmarking results are stored as JSON. The benchmark results JSON is defined in `scripts/logging/benchmark_result.py`
# JSON Config Examples
Examples of the config files can be found <a href=https://github.com/neuralmagic/nm-vllm/tree/main/neuralmagic/benchmarks/configs>here</a>.

## Benchmark Serving Config
from `configs/benchmark_serving.py`
```
{
	"configs": [
		{
			"description": "VLLM Serving - Dense",
			"models": [
                          "teknium/OpenHermes-2.5-Mistral-7B",
                          "NousResearch/Llama-2-7b-chat-hf",
                          "neuralmagic/OpenHermes-2.5-Mistral-7B-marlin",
                          "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ"
			],
			"use_all_available_gpus" : "",
			"max_model_lens": [
				4096
			],
			"sparsity": [],
			"script_name": "benchmark_serving",
			"script_args": {
				"nr-qps-pair_": [
                                        "150,0.5",
                                        "300,1",
                                        "750,2.5",
                                        "1500,5",
                                        "3000,10"
				],
				"dataset": [
					"sharegpt"
				]
			}
		}
	]
}
```
### JSON Breakdown
- `description` : A brief description about what is being benchmarked in this config. Used for logging.
- `models` : A list of huggingface models to benchmark.
- `use_all_available_gpus` : When this option is present, all GPUs in the system will be used. Alternatively, use `tensor_parallel_size : "x"`, to use only `x` GPUS. By default, i.e. if neither of these options are specified, the system uses only 1 GPU.
- `max_model_lens` : The sequence-lengths to benchmark with. This is the maximum number of tokens the system can process.
- `sparsity` : Use this option to specify model sparsities.
- `script-name` : Name of the benchmarking script to run.
- `script-args` : This option is a JSON in itself. The keys in the JSON are the argument names that the script takes. The values of each of the keys is a list of values to benchmark with. The driver, enumerates all the combinations of these argument and invokes the benchmarking script once with each set of arguments. The `script-args` keys can be found with `python3 -m neuralmagic.benchmarks.scripts.benchmark_serving --help`

## Benchmark Throughput Config
from `configs/benchmark_throughput.py`
```
{
	"configs": [
		{
			"description": "VLLM Engine throughput - Dense (with dataset)",
			"models": [
				"teknium/OpenHermes-2.5-Mistral-7B",
				"neuralmagic/OpenHermes-2.5-Mistral-7B-marlin",
				"TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ",
				"NousResearch/Llama-2-7b-chat-hf"
			],
			"max_model_lens": [
				4096
			],
			"script_name": "benchmark_throughput",
			"script_args": {
				"dataset": [
					"sharegpt"
				],
				"output-len": [
					128
				],
				"num-prompts": [
					1000
				],
				"use-all-available-gpus_": []
			}
		}
	]
}
```
### JSON Breakdown
Most keys in this JSON are similar to the benchmark serving JSON. However, there are some differences. Note that the `use-all-available-gpus_` is in `script-args` instead. This is primarily due to that fact that in the benchmark serving case, the driver spawns the vLLM Engine Server as a separate process and it is the server process that needs this information. In the benchmark throughput case, the vLLM engine is created inside the script.

# Standardized Benchmark Results
The benchmarking scripts `scripts/benchmark_serving.py` and `scripts/benchmark_throughput.py` both subscribe to `BenchmarkResult` in `scripts/logging/benchmark_result.py` for logging JSON results. This is so the output JSONs across all benchmarking scripts have a standard JSON format. Reference <a href=https://github.com/neuralmagic/nm-vllm/blob/main/neuralmagic/benchmarks/scripts/logging/benchmark_result.py>benchmark_result.py</a>

# Github Benchmark Run
The JSON configs to run scheduled/event-triggered benchmarks can be found at `<nm-vllm-root>/.github/data/`

# How to run
### Steps to run:
   - **Locally**:
    - `cd <nm-vllm-root>`
	- `python3 -m neuralmagic.benchmarks.run_benchmarks -i <path-to-config-file> -o <path-to-output-dir>`

   - **GHA Action**:
    - `echo '{"label":"<machine-label>", "timeout":"<max-time-machine-should-run>", "gitref":"<github-branch>", "benchmark_config_list_file":"<path-to-txt-file-listing-config-file-paths>", "python" : "<python-version>", "Gi_per_thread" : "<num-gb-per-thread-for-building-nm-vllm>", "nvcc_threads" : "<num-nvcc-build-threads-to-use>" }' | gh workflow run nm-benchmark.yml --ref <github-branch> --json`
	 - Example `echo '{"label":"aws-avx2-32G-a10g-24G", "timeout":"180", "gitref":"varun/flakiness-experiments", "benchmark_config_list_file":".github/data/nm_benchmark_configs_minimal_test_list.txt", "python" : "3.10.12", "Gi_per_thread" : "12", "nvcc_threads" : "1"}' | gh workflow run nm-benchmark.yml --ref varun/flakiness-experiments --json`
	 
# How To Add A Benchmark

There are 5 Steps in making a Benchmark.
### The Script
Write your metrics generation script.

### The Driver 
Write a driver that invokes the script as a subprocess. Refer to the interaction between `scripts/benchmark_serving.py` and `run_benchmark_serving.py`.

### Register Driver
Register the Driver with `run_benchmarks.py` so there is just one command to run any config file. Refer to the interaction between `run_benchmark_serving.py` and `run_benchmarks.py`

### Register Script Metrics
For the purposes of logging and keeping the benchmark result JSON dump standardized, register the script metrics with `scripts/logging/benchmark_result.py`. This registration mainly captures the information required for UI and regression triggers. Example, 

```
BenchmarkThroughputResultMetricTemplates = SimpleNamespace(
    request_throughput=MetricTemplate("request_throughput", "prompts/s", None,
                                      GHABenchmarkToolName.BiggerIsBetter),
    token_throughput=MetricTemplate("token_throughput", "tokens/s", None,
                                    GHABenchmarkToolName.BiggerIsBetter))
```
This registers the metrics `request_throughput`, `token_throughput` and adds the metric's unit and also information about which direction (up/down) is considered better in general.

### Creating a BenchmarkResult object for JSON result dump
As an example, the `script/benchmark_throughput.py` creates the BenchmarkResult as follows,
```
   result = BenchmarkResult(
       description=args.description,
       date=current_dt,
       script_name=Path(__file__).name,
       script_args=vars(args),
       tensor_parallel_size=get_tensor_parallel_size(args),
       model=args.model,
       tokenizer=args.tokenizer,
       dataset=args.dataset)

   result.add_metric(ResultMetricTemplates.request_throughput,
                     request_throughput)
   result.add_metric(ResultMetricTemplates.token_throughput,
                     token_throughput)

   result.store("./output.json")
```
BenchmarkResult constructor takes all the general information about the environment. Then, invoke the `add_metric` method to add the observed metrics. Finally, store the benchmark results as a JSON by invoking the `store` method.

# Permanent Storage
As of 2024-03-28, all benchmark run results are stored on EFS. For access, please reach out to Andy Linfoot / Varun / Dan Huang.

# About sparsity
The benchmark configs have a `sparsity` field. Populate this field with proper sparsity identifiers to inform vllm about model sparsity.
For the list of valid sparsity args, check `vllm/model_executor/layers/sparsity/*`
