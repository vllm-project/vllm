# Directory Structure:

- scripts/*.py - Benchmark scripts that perform the metric computation.

- configs/*.json - Config JSON files. These JSONs define what benchmark script to run and what combination of script parameters to use. 

- *.py - Benchmark drivers. Given a config JSON, executes all the commands defined by the config JSON.

# Run Benchmark scripts

All `scripts/benchmark_*.py` files can be executed on their own.

Run `python -m neuralmagic/benchmarks/scripts/* --help` for script description and How-To run.

# Benchmarking drivers and Configs

All the benchmark driver *.py files, input a JSON config file and an output directory path.

As mentioned above, the config file defines what benchmark-script to run and what arguments to run it with.

The following is an example config JSON,

```
		{
			"description": "Benchmark vllm engine throughput - with dataset",
			"models": [
				"facebook/opt-125m",
				"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			],
			"sparsity" : [],
			"script_name": "benchmark_throughput",
			"script_args": {
				"dataset": [
					"sharegpt",
                    "ultrachat"
				],
				"output-len": [
					128
				],
				"num-prompts": [
					1000
				],
			}
		}
```
This config tells the benchmark driver to run benchmark_throughput script on all the listed models with all possible script-args combinations.
i.e. the config essentially translates to,

python -m neuralmagic.benchmarks.benchmark_throughput.py --model facebook/opt-125m --dataset sharegpt --output-len 128 --num-prompts 1000

python -m neuralmagic.benchmarks.benchmark_throughput.py --model facebook/opt-125m --dataset ultrachat --output-len 128 --num-prompts 1000

python -m neuralmagic.benchmarks.benchmark_throughput.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset sharegpt --output-len 128 --num-prompts 1000

python -m neuralmagic.benchmarks.benchmark_throughput.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset ultrachat --output-len 128 --num-prompts 1000

# Benchmarking with driver
```
python3 -m neuralmagic.benchmarks.run_benchmarks -i <path-to-config-file> -o <output-directory-path>
```

# About sparsity
The benchmark configs have a `sparsity` field. Populate this field with proper sparsity identifiers to inform vllm about model sparsity.
For the list of valid sparsity args, check `vllm/model_executor/layers/sparsity/*`
