
## Latency tests

This test suite aims to test vllm's end-to-end latency under a controlled setup.

- Input length: 32 tokens.
- Output length: 128 tokens.
- Batch size: fixed (8).
- Models: llama-3 8B, llama-3 70B, mixtral 8x7B.
- Evaluation metrics: end-to-end latency (mean, median, p99).

### Latency benchmarking results

{latency_tests_markdown_table}

## Throughput tests

This test suite aims to test vllm's throughput.

- Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm to achieve maximum throughput.
- Models: llama-3 8B, llama-3 70B, mixtral 8x7B.
- Evaluation metrics: throughput.

### Throughput benchmarking results

{throughput_tests_markdown_table}

## Serving tests

This test suite aims to test vllm's real serving metrics.

- Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 200 prompts.
- Batch size: dynamically determined by vllm and the arrival pattern of the requests.
- **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
- Models: llama-3 8B, llama-3 70B, mixtral 8x7B.
- Evaluation metrics: throughput, TTFT (time to the first token, with mean, median and p99), ITL (inter-token latency, with mean, median and p99).

### Serving benchmarking results

{serving_tests_markdown_table}

## json version of the benchmarking tables

This section contains the data of the markdown tables above in JSON format. 
You can load the benchmarking tables into pandas dataframes as follows:

```python
import json
import pandas as pd

benchmarking_results_json = """The json string"""
benchmarking_results = json.loads(benchmarking_results_json)
latency_results = pd.DataFrame.from_dict(benchmarking_results["latency"])
throughput_results = pd.DataFrame.from_dict(benchmarking_results["throughput"])
serving_results = pd.DataFrame.from_dict(benchmarking_results["serving"])
```

The json string for all benchmarking tables:
```json
{benchmarking_results_in_json_string}
```

You can also check the raw experiment data in the Artifact tab of the Buildkite page.

