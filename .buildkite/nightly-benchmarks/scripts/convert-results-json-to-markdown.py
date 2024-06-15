import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate

results_folder = Path("results/")

# latency results and the keys that will be printed into markdown
latency_results = []
latency_column_mapping = {
    "test_name": "Test name",
    "gpu_type": "GPU",
    "avg_latency": "Average latency (s)",
    "P10": "P10 (s)",
    "P25": "P25 (s)",
    "P50": "P50 (s)",
    "P75": "P75 (s)",
    "P90": "P90 (s)",
}

# thoughput tests and the keys that will be printed into markdown
throughput_results = []
throughput_results_column_mapping = {
    "test_name": "Test name",
    "gpu_type": "GPU",
    "num_requests": "# of req.",
    "total_num_tokens": "Total # of tokens",
    "elapsed_time": "Elapsed time (s)",
    "requests_per_second": "Tput (req/s)",
    "tokens_per_second": "Tput (tok/s)",
}

# serving results and the keys that will be printed into markdown
serving_results = []
serving_column_mapping = {
    "test_name": "Test name",
    "gpu_type": "GPU",
    "completed": "# of req.",
    "request_throughput": "Tput (req/s)",
    "input_throughput": "Input Tput (tok/s)",
    "output_throughput": "Output Tput (tok/s)",
    "mean_ttft_ms": "Mean TTFT (ms)",
    # do not say TTFT again to avoid the table getting too wide
    "median_ttft_ms": "Median",
    "p99_ttft_ms": "P99",
    "mean_tpot_ms": "Mean TPOT (ms)",
    "median_tpot_ms": "Median",
    "p99_tpot_ms": "P99",
    "mean_itl_ms": "Mean ITL (ms)",
    "median_itl_ms": "Median",
    "p99_itl_ms": "P99",
}

for test_file in results_folder.glob("*.json"):

    with open(test_file, "r") as f:
        raw_result = json.loads(f.read())

    if "serving" in str(test_file):
        # this result is generated via `benchmark_serving.py`

        # attach the benchmarking command to raw_result
        with open(test_file.with_suffix(".commands"), "r") as f:
            command = json.loads(f.read())
        raw_result.update(command)

        # update the test name of this result
        raw_result.update({"test_name": test_file.stem})

        # add the result to raw_result
        serving_results.append(raw_result)
        continue

    elif "latency" in f.name:
        # this result is generated via `benchmark_latency.py`

        # attach the benchmarking command to raw_result
        with open(test_file.with_suffix(".commands"), "r") as f:
            command = json.loads(f.read())
        raw_result.update(command)

        # update the test name of this result
        raw_result.update({"test_name": test_file.stem})

        # get different percentiles
        for perc in [10, 25, 50, 75, 90]:
            raw_result.update(
                {f"P{perc}": raw_result["percentiles"][str(perc)]})

        # add the result to raw_result
        latency_results.append(raw_result)
        continue

    elif "throughput" in f.name:
        # this result is generated via `benchmark_throughput.py`

        # attach the benchmarking command to raw_result
        with open(test_file.with_suffix(".commands"), "r") as f:
            command = json.loads(f.read())
        raw_result.update(command)

        # update the test name of this result
        raw_result.update({"test_name": test_file.stem})

        # add the result to raw_result
        throughput_results.append(raw_result)
        continue

    print(f"Skipping {test_file}")

latency_results = pd.DataFrame.from_dict(latency_results)
serving_results = pd.DataFrame.from_dict(serving_results)
throughput_results = pd.DataFrame.from_dict(throughput_results)

# remapping the key, for visualization purpose
if not latency_results.empty:
    latency_results = latency_results[list(
        latency_column_mapping.keys())].rename(columns=latency_column_mapping)
if not serving_results.empty:
    serving_results = serving_results[list(
        serving_column_mapping.keys())].rename(columns=serving_column_mapping)
if not throughput_results.empty:
    throughput_results = throughput_results[list(
        throughput_results_column_mapping.keys())].rename(
            columns=throughput_results_column_mapping)

# get markdown tables
latency_md_table = tabulate(latency_results,
                            headers='keys',
                            tablefmt='pipe',
                            showindex=False)
serving_md_table = tabulate(serving_results,
                            headers='keys',
                            tablefmt='pipe',
                            showindex=False)
throughput_md_table = tabulate(throughput_results,
                               headers='keys',
                               tablefmt='pipe',
                               showindex=False)

# document the result
with open(results_folder / "benchmark_results.md", "w") as f:
    if not latency_results.empty:
        f.write("## Latency tests\n")
        f.write(latency_md_table)
        f.write("\n")
    if not throughput_results.empty:
        f.write("## Throughput tests\n")
        f.write(throughput_md_table)
        f.write("\n")
    if not serving_results.empty:
        f.write("## Serving tests\n")
        f.write(serving_md_table)
        f.write("\n")
