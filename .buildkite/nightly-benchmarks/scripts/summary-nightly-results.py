import datetime
import json
import os
from pathlib import Path

import pandas as pd
from tabulate import tabulate

results_folder = Path("results/")

# serving results and the keys that will be printed into markdown
serving_results = []
serving_column_mapping = {
    "test_name": "Test name",
    "gpu_type": "GPU",
    "completed": "Successful req.",
    "request_throughput": "Tput (req/s)",
    "mean_ttft_ms": "Mean TTFT (ms)",
    "std_ttft_ms": "Std TTFT (ms)",
    "median_ttft_ms": "Median TTFT (ms)",
    "mean_itl_ms": "Mean ITL (ms)",
    "std_itl_ms": "Std ITL (ms)",
    "median_itl_ms": "Median ITL (ms)",
    "mean_tpot_ms": "Mean TPOT (ms)",
    "std_tpot_ms": "Std TPOT (ms)",
    "median_tpot_ms": "Median TPOT (ms)",
    "total_token_throughput": "Total Token Tput (tok/s)",
    "output_throughput": "Output Tput (tok/s)",
    "total_input_tokens": "Total input tokens",
    "total_output_tokens": "Total output tokens",
    "engine": "Engine",
}

if __name__ == "__main__":

    # collect results
    for test_file in results_folder.glob("*.json"):

        with open(test_file) as f:
            raw_result = json.loads(f.read())

        # attach the benchmarking command to raw_result
        with open(test_file.with_suffix(".commands")) as f:
            command = json.loads(f.read())
        raw_result.update(command)

        # update the test name of this result
        raw_result.update({"test_name": test_file.stem})

        # add the result to raw_result
        serving_results.append(raw_result)
        continue

    serving_results = pd.DataFrame.from_dict(serving_results)

    if not serving_results.empty:
        serving_results = serving_results[list(
            serving_column_mapping.keys())].rename(
                columns=serving_column_mapping)

    serving_md_table_with_headers = tabulate(serving_results,
                                             headers='keys',
                                             tablefmt='pipe',
                                             showindex=False)
    # remove the first line of header
    serving_md_table_lines = serving_md_table_with_headers.split('\n')
    serving_md_table_without_header = '\n'.join(serving_md_table_lines[2:])

    prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = prefix + "_" + os.environ.get("CURRENT_LLM_SERVING_ENGINE")

    # document benchmarking results in markdown
    with open(results_folder / f"{prefix}_nightly_results.md", "w") as f:
        # document results with header.
        # for those who wants to reproduce our benchmark.
        f.write(serving_md_table_with_headers)
        f.write('\n')

    # document benchmarking results in json
    with open(results_folder / f"{prefix}_nightly_results.json", "w") as f:

        results = serving_results.to_dict(orient='records')
        f.write(json.dumps(results))
