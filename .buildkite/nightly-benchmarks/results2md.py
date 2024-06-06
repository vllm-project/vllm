
import json
import pandas as pd
from pathlib import Path
from tabulate import tabulate

results_folder = Path("results/")

# serving results and the keys that will be printed into markdown
serving_results = []
serving_column_mapping = {
    "test_name": "Test name",
    "completed": "# of req.",
    "request_throughput": "Tput (req/s)",
    "mean_ttft_ms": "Mean TTFT (ms)",
    "median_ttft_ms": "Median (ms)",
    "p99_ttft_ms": "P99 (ms)",
    "mean_tpot_ms": "Mean TPOT (ms)",
    "median_tpot_ms": "Median (ms)",
    "p99_tpot_ms": "P99 (ms)",
}

for test_file in results_folder.glob(f"*.json"):
    
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
        continue
    
    print(f"Skipping {test_file}")

serving_results = pd.DataFrame.from_dict(serving_results)

# Remapping the keys according to serving_column_mapping
serving_results = serving_results[
    list(serving_column_mapping.keys())
].rename(columns=serving_column_mapping)

# get markdown table for serving
serving_md_table = tabulate(
    serving_results, 
    headers='keys', 
    tablefmt='pipe',
    showindex=False)

# document the result
with open(results_folder / "benchmark_results.md", "w") as f: 
    f.write("## Online serving tests\n")
    f.write(serving_md_table)
    f.write("\n")
