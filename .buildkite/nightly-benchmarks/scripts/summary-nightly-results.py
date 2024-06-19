import json
import os
from pathlib import Path

import pandas as pd

results_folder = Path("results/")

# serving results and the keys that will be printed into markdown
serving_results = []
serving_column_mapping = {
    "test_name": "Test name",
    "gpu_type": "GPU",
    "completed": "Successful req.",
    "request_throughput": "Tput (req/s)",
    # "input_throughput": "Input Tput (tok/s)",
    # "output_throughput": "Output Tput (tok/s)",
    "mean_ttft_ms": "Mean TTFT (ms)",
    "median_ttft_ms": "Median TTFT (ms)",
    "p99_ttft_ms": "P99 TTFT (ms)",
    # "mean_tpot_ms": "Mean TPOT (ms)",
    # "median_tpot_ms": "Median",
    # "p99_tpot_ms": "P99",
    "mean_itl_ms": "Mean ITL (ms)",
    "median_itl_ms": "Median ITL (ms)",
    "p99_itl_ms": "P99 ITL (ms)",
    "engine": "Engine",
}




if __name__ == "__main__":

    # collect results
    for test_file in results_folder.glob("*.json"):

        with open(test_file, "r") as f:
            raw_result = json.loads(f.read())

            
        # attach the benchmarking command to raw_result
        with open(test_file.with_suffix(".commands"), "r") as f:
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
            
            
    prefix = os.environ.get("CURRENT_LLM_SERVING_ENGINE")

    # document benchmarking results in json
    with open(results_folder / f"{prefix}_nightly_results.json", "w") as f:

        results = serving_results.to_dict(orient='records')
        f.write(json.dumps(results))
