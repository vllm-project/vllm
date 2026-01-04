# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import shlex
from importlib import util
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import regex as re
from tabulate import tabulate

# latency results and the keys that will be printed into markdown
latency_results = []
latency_column_mapping = {
    "test_name": "Test name",
    "gpu_type": "GPU",
    "avg_latency": "Mean latency (ms)",
    # "P10": "P10 (s)",
    # "P25": "P25 (s)",
    "P50": "Median latency (ms)",
    # "P75": "P75 (s)",
    # "P90": "P90 (s)",
    "P99": "P99 latency (ms)",
}

# throughput tests and the keys that will be printed into markdown
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
    "model_id": "Model",
    "dataset_name": "Dataset Name",
    "input_len": "Input Len",
    "output_len": "Output Len",
    "tp_size": "TP Size",
    "pp_size": "PP Size",
    "dtype": "dtype",
    "gpu_type": "GPU",
    "completed": "# of req.",
    "qps": "qps",
    "max_concurrency": "# of max concurrency.",
    "request_throughput": "Tput (req/s)",
    "total_token_throughput": "Total Token Tput (tok/s)",
    "output_throughput": "Output Tput (tok/s)",
    # "total_input_tokens": "Total input tokens",
    # "total_output_tokens": "Total output tokens",
    "mean_ttft_ms": "Mean TTFT (ms)",
    "median_ttft_ms": "Median TTFT (ms)",
    "p99_ttft_ms": "P99 TTFT (ms)",
    "std_ttft_ms": "STD TTFT (ms)",
    "mean_tpot_ms": "Mean TPOT (ms)",
    "median_tpot_ms": "Median",
    "p99_tpot_ms": "P99",
    "std_tpot_ms": "STD TPOT (ms)",
    "mean_itl_ms": "Mean ITL (ms)",
    "median_itl_ms": "Median ITL (ms)",
    "p99_itl_ms": "P99 ITL (ms)",
}


def read_markdown(file):
    if os.path.exists(file):
        with open(file) as f:
            return f.read() + "\n"
    else:
        return f"{file} not found.\n"


def results_to_json(latency, throughput, serving):
    return json.dumps(
        {
            "latency": latency.to_dict(),
            "throughput": throughput.to_dict(),
            "serving": serving.to_dict(),
        }
    )


def get_size_with_unit(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def _coerce(val: str) -> Any:
    """Best-effort type coercion from string to Python types."""
    low = val.lower()
    if low == "null":
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    # integers
    if re.fullmatch(r"[+-]?\d+", val):
        try:
            return int(val)
        except ValueError:
            pass
    # floats (keep 'inf'/'-inf'/'nan' as strings)
    if re.fullmatch(r"[+-]?\d*\.\d+", val):
        try:
            return float(val)
        except ValueError:
            pass
    return val


def parse_client_command(cmd: str) -> dict[str, Any]:
    """Parse the client_command shell string into {executable, script, args}."""
    toks = shlex.split(cmd)
    if len(toks) < 2:
        raise ValueError("client_command must include an executable and a script")
    executable, script = toks[0], toks[1]
    args: dict[str, Any] = {}

    i = 2
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            # --key=value or --key (value) or boolean flag
            if "=" in t:
                key, val = t.split("=", 1)
                if key == "--metadata":
                    md = {}
                    if val:
                        if "=" in val:
                            k, v = val.split("=", 1)
                            md[k] = _coerce(v)
                        else:
                            md[val] = True
                    args[key] = md
                else:
                    args[key] = _coerce(val)
                i += 1
                continue

            key = t

            # Special: consume metadata k=v pairs until next --flag
            if key == "--metadata":
                i += 1
                md = {}
                while i < len(toks) and not toks[i].startswith("--"):
                    pair = toks[i]
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        md[k] = _coerce(v)
                    else:
                        md[pair] = True
                    i += 1
                args[key] = md
                continue

            # Standard: check if next token is a value (not a flag)
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                args[key] = _coerce(toks[i + 1])
                i += 2
            else:
                # lone flag -> True
                args[key] = True
                i += 1
        else:
            # unexpected positional; skip
            i += 1

    return {"executable": executable, "script": script, "args": args}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--result",
        type=str,
        default="results",
        help="Folder name for benchmark output results.",
    )
    args = parser.parse_args()
    results_folder = Path(args.result)
    if not results_folder.exists():
        raise FileNotFoundError(f"results folder does not exist: {results_folder}")
    # collect results
    for test_file in results_folder.glob("*.json"):
        with open(test_file) as f:
            raw_result = json.loads(f.read())

        if "serving" in str(test_file):
            # this result is generated via `vllm bench serve` command
            # attach the benchmarking command to raw_result
            try:
                with open(test_file.with_suffix(".commands")) as f:
                    command = json.loads(f.read())
            except OSError as e:
                print(e)
                continue
            # Parse Server Command Arg
            out: dict[str, Any] = {
                "server_command": parse_client_command(command["server_command"])
            }
            parse_args = [
                "--tensor-parallel-size",
                "--pipeline-parallel-size",
                "--dtype",
            ]
            col_mapping = ["tp_size", "pp_size", "dtype"]
            for index, arg in enumerate(parse_args):
                if arg in out["server_command"]["args"]:
                    raw_result.update(
                        {col_mapping[index]: out["server_command"]["args"][arg]}
                    )

            # Parse Client Command Arg
            out: dict[str, Any] = {
                "client_command": parse_client_command(command["client_command"])
            }
            parse_args = [
                "--dataset-name",
                "--random-input-len",
                "--random-output-len",
                "--request-rate",
            ]
            col_mapping = ["dataset_name", "input_len", "output_len", "qps"]

            for index, arg in enumerate(parse_args):
                if arg in out["client_command"]["args"]:
                    raw_result.update(
                        {col_mapping[index]: out["client_command"]["args"][arg]}
                    )
            # Add Server, Client command
            raw_result.update(command)

            # update the test name of this result
            raw_result.update({"test_name": test_file.stem})
            # add the result to raw_result
            serving_results.append(raw_result)
            continue

        elif "latency" in f.name:
            # this result is generated via `vllm bench latency` command

            # attach the benchmarking command to raw_result
            try:
                with open(test_file.with_suffix(".commands")) as f:
                    command = json.loads(f.read())
            except OSError as e:
                print(e)
                continue

            raw_result.update(command)

            # update the test name of this result
            raw_result.update({"test_name": test_file.stem})

            # get different percentiles
            for perc in [10, 25, 50, 75, 90, 99]:
                # Multiply 1000 to convert the time unit from s to ms
                raw_result.update(
                    {f"P{perc}": 1000 * raw_result["percentiles"][str(perc)]}
                )
            raw_result["avg_latency"] = raw_result["avg_latency"] * 1000

            # add the result to raw_result
            latency_results.append(raw_result)
            continue

        elif "throughput" in f.name:
            # this result is generated via `vllm bench throughput` command

            # attach the benchmarking command to raw_result
            try:
                with open(test_file.with_suffix(".commands")) as f:
                    command = json.loads(f.read())
            except OSError as e:
                print(e)
                continue

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

    svmem = psutil.virtual_memory()
    platform_data = {
        "Physical cores": [psutil.cpu_count(logical=False)],
        "Total cores": [psutil.cpu_count(logical=True)],
        "Total Memory": [get_size_with_unit(svmem.total)],
    }

    if util.find_spec("numa") is not None:
        from numa import info

        platform_data["Total NUMA nodes"] = [info.get_num_configured_nodes()]

    if util.find_spec("cpuinfo") is not None:
        from cpuinfo import get_cpu_info

        platform_data["CPU Brand"] = [get_cpu_info()["brand_raw"]]

    platform_results = pd.DataFrame.from_dict(
        platform_data, orient="index", columns=["Platform Info"]
    )

    raw_results_json = results_to_json(
        latency_results, throughput_results, serving_results
    )

    # remapping the key, for visualization purpose
    if not latency_results.empty:
        latency_results = latency_results[list(latency_column_mapping.keys())].rename(
            columns=latency_column_mapping
        )
    if not serving_results.empty:
        valid_columns = [
            col for col in serving_column_mapping if col in serving_results.columns
        ]
        serving_results = serving_results[valid_columns].rename(
            columns=serving_column_mapping
        )
    if not throughput_results.empty:
        throughput_results = throughput_results[
            list(throughput_results_column_mapping.keys())
        ].rename(columns=throughput_results_column_mapping)

    processed_results_json = results_to_json(
        latency_results, throughput_results, serving_results
    )

    for df in [latency_results, serving_results, throughput_results]:
        if df.empty:
            continue

        # Sort all dataframes by their respective "Test name" columns
        df.sort_values(by="Test name", inplace=True)

        # The GPUs sometimes come in format of "GPUTYPE\nGPUTYPE\n...",
        # we want to turn it into "8xGPUTYPE"
        df["GPU"] = df["GPU"].apply(
            lambda x: "{}x{}".format(len(x.split("\n")), x.split("\n")[0])
        )

    # get markdown tables
    latency_md_table = tabulate(
        latency_results, headers="keys", tablefmt="pipe", showindex=False
    )
    serving_md_table = tabulate(
        serving_results, headers="keys", tablefmt="pipe", showindex=False
    )
    throughput_md_table = tabulate(
        throughput_results, headers="keys", tablefmt="pipe", showindex=False
    )
    platform_md_table = tabulate(
        platform_results, headers="keys", tablefmt="pipe", showindex=True
    )

    # document the result
    md_file = "benchmark_results.md"
    json_file = "benchmark_results.json"
    with open(results_folder / md_file, "w") as f:
        results = read_markdown(
            "../.buildkite/performance-benchmarks/"
            + "performance-benchmarks-descriptions.md"
        )
        results = results.format(
            latency_tests_markdown_table=latency_md_table,
            throughput_tests_markdown_table=throughput_md_table,
            serving_tests_markdown_table=serving_md_table,
            platform_markdown_table=platform_md_table,
            benchmarking_results_in_json_string=processed_results_json,
        )
        f.write(results)

    # document benchmarking results in json
    with open(results_folder / json_file, "w") as f:
        results = (
            latency_results.to_dict(orient="records")
            + throughput_results.to_dict(orient="records")
            + serving_results.to_dict(orient="records")
        )
        f.write(json.dumps(results))
