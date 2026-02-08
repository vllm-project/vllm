# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os
import re


# Format strings for speculative config naming in benchmark result files.
NGRAM_FMT = "min-{min}-max-{max}-k-{k}"
EAGLE_FMT = "k-{k}"


def reverse_fmt(fmt_str):
    # e.g., convert 'min-{min}-max-{max}-k-{k}' -> 'min-{}-max-{}-k-{}'
    FMT = re.sub(r"\{[^}]+\}", "{}", fmt_str)
    # e.g., convert 'min-{}-max-{}-k-{}' -> 'min-(.+)-max-(.+)-k-(.+)'
    FMT = FMT.replace("{}", "(.+)")
    return FMT


NGRAM_FMT_REVERSE = reverse_fmt(NGRAM_FMT)
EAGLE_FMT_REVERSE = reverse_fmt(EAGLE_FMT)


def parse_itl(method, benchmark_path_parent):
    """
    DynamicSpeculativeConfig.batch_stats: dict
    The structure is as follows:
    {
        batch_size: {
            num_drafts: itl (i.e., inter token latency in ms)
        }
    }
    """
    batch_stats = {}

    for method in ["vanilla", method]:
        # find the names of all log files in this folder
        benchmark_path = os.path.join(benchmark_path_parent, method)
        all_log_files = [
            f
            for f in os.listdir(benchmark_path)
            if os.path.isfile(os.path.join(benchmark_path, f))
            and f.endswith(".txt")
        ]

        # parse the log files to get the config params
        for log_file in all_log_files:
            # find bs
            bs = re.search(r"_bs-(\d+)", log_file).group(1)

            # find sd params
            spec_config_str = log_file.split("_")[0]
            if method == "vanilla":
                k = 0
            elif method == "ngram":
                min, max, k = re.match(NGRAM_FMT_REVERSE, spec_config_str).groups()
            elif method == "eagle":
                k = re.match(EAGLE_FMT_REVERSE, spec_config_str).groups()[0]

            # read the log file to get the itl
            with open(os.path.join(benchmark_path, log_file)) as f:
                data = json.load(f)
                itl = data["median_itl_ms"]

            # add to batch_stats
            if int(bs) not in batch_stats:
                batch_stats[int(bs)] = {}

            batch_stats[int(bs)][int(k)] = itl

    print(json.dumps(batch_stats, indent=4))
    return batch_stats


"""
python3 vllm/v1/spec_decode/process_benchmark_results.py \
    --method eagle \
    --benchmark-path-parent 'log/dynamic_sd/tp-1_temp-0_top_p-1/philschmid/mt-bench/'
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument(
        "--benchmark-path-parent",
        type=str,
        default=None,
        help="Root folder which has the log files",
    )

    args = parser.parse_args()
    assert args.method in ["ngram", "eagle", "eagle3", "mtp"], "Invalid method specified."

    batch_stats = parse_itl(method=args.method, 
                            benchmark_path_parent=args.benchmark_path_parent)
