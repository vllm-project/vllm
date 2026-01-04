import re
import os
import json
import argparse
from vllm.v1.spec_decode.dynamic.online_profiling_client import (NGRAM_FMT, EAGLE_FMT)


def reverse_fmt(fmt_str):
    # e.g., convert 'min-{min}-max-{max}-k-{k}' -> 'min-{}-max-{}-k-{}'
    FMT = re.sub(r"\{[^}]+\}", "{}", fmt_str)
    # e.g., convert 'min-{}-max-{}-k-{}' -> 'min-(.+)-max-(.+)-k-(.+)'
    FMT = FMT.replace("{}", "(.+)")
    return FMT

NGRAM_FMT_REVERSE = reverse_fmt(NGRAM_FMT)
EAGLE_FMT_REVERSE = reverse_fmt(EAGLE_FMT)


def parse_itl(args):
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

    for method in ["vanilla", args.sd_method]:
        # find the names of all log files in this folder
        args.benchmark_path = os.path.join(args.benchmark_path_parent, method)
        all_log_files = [f for f in os.listdir(args.benchmark_path) \
                        if os.path.isfile(os.path.join(args.benchmark_path, f)) \
                            and f.endswith(".txt")]
    
        # parse the log files to get the config params
        for log_file in all_log_files:
            # find bs
            bs = re.search(r'_bs-(\d+)', log_file).group(1)
            
            # find sd params
            spec_config_str = log_file.split("_")[0]
            if method == "vanilla":
                k=0
            elif method == "ngram":
                min, max, k = re.match(NGRAM_FMT_REVERSE, spec_config_str).groups()
            elif method == "eagle":
                k = re.match(EAGLE_FMT_REVERSE, spec_config_str).groups()[0]

            # read the log file to get the itl
            with open(os.path.join(args.benchmark_path, log_file), "r") as f:
                data = json.load(f)
                itl = data["median_itl_ms"]

            # add to batch_stats
            if int(bs) not in batch_stats:
                batch_stats[int(bs)] = {}
            
            batch_stats[int(bs)][int(k)] = itl

    return batch_stats

"""
python3 vllm/v1/spec_decode/process_benchmark_results.py \
    --sd-method eagle \
    --benchmark-path-parent 'log/dynamic_sd/tp-1_temp-0_top_p-1/philschmid/mt-bench/'
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-method", type=str, default=None)
    parser.add_argument("--benchmark-path-parent", type=str, default=None, help="Root folder which has the log files")

    args = parser.parse_args()
    assert args.sd_method in ["ngram", "eagle"], "Invalid method specified."

    batch_stats = parse_itl(args)
    print(json.dumps(batch_stats, indent=4))