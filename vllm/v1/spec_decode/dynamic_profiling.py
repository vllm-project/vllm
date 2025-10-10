import re
import os
import json
import argparse
from vllm.v1.spec_decode.online_profiling_client import (NGRAM_FMT, EAGLE_FMT)


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
            if method == "ngram":
                FMT = NGRAM_FMT.replace("{}", "(.+)")
                min, max, k = re.match(FMT, spec_config_str).groups()
            elif method == "eagle":
                FMT = EAGLE_FMT.replace("{}", "(.+)")
                k = re.match(FMT, spec_config_str).groups()[0]

            # read the log file to get the itl
            with open(os.path.join(args.benchmark_path, log_file), "r") as f:
                data = json.load(f)
                itl = data["median_itl_ms"]

            # add to batch_stats
            if int(bs) not in batch_stats:
                batch_stats[int(bs)] = {}
            
            batch_stats[int(bs)][int(k)] = itl

    return batch_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-method", type=str, default=None)
    parser.add_argument("--benchmark-path-parent", type=str, default=None, help="Root folder which has the log files")

    args = parser.parse_args()
    assert args.sd_method in ["ngram", "eagle"], "Invalid method specified."

    batch_stats = parse_itl(args)