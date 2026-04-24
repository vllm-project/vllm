# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for benchmarking")
    parser.add_argument(
        "--data_dir", type=str, default="/data", help="Path to the dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/output", help="Path to the output directory"
    )
    return parser.parse_args()


def cohere_dataset(args):
    jsonInput = pd.read_json(path_or_buf=args.data_dir, lines=True)
    filename = os.path.basename(args.data_dir).split(".")[0]

    # save jsonOutput to args.output_dir
    output_folder = os.path.join(args.output_dir, filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # write jsonOutput to file
    output_file = os.path.join(output_folder, "question.jsonl")
    with open(output_file, "w") as f:
        for item in jsonInput["prompt"]:
            output_item = {"turns": [item]}
            f.write(json.dumps(output_item) + "\n")

    print(f"Dataset prepared and saved to {output_file}")


def main():
    args = parse_args()
    cohere_dataset(args)


# USAGE:
# python3 vllm/cohere/utils/eagle/benchmark_prepare_dataset.py \
#     --data_dir="data/code.jsonl" --output_dir="data"
if __name__ == "__main__":
    main()
