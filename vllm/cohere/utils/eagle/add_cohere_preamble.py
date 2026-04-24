# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for benchmarking")
    parser.add_argument(
        "--input_file", type=str, default="/data", help="Path to the dataset"
    )
    return parser.parse_args()


# load a jsonl file
def load_jsonl(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def format_prompt(prompt):
    new_prompt = (
        "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
        "You are Coral, a brilliant, sophisticated, AI-assistant chatbot "
        "trained to assist human users by providing thorough responses. "
        "You are powered by Command, a large language model built by the "
        "company Cohere. Today's date is Saturday, November 23, 2024."
        "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
    )
    new_prompt += prompt
    new_prompt += "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    return new_prompt


def main(args):
    # load a jsonl file
    src_file_path = args.input_file
    filename = os.path.basename(src_file_path).split(".")[0]
    dirname = os.path.dirname(src_file_path)
    import pdb

    pdb.set_trace()
    # output folder
    dst_file_path = os.path.join(dirname, f"{filename}-cohere.jsonl")
    src_data = load_jsonl(src_file_path)
    dst_data = []
    for src_d in src_data:
        dst_d = {"prompt": format_prompt(src_d["turns"][0])}
        dst_data.append(dst_d)

    save_jsonl(dst_data, dst_file_path)
    print(f"Dataset prepared and saved to {dst_file_path}")


# USAGE:
# python3 vllm/cohere/utils/eagle/add_cohere_preamble.py \
#     --input_file="data/mt_bench/question.jsonl"
if __name__ == "__main__":
    args = parse_args()
    main(args)
